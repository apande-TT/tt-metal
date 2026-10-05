# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on a decode step's FFN down projection `gated @ W_down` (32 x 9216 x 3072),
through ttnn.generic_op: `cpp_down`'s kernels (tt/cpp_down_kernels) on the stock decode recipe --
bf8_b weight, bf16 activation, HiFi4 with fp32 DEST, float32 out for the residual add.

The weight is width-sharded over the DRAM banks with each core's PN output columns contiguous in one
bank's shard row (one read per K row); the one-tile-row activation is read from L1 one K block at a
time and the 1 x PN output block stays in DEST across every K block.

On unless VOXTRAL_CPP_DOWN_DEC=0, for the first VOXTRAL_CPP_DOWN_DEC_LAYERS (default: all) layers built.
"""

from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_down_kernels"
_READER_W = str(_DIR / "reader_w.cpp")
_READER_X = str(_DIR / "reader_x_writer.cpp")
_COMPUTE = str(_DIR / "compute.cpp")

_TILE = 32
_CB_BUDGET = 900 * 1024
_DEST_TILES = 8
_KB_DEPTH = ((8, 4), (8, 3), (8, 2), (6, 3), (4, 3), (4, 2), (2, 2), (1, 2))
_BF16 = 2048
_W_DTYPE, _WB = ttnn.bfloat8_b, 1088


def _log(msg):
    path = os.environ.get("VOXTRAL_CPP_SWIGLU_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write("down " + msg + "\n")


def enabled() -> bool:
    # OFF by default since 2026-10-05. Numerically this kernel matches the stock down projection to
    # rounding order (same bf8_b weight, HiFi4, fp32 accumulation; the teacher-forced codes test is
    # green with it), but free-running greedy decode is chaotic, and that rounding put demo row 18
    # into a repeat loop ("discovering very, very, very, very books", 147 frames for 13 words) that the
    # corpus-level WER/MOS margins and the trajectory-aligned PCC gates could not see. Bisected over
    # the optimize run's 84 gated states and confirmed by switching it off on the final code. It is
    # worth 0.23 ms of 89.4 per inference (decode 14.19 -> 14.42 ms). Opt back in to keep working on it.
    return os.environ.get("VOXTRAL_CPP_DOWN_DEC", "0") == "1"


_LAYERS = int(os.environ.get("VOXTRAL_CPP_DOWN_DEC_LAYERS", "1000"))
_built = [0]


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_down_dec: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


class Sharded:
    """w2 as one bank-sharded tensor, plus the core split it was laid out for."""

    def __init__(self, tensor, k, n, pn, ncores, banks):
        self.tensor, self.k, self.n, self.pn, self.ncores, self.banks = tensor, k, n, pn, ncores, banks
        self.shard_tiles = n // _TILE // banks


def shard(w2, device):
    """`w2`: torch `[K, N]` (already transposed). None when the shape has no split."""
    if not enabled() or _built[0] >= _LAYERS:
        return None
    _built[0] += 1
    k, n = int(w2.shape[0]), int(w2.shape[1])
    if k % _TILE or n % _TILE:
        return None
    grid = device.compute_with_storage_grid_size()
    ncores_max = int(grid.x) * int(grid.y)
    banks = int(device.dram_grid_size().x)
    nt = n // _TILE
    want = int(os.environ.get("VOXTRAL_CPP_DOWN_DEC_PN", "3"))
    pn = next(
        (p for p in range(max(1, want), nt + 1) if nt % p == 0 and nt // p <= ncores_max and (nt // p) % banks == 0),
        None,
    )
    _log(f"shard {k}x{n} banks={banks} grid={ncores_max} pn={pn}")
    if pn is None:
        return None
    ncores = nt // pn
    order = [q * banks + s for s in range(banks) for q in range(ncores // banks)]
    t = w2.float().reshape(k, ncores, pn * _TILE)[:, order].reshape(k, n)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            [k, n // banks],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    mapper = ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None
    tt = ttnn.from_torch(
        t.contiguous(), dtype=_W_DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, mesh_mapper=mapper
    )
    return Sharded(tt, k, n, pn, ncores, banks)


class _Plan:
    def __init__(self, device, mt, w):
        grid = device.compute_with_storage_grid_size()
        self.gx = int(grid.x)
        self.mt, self.kt, self.nt = mt, w.k // _TILE, w.n // _TILE
        self.pn, self.ncores = w.pn, w.ncores
        self.ct = max(c for c in range(1, self.pn + 1) if self.pn % c == 0 and mt * c <= _DEST_TILES)
        self.spill = self.ct < self.pn
        fixed = mt * self.pn * 4096 * (2 if self.spill else 1)
        self.kb, self.depth = next(
            (
                (c, d)
                for c, d in _KB_DEPTH
                if self.kt % c == 0 and fixed + 2 * mt * c * _BF16 + d * c * self.pn * _WB <= _CB_BUDGET
            ),
            (None, None),
        )
        if self.kb is None:
            raise RuntimeError(f"cpp_down_dec: no K block for {mt}x{self.kt}x{self.nt}")
        self.nb = self.kt // self.kb
        full, rem = divmod(self.ncores, self.gx)
        ranges = []
        if full:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, full - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
        self.cores = ttnn.CoreRangeSet(ranges)

    def _cb(self, index, fmt, tile_bytes, tiles):
        return ttnn.CBDescriptor(
            total_size=tiles * tile_bytes,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=tile_bytes)],
        )

    def descriptor(self, x, w, y):
        mt, kt, nt, kb, nb, pn = self.mt, self.kt, self.nt, self.kb, self.nb, self.pn
        xa, wa, ya = x.buffer_address(), w.tensor.buffer_address(), y.buffer_address()
        rw, rx, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in range(self.ncores):
            cy, cx = divmod(c, self.gx)
            q, bank = divmod(c, w.banks)
            rw[cx][cy] = [wa, bank, q * pn * _WB]
            rx[cx][cy] = [xa, ya, c * pn]
            cp[cx][cy] = []
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_READER_W,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[kb, nb, w.shard_tiles * _WB, pn * _WB, pn],
                runtime_args=rw,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_READER_X,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[mt, kb, nb, pn, kt, nt, self.ct] + _accessor_args(x) + _accessor_args(y),
                runtime_args=rx,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[mt, kb, nb, pn, self.ct],
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(),
            ),
        ]
        cfg = kernels[2].config
        # HiFi4, as the stock decode down projection runs.
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        cfg.dst_full_sync_en = True
        cbs = [
            self._cb(0, ttnn.bfloat16, _BF16, 2 * mt * kb),
            self._cb(1, _W_DTYPE, _WB, self.depth * kb * pn),
            self._cb(16, ttnn.float32, 4096, mt * pn),
        ]
        if self.spill:
            cbs.append(self._cb(24, ttnn.float32, 4096, mt * pn))
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        desc.custom_program_hash = (
            hash(("voxtral_cpp_down_dec", mt, kt, nt, kb, pn, self.ct, xa, wa, ya)) & 0xFFFFFFFFFFFFFFFF
        )
        return desc


_PLANS: dict = {}


def _rows(x):
    rows = 1
    for d in tuple(x.padded_shape)[:-1]:
        rows *= int(d)
    return rows


def serves(x, w) -> bool:
    if w is None or not enabled():
        return False
    try:
        if x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or x.dtype != ttnn.bfloat16:
            return False
        rows, k = _rows(x), int(tuple(x.padded_shape)[-1])
        return k == w.k and rows == _TILE
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def apply(x, w):
    """`x @ w2` as float32 in L1; x is moved to L1 first if it is not there."""
    device = x.device()
    shape = [int(d) for d in x.shape]
    rows, k, n = _rows(x), shape[-1], w.n
    mt = rows // _TILE
    key = (mt, id(w.tensor))
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _Plan(device, mt, w)
        _log(
            f"plan {mt}x{plan.kt}x{plan.nt} cores={plan.ncores} pn={plan.pn} ct={plan.ct} kb={plan.kb} depth={plan.depth}"
        )
    flat = ttnn.reshape(x, [1, 1, rows, k])
    if flat.memory_config().buffer_type != ttnn.BufferType.L1:
        flat = ttnn.to_memory_config(flat, ttnn.L1_MEMORY_CONFIG)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, n]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    ttnn.generic_op([flat, w.tensor, y], plan.descriptor(flat, w, y))
    return ttnn.reshape(y, shape[:-1] + [n])
