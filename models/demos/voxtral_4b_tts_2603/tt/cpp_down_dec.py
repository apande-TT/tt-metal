# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on a decode step's FFN down projection `gated @ W_down` (32 x 9216 x 3072),
through ttnn.generic_op: `cpp_down`'s kernels (tt/cpp_down_kernels) on the stock decode recipe --
bf8_b weight, bf16 activation, HiFi4 with fp32 DEST, float32 out for the residual add.

The weight is width-sharded over the DRAM banks with each core's PN output columns contiguous in one
bank's shard row (one read per K row); the one-tile-row activation is read from L1 one K block at a
time.

THE STOCK BITS. The output is accumulated exactly as the stock 1D-multicast matmul it replaces
accumulates it (cpp_down_kernels/compute_exact.cpp): the stock config's own K block (`_stock_kb`, the
`_row_cfg` rule of the stubs), each block summed into a fresh DEST, blocks packer-L1-accumulated into an
fp32 partials CB, and the last block summed on top of the reloaded partials. The first version held one
DEST accumulation across every K block; that differs from stock only in rounding order, but free-running
greedy decode is chaotic and it put demo row 18 into a repeat loop.

On unless VOXTRAL_CPP_DOWN_DEC=0, for the first VOXTRAL_CPP_DOWN_DEC_LAYERS (default: all) layers built.
VOXTRAL_CPP_DOWN_DEC_AB=<log path> (or the file /tmp/voxtral_wip/ab_down_dec.on, whose first line is the log
path) also runs the calling stub's own stock down projection on every call (its `_down_short` on its own
`w_down`, no extra weight copy) and logs how many output elements differ: an on-device A/B for eager runs
(host reads; not for trace capture).
"""

from __future__ import annotations

import os
import pathlib
import sys

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_down_kernels"
_READER_W = str(_DIR / "reader_w.cpp")
_READER_X = str(_DIR / "reader_x_writer.cpp")
_COMPUTE = str(_DIR / "compute_exact.cpp")

_TILE = 32
# At the stock K block of 32 one weight block is 104 KB a core; two in flight keep the bank stream busy, and
# a deeper ring grows the static CB region the decode step's L1 tensors have to sit above.
_CB_BUDGET = 400 * 1024
_DEST_TILES = 4  # half-sync fp32 DEST, as the stock matmul runs
_DEPTHS = (4, 3, 2)  # weight K blocks in flight
_CHUNK = 8  # K rows a reader push
_BF16 = 2048
_W_DTYPE, _WB = ttnn.bfloat8_b, 1088
_NUM_CBS = 64

# The stubs' `_row_cfg` rule (layer / mistral_decoder_layer / mlp / mistral_m_l_p): the widest K block that
# fits their L1 budget with a bf16 activation, bf8_b weight and float32 output at per_core_N = 1.
_STOCK_L1_BUDGET = 1_100_000
_STOCK_KBS = (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
_FP32 = 4096


def _stock_kb(kt):
    """The in0_block_w the stock decode down projection runs at (the K block the bits depend on)."""
    return next((c for c in _STOCK_KBS if kt % c == 0 and _FP32 + 2 * c * (_BF16 + _WB) <= _STOCK_L1_BUDGET), None)


def _ab_log():
    path = os.environ.get("VOXTRAL_CPP_DOWN_DEC_AB")
    if path:
        return path
    try:
        with open("/tmp/voxtral_wip/ab_down_dec.on") as fh:
            return fh.readline().strip() or "/tmp/voxtral_wip/ab_down_dec.log"
    except OSError:
        return None


_AB = _ab_log()


def _log(msg):
    path = os.environ.get("VOXTRAL_CPP_SWIGLU_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write("down " + msg + "\n")


def enabled() -> bool:
    # Off from 2026-10-05 morning until the accumulation matched stock: the earlier single-DEST
    # accumulation differed from the stock down projection only in rounding order, but free-running
    # greedy decode is chaotic, and that rounding put demo row 18 into a repeat loop ("discovering
    # very, very, very, very books", 147 frames for 13 words) that the corpus-level WER/MOS margins and
    # the trajectory-aligned PCC gates could not see. With compute_exact.cpp the bits are the stock ones.
    return os.environ.get("VOXTRAL_CPP_DOWN_DEC", "1") == "1"


_LAYERS = int(os.environ.get("VOXTRAL_CPP_DOWN_DEC_LAYERS", "1000"))
_built = [0]


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_down_dec: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


class Sharded:
    """w2 as one bank-sharded tensor, plus the core split it was laid out for (and, for the A/B, the
    stock interleaved copy)."""

    def __init__(self, tensor, k, n, pn, ncores, banks, stock=None):
        self.tensor, self.k, self.n, self.pn, self.ncores, self.banks = tensor, k, n, pn, ncores, banks
        self.shard_tiles = n // _TILE // banks
        self.stock = stock


def shard(w2, device):
    """`w2`: torch `[K, N]` (already transposed). None when the shape has no split."""
    if not enabled() or _built[0] >= _LAYERS:
        return None
    _built[0] += 1
    k, n = int(w2.shape[0]), int(w2.shape[1])
    if k % _TILE or n % _TILE or _stock_kb(k // _TILE) is None:
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
    # The A/B compares against the calling stub's own stock weight (its `w_down`, built just before this).
    stock = sys._getframe(1).f_locals.get("w_down") if _AB else None
    sharded = Sharded(tt, k, n, pn, ncores, banks, stock)
    sharded.index = _built[0]
    return sharded


class _Plan:
    def __init__(self, device, mt, w):
        grid = device.compute_with_storage_grid_size()
        self.gx = int(grid.x)
        self.mt, self.kt, self.nt = mt, w.k // _TILE, w.n // _TILE
        self.pn, self.ncores = w.pn, w.ncores
        self.ct = max(c for c in range(1, self.pn + 1) if self.pn % c == 0 and mt * c <= _DEST_TILES)
        # The stock K block, not a free choice: the bits depend on where the packer accumulates.
        self.kb = _stock_kb(self.kt)
        # The stream chunk (K rows a reader push), independent of the accumulation's K block.
        self.ch = max(c for c in range(1, _CHUNK + 1) if self.kb % c == 0)
        fixed = 2 * mt * self.pn * _FP32 + 2 * mt * self.kb * _BF16  # output + partials, x double-buffered
        self.depth = next((d for d in _DEPTHS if fixed + d * self.kb * self.pn * _WB <= _CB_BUDGET), None)
        if self.depth is None:
            raise RuntimeError(f"cpp_down_dec: no weight depth for {mt}x{self.kt}x{self.nt} at kb={self.kb}")
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
        mt, kt, nt, kb, nb, pn, ch = self.mt, self.kt, self.nt, self.kb, self.nb, self.pn, self.ch
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
                # Both readers stream CH K rows a push; the compute waits for them inside its K block.
                compile_time_args=[ch, kt // ch, w.shard_tiles * _WB, pn * _WB, pn],
                runtime_args=rw,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_READER_X,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[mt, ch, kt // ch, pn, kt, nt, self.ct] + _accessor_args(x) + _accessor_args(y),
                runtime_args=rx,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[mt, kb, nb, pn, self.ct, ch],
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(),
            ),
        ]
        cfg = kernels[2].config
        # The stock decode down projection's compute config: HiFi4, fp32 DEST, half-sync DEST, and its
        # fp32 partials CB unpacked straight to DEST for the last block's reload.
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        cfg.dst_full_sync_en = False
        modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
        modes[24] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        cbs = [
            self._cb(0, ttnn.bfloat16, _BF16, 2 * mt * kb),
            self._cb(1, _W_DTYPE, _WB, self.depth * kb * pn),
            self._cb(16, ttnn.float32, _FP32, mt * pn),
            # Exactly one output block: each block's packer accumulation lands on the previous one's tiles.
            self._cb(24, ttnn.float32, _FP32, mt * pn),
        ]
        # No custom_program_hash: one keyed on buffer addresses misses the program cache inside a trace
        # capture (the default hash leaves runtime-arg values out, and a hit re-copies them).
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


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
    if _AB and w.stock is not None:
        _ab_compare(x, w, y, sys._getframe(1).f_globals)
    return ttnn.reshape(y, shape[:-1] + [n])


_AB_CALLS = [0]


def _ab_compare(x, w, y, stub):
    """Log how many elements of `y` differ from what the calling stub's OWN stock path computes: its
    `_down_short` on its own bf8_b `w_down` (taken from its build frame by `shard`) and its `_COMPUTE`."""
    if _AB_CALLS[0] >= 20000:
        return
    _AB_CALLS[0] += 1
    shape = [int(d) for d in x.shape]
    try:
        ref = stub["_down_short"](
            x, w.stock, dtype=ttnn.float32, compute_kernel_config=stub["_COMPUTE"], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        a = ttnn.to_torch(ttnn.reshape(y, shape[:-1] + [w.n])).float()
        b = ttnn.to_torch(ref).float()
        ttnn.deallocate(ref)
        diff = (a - b).abs()
        msg = (
            f"call {_AB_CALLS[0]} layer={w.index} shape={shape} padded={list(x.padded_shape)} "
            f"differ={int((diff > 0).sum())}/{diff.numel()} maxdiff={float(diff.max()):.3e}\n"
        )
    except Exception as exc:  # noqa: BLE001 -- a diagnostic; never break the forward
        msg = f"call {_AB_CALLS[0]} layer={w.index} shape={shape} A/B failed: {str(exc)[:300]}\n"
    with open(_AB, "a") as fh:
        fh.write(msg)
