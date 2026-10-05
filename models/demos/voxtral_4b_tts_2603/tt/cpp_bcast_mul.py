# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`ttnn.multiply(a, b)` of a float32 `a [..., R, W]` by a float32 per-row `b [..., R, 1]` (an RMS norm's
`x * rsqrt(mean(x^2) + eps)`) as ONE ttnn.generic_op, bit for bit (kernels in tt/cpp_bcast_mul_kernels).

The stock op (binary_ng, float32 column broadcast on Blackhole: no LLK broadcast, so its reader
column-fills b) runs one tile a core at a time, every read and write behind its own barrier: ~3.5 us a
tile at the 160-row prefix (15-17 us a norm), and on a decode step's one tile row all 96 cores read the
SAME b tile out of one core's L1 (10.4 us). Most of that time is binary_ng's reader column-filling b with a
volatile word-by-word loop (~5 us a tile). Here a core owns a run of >= 3 tiles, reads it CH tiles a chunk
behind one barrier, fills b with straight-line stores (the same data), and runs the same per-tile steps --
both operands unpacked straight to DEST, mul_binary_tile, packed to the output's format -- so every output is
the stock one. Measured: decode 10.7 -> 4.9 us, 160-row prefix 12.6 -> 7.2 us.

On unless VOXTRAL_CPP_BCAST_MUL=0. VOXTRAL_CPP_BCAST_MUL_AB=<log path> (or the file
/tmp/voxtral_wip/ab_bcast_mul.on, first line the log path) also runs the stock multiply on every call and
logs how many output elements differ (host reads: eager runs only).
"""

from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_bcast_mul_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_CH = 8  # a tiles a chunk (one read barrier, one write barrier)
_MIN_TILES = 3  # a core's run is at least this long: fewer cores read each b tile
_MAX_RUN = 4 * _CH  # longer runs (bigger tensors) stay on the stock op
_NUM_CBS = 64
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_BCAST_MUL", "1") == "1"


def _ab_log():
    path = os.environ.get("VOXTRAL_CPP_BCAST_MUL_AB")
    if path:
        return path
    try:
        with open("/tmp/voxtral_wip/ab_bcast_mul.on") as fh:
            return fh.readline().strip() or "/tmp/voxtral_wip/ab_bcast_mul.log"
    except OSError:
        return None


_AB = _ab_log()
_AB_CALLS = [0]


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_bcast_mul: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _interleaved(mc):
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _per_core(a):
    """(a's tile count, the run a core owns)."""
    shape = [int(d) for d in a.shape]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    total = (rows // _TILE) * (shape[-1] // _TILE)
    grid = a.device().compute_with_storage_grid_size()
    return total, max(-(-total // (int(grid.x) * int(grid.y))), _MIN_TILES)


def supports(a, b, dtype=None, memory_config=None) -> bool:
    """float32 TILE interleaved `a [..., R, W]` and `b [..., R, 1]` with the same leading shape, R a tile
    multiple, a float32 / bf16 / bf8_b interleaved output, and at most _MAX_RUN tiles a core."""
    if not enabled():
        return False
    try:
        sa, sb = [int(d) for d in a.shape], [int(d) for d in b.shape]
        out_dtype = dtype or a.dtype
        mc = memory_config or a.memory_config()
        return (
            a.dtype == ttnn.float32
            and b.dtype == ttnn.float32
            and a.layout == ttnn.TILE_LAYOUT
            and b.layout == ttnn.TILE_LAYOUT
            and _interleaved(a.memory_config())
            and _interleaved(b.memory_config())
            and _interleaved(mc)
            and out_dtype in _TILE_BYTES
            and len(sa) == len(sb)
            and sa[:-1] == sb[:-1]
            and sb[-1] == 1
            and sa[-1] % _TILE == 0
            and sa[-2] % _TILE == 0
            and [int(d) for d in a.padded_shape] == sa
            and _per_core(a)[1] <= _MAX_RUN
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, fmt, tiles):
    size = _TILE_BYTES[fmt]
    return ttnn.CBDescriptor(
        total_size=tiles * size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)],
    )


def mul_col(a, b, dtype=None, memory_config=None):
    """`a * b` (b broadcast along the last dim) as `dtype` (default a's) in `memory_config` (default a's)."""
    device = a.device()
    shape = [int(d) for d in a.shape]
    out_dtype = dtype or a.dtype
    mc = memory_config or a.memory_config()
    wt = shape[-1] // _TILE
    rows = 1
    for d in shape[:-1]:
        rows *= d
    total = (rows // _TILE) * wt
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    per = max(-(-total // (gx * gy)), _MIN_TILES)
    ncores = -(-total // per)
    nbr = min(_CH, (_CH - 1) // wt + 2)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, device, mc)
    aa, ba, ya = a.buffer_address(), b.buffer_address(), y.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    t0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        n = min(per, total - t0)
        rr[cx][cy] = [aa, ba, t0, n]
        rc[cx][cy] = [t0, n]
        rw[cx][cy] = [ya, t0, n]
        t0 += n
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, _CH, nbr] + _accessor_args(a) + _accessor_args(b),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, _CH, nbr, int(out_dtype == ttnn.bfloat16)],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[_CH] + _accessor_args(y),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # binary_ng's float32 SFPU config: fp32 DEST, both inputs unpacked straight to DEST, the other fields at
    # their defaults (approx off, half-sync DEST, default bf8 packing).
    cfg.fp32_dest_acc_en = True
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, ttnn.float32, 2 * _CH),
        _cb(cores, 1, ttnn.float32, 2 * nbr),
        _cb(cores, 16, out_dtype, 2 * _CH),
    ]
    # No custom_program_hash: one keyed on buffer addresses misses the program cache inside a trace capture.
    ttnn.generic_op([a, b, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    if _AB:
        _ab_compare(a, b, y, out_dtype, mc)
    return y


def _ab_compare(a, b, y, out_dtype, mc):
    if _AB_CALLS[0] >= 20000:
        return
    _AB_CALLS[0] += 1
    shape = [int(d) for d in a.shape]
    try:
        ref = ttnn.multiply(a, b, dtype=out_dtype, memory_config=mc)
        p, q = ttnn.to_torch(y).float(), ttnn.to_torch(ref).float()
        ttnn.deallocate(ref)
        diff = (p - q).abs()
        msg = (
            f"call {_AB_CALLS[0]} shape={shape} out={out_dtype} "
            f"differ={int((diff > 0).sum())}/{diff.numel()} maxdiff={float(diff.max()):.3e}\n"
        )
        if int((diff > 0).sum()) and _AB_CALLS[0] < 400:
            # the exact float32 product beside both narrowings, as bit patterns (bf16 = upper half)
            prod = ttnn.to_torch(a).float() * ttnn.to_torch(b).float()[..., :1]
            for i in (diff > 0).flatten().nonzero().flatten()[:4].tolist():
                bits = lambda t: int(t.flatten()[i : i + 1].view(dtype=__import__("torch").int32)) & 0xFFFFFFFF
                msg += f"   prod={bits(prod):08x} mine={bits(p):08x} stock={bits(q):08x}\n"
    except Exception as exc:  # noqa: BLE001 -- a diagnostic; never break the forward
        msg = f"call {_AB_CALLS[0]} shape={shape} A/B failed: {str(exc)[:300]}\n"
    with open(_AB, "a") as fh:
        fh.write(msg)
