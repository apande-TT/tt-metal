# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""C++ Metalium residual add fused with the next RMSNorm's statistics (generic_op, kernels/add_stats_*.cpp).

For the decoder's fp32 (or, entering the first layer, bf16) hidden-sharded prefill residual x and an attention / MLP
delta d it returns h = x + d (fp32, added on the SFPU exactly as binary_ng adds mixed dtypes) and, per row, sum(h^2)
in column 0 of a one-tile-wide fp32 stats tensor -- what ttnn.rms_norm_pre_all_gather computes from h, without
reading h back from DRAM. Like that op's fp32 path, the other 31 stats columns hold reduce partials, so the gathered
stats are masked before summing. Tile rows are split over the full compute grid."""
from __future__ import annotations

from pathlib import Path

import ttnn

_DIR = Path(__file__).resolve().parent / "kernels"
_READER = str(_DIR / "add_stats_reader.cpp")
_WRITER = str(_DIR / "add_stats_writer.cpp")
_COMPUTE = str(_DIR / "add_stats_compute.cpp")
_CAST_WRITER = str(_DIR / "add_cast_writer.cpp")
_CAST_COMPUTE = str(_DIR / "add_cast_compute.cpp")

_TILE = 32
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}
BLK = 4  # tiles a block (fp32 DEST holds 4)
CB_X, CB_D, CB_SCALER, CB_H, CB_X2, CB_MASK, CB_OUT, CB_S, CB_SRAW = 0, 1, 2, 3, 4, 5, 16, 17, 18
# apply()'s stats tile holds the row sum in column 0 and exact zeros in columns 1..31 (no masking needed after a gather)
CLEAN_STATS = True
# shard try (run 8): the one-tile-wide stats (~16 KB a core at 416 rows) in L1 for the norm's stats all_gather
STATS_L1 = True
# structural try (run 8): the stats written compacted -- each stats tile's column 0 (its 32 row sums) as one row of a
# compact tile, (1, 1, tile_rows, 32) for tile_rows a multiple of 32 -- so the norm's stats all_gather moves 1/32 of
# the bytes; tt/stats_expand.py rebuilds the gathered (..., S, 128) tiles exactly (zeros + column 0) for the sum
COMPACT_STATS = True
CB_ROW = 19
# apply_l1's per-core shard bounds (see l1_copy_cores)
L1_COPY_MIN_CORES = 64
L1_COPY_MAX_BYTES = 640 * 1024


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("add_stats: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _cb(cores, index, dtype, tiles):
    size = _TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=tiles * size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
    )


def supports(x, d) -> bool:
    shape = tuple(int(s) for s in x.shape)
    return (
        x.layout == ttnn.TILE_LAYOUT
        and d.layout == ttnn.TILE_LAYOUT
        and not x.is_sharded()
        and not d.is_sharded()
        and x.dtype in (ttnn.float32, ttnn.bfloat16)
        and d.dtype in (ttnn.float32, ttnn.bfloat16)
        and tuple(int(s) for s in d.shape) == shape
        and shape[-1] % (_TILE * BLK) == 0
        and shape[-2] % _TILE == 0
    )


def apply(x, d):
    """x: fp32 (or bf16, widened exactly) TILE interleaved [..., S, W]; d: same shape, bf16 or fp32. Returns
    (h = x + d fp32 [..., S, W], stats fp32 [..., S, 32] with sum(h^2) of each row in column 0)."""
    return _run(x, d, cast=False)


def compact_rows(x):
    """Tile rows of x's stats when apply() writes them compacted (COMPACT_STATS), else 0."""
    shape = [int(s) for s in x.shape]
    rows = 1
    for s in shape[:-1]:
        rows *= s
    rt = rows // _TILE
    return rt if COMPACT_STATS and rt % _TILE == 0 and rt >= 2 * _TILE else 0


def is_compact(stats, x) -> bool:
    """Whether stats is apply()'s compacted stats tensor for the residual x."""
    rt = compact_rows(x)
    return bool(rt) and tuple(int(s) for s in stats.shape) == (1, 1, rt, _TILE)


def l1_copy_cores(x):
    """Cores of apply_l1's height shard for x's rows (whole tile rows a core, the most cores that split them evenly),
    or 0 when no even split reaches L1_COPY_MIN_CORES or a core's fp32 shard would exceed L1_COPY_MAX_BYTES."""
    shape = [int(s) for s in x.shape]
    rows = 1
    for s in shape[:-1]:
        rows *= s
    rt, wt = rows // _TILE, shape[-1] // _TILE
    grid = x.device().compute_with_storage_grid_size()
    n = max(c for c in range(1, min(int(grid.x) * int(grid.y), rt) + 1) if rt % c == 0)
    if n < L1_COPY_MIN_CORES or rt // n * wt * _TILE_BYTES[ttnn.float32] > L1_COPY_MAX_BYTES:
        return 0
    return n


def apply_l1(x, d):
    """apply() that also returns a height-sharded L1 copy of h (fp32, l1_copy_cores(x) cores, each its own tile rows,
    written by the same pass): (h, stats, h_l1). The caller deallocates h_l1."""
    return _run(x, d, cast=False, l1_copy=True)


def apply_bf16(x, d):
    """bf16(x + d) [..., S, W] for the same inputs: the fp32 sum rounded by the SFPU typecast ttnn.typecast runs
    (bit-identical to apply()'s h followed by ttnn.typecast to bf16), with no fp32 h and no stats written. For the
    last residual add, whose sum only feeds a bf16 all_gather."""
    return _run(x, d, cast=True)


def _run(x, d, cast, l1_copy=False):
    device = x.device()
    shape = [int(s) for s in x.shape]
    rows = 1
    for s in shape[:-1]:
        rows *= s
    rt, wt = rows // _TILE, shape[-1] // _TILE
    grid = device.compute_with_storage_grid_size()
    gx, ncores = int(grid.x), min(int(grid.x) * int(grid.y), rt)
    if l1_copy:
        ncores = l1_copy_cores(x)
        assert ncores, "add_stats.apply_l1: no even L1 shard for these rows"
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    out_dtype = ttnn.bfloat16 if cast else ttnn.float32
    h = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), out_dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    h_l1 = None
    if l1_copy:
        h_l1 = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.create_sharded_memory_config(
                (rt // ncores * _TILE, shape[-1]),
                core_grid=cores,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )
    stats = None
    compact = not cast and bool(compact_rows(x))
    if not cast:
        stats = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rt, _TILE] if compact else shape[:-1] + [_TILE]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG if STATS_L1 else ttnn.DRAM_MEMORY_CONFIG,
        )
    xa, da, ha = x.buffer_address(), d.buffer_address(), h.buffer_address()
    base, extra = divmod(rt, ncores)
    rr, rw, rc = [], [], []
    r0 = 0
    for c in range(ncores):
        nr = base + (1 if c < extra else 0)
        cy, cx = divmod(c, gx)
        rr.append(((cx, cy), [xa, da, r0, nr]))
        rw.append(((cx, cy), [ha, r0, nr] if cast else [ha, stats.buffer_address(), r0, nr]))
        if h_l1 is not None:
            rw[-1][1].append(h_l1.buffer_address())
        rc.append(((cx, cy), [nr]))
        r0 += nr
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in (CB_H, CB_X2, CB_MASK, CB_SRAW):  # fp32 operands of the SFPU copy / square / reduce / mask multiply
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    for cb, t in ((CB_X, x), (CB_D, d)):  # the add's operands: fp32 straight to DEST, bf16 widened exactly
        if t.dtype == ttnn.float32:
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, BLK] + _accessor_args(x) + _accessor_args(d),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
            defines=[] if cast else [("STATS_MASK", "1")],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_CAST_WRITER if cast else _WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, BLK] + _accessor_args(h) + ([] if cast else _accessor_args(stats)),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
            defines=([("L1_COPY", "1")] if h_l1 is not None else []) + ([("COMPACT_STATS", "1")] if compact else []),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_CAST_COMPUTE if cast else _COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, BLK],
            runtime_args=rc,
            config=cfg,
        ),
    ]
    cbs = [
        _cb(cores, CB_X, x.dtype, 2 * BLK),
        _cb(cores, CB_D, d.dtype, 2 * BLK),
        _cb(cores, CB_SCALER, ttnn.float32, 1),
    ]
    if cast:
        cbs.append(_cb(cores, CB_OUT, out_dtype, 2 * BLK))
        ttnn.generic_op([x, d, h], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return h
    cbs += [
        _cb(cores, CB_H, ttnn.float32, wt),
        _cb(cores, CB_X2, ttnn.float32, wt),
        _cb(cores, CB_OUT, ttnn.float32, 2 * BLK),
        _cb(cores, CB_S, ttnn.float32, 2),
        _cb(cores, CB_MASK, ttnn.float32, 1),
        _cb(cores, CB_SRAW, ttnn.float32, 2),
    ]
    if compact:  # the 32 row sums of a stats tile, gathered into one row
        fmt = ttnn.CBFormatDescriptor(buffer_index=CB_ROW, data_format=ttnn.uint32, page_size=128)
        cbs.append(ttnn.CBDescriptor(total_size=128, core_ranges=cores, format_descriptors=[fmt]))
    if h_l1 is not None:
        ttnn.generic_op([x, d, h, stats, h_l1], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return h, stats, h_l1
    ttnn.generic_op([x, d, h, stats], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return h, stats
