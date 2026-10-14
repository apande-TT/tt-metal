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

_TILE = 32
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}
BLK = 4  # tiles a block (fp32 DEST holds 4)
CB_X, CB_D, CB_SCALER, CB_H, CB_X2, CB_OUT, CB_S = 0, 1, 2, 3, 4, 16, 17


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
    device = x.device()
    shape = [int(s) for s in x.shape]
    rows = 1
    for s in shape[:-1]:
        rows *= s
    rt, wt = rows // _TILE, shape[-1] // _TILE
    grid = device.compute_with_storage_grid_size()
    gx, ncores = int(grid.x), min(int(grid.x) * int(grid.y), rt)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    h = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    stats = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape[:-1] + [_TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    xa, da, ha, sa = x.buffer_address(), d.buffer_address(), h.buffer_address(), stats.buffer_address()
    base, extra = divmod(rt, ncores)
    rr, rw, rc = [], [], []
    r0 = 0
    for c in range(ncores):
        nr = base + (1 if c < extra else 0)
        cy, cx = divmod(c, gx)
        rr.append(((cx, cy), [xa, da, r0, nr]))
        rw.append(((cx, cy), [ha, sa, r0, nr]))
        rc.append(((cx, cy), [nr]))
        r0 += nr
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in (CB_H, CB_X2):  # fp32 operands of the SFPU copy / square / accurate reduce
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
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, BLK] + _accessor_args(h) + _accessor_args(stats),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
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
        _cb(cores, CB_H, ttnn.float32, wt),
        _cb(cores, CB_X2, ttnn.float32, wt),
        _cb(cores, CB_OUT, ttnn.float32, 2 * BLK),
        _cb(cores, CB_S, ttnn.float32, 2),
    ]
    ttnn.generic_op([x, d, h, stats], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return h, stats
