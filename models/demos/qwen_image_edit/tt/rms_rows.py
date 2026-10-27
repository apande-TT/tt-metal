# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The precise float32 RMSNorm as two C++ Metalium kernels (ttnn.generic_op).

Replaces the chain x * x, mean, + eps, rsqrt, x * rs, * w (_stubs/encoder_stack.py TtRMSNorm, precise path):
kernels/rms_scale.cpp makes the row scale rs = rsqrt(mean(x * x) + eps) from one read of x, one tile row per
core, and kernels/rms_apply.cpp the output (x * rs) * w from a second read, spread over every core by tile. Six
ops and four float32 intermediates of x's size become two ops and one [.., 1] intermediate.

Bit-identical: each step makes the stock op's LLK calls with its configuration (see the kernels' headers):
float32 inputs unpacked to DEST (UnpackToDestFp32), float32 DEST, math_approx_mode off; the squares and the
row scale stay in DEST instead of float32 tensors, so the same float32 values reach the same SFPU ops in the
same order.
"""
from __future__ import annotations

import os

import numpy as np

import ttnn

# the kernels live next to this module; the stock writer is found under TT_METAL_HOME
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TT_METAL = os.environ.get("TT_METAL_HOME", "")
_WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"


def _f32_bits(v):
    return int(np.array([v], dtype=np.float32).view(np.uint32)[0])


def fused_rms_norm(x, w, eps, mem=None):
    """x [..., S, W] float32 TILE (interleaved, W a multiple of 32), w [1, W] float32 TILE
    -> (x * rsqrt(mean(x * x, -1) + eps)) * w, bit-identical to the ttnn op chain."""
    assert x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT, "float32 TILE x"
    assert w.dtype == ttnn.float32 and w.layout == ttnn.TILE_LAYOUT, "float32 TILE w"
    W = int(x.shape[-1])
    Wt = int(x.padded_shape[-1]) // 32
    assert W == Wt * 32, "no padding along the normalized dim"
    vol = 1
    for d in x.padded_shape:
        vol *= int(d)
    rows = vol // 1024 // Wt
    mem = mem or ttnn.DRAM_MEMORY_CONFIG
    device = x.device()
    rs = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)[:-1] + [1]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.allocate_tensor_on_device(x.shape, ttnn.float32, ttnn.TILE_LAYOUT, device, mem)
    batch = next(b for b in (4, 2, 1) if Wt % b == 0)
    accessors = []
    for t in (x, rs, w):
        accessors += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    addrs = [x.buffer_address(), rs.buffer_address(), w.buffer_address()]
    ttnn.generic_op(
        [x, rs],
        _program(device, rows, rs, [(0, 2 * batch), (16, 2)], [Wt, 0, batch] + accessors, addrs,
                 "rms_scale.cpp", [Wt, _f32_bits(1.0 / W), _f32_bits(eps)], with_start=False),
    )
    return ttnn.generic_op(
        [x, rs, w, out],
        _program(device, rows * Wt, out, [(0, 2), (1, 2), (2, 2), (16, 2)], [Wt, 1, 1] + accessors, addrs,
                 "rms_apply.cpp", [Wt], with_start=True),
    )


def _program(device, work, out, pages, reader_ct, addrs, compute, compute_ct, with_start):
    """reader_rms_rows.cpp + the stock tile writer of `out` + `compute`, over `work` units (tile rows or tiles)
    split across the grid; every unit makes one output tile."""
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, work)
    page = 4096  # one float32 tile
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=page)],
        )
        for i, n in pages
    ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    rd_rt[x][y] = addrs + [per, start]
                    wr_rt[x][y] = [out.buffer_address(), per, start]
                    cp_rt[x][y] = [per, start] if with_start else [per]
                    start += per
    # as binary_ng / unary / the accurate fp32 reduce: float32 inputs unpacked to DEST, float32 DEST, precise SFPU
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 32
    for i, _ in pages:
        if i < 16:
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_rms_rows.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_TT_METAL, _WRITER),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, compute),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
