# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""float32 -> (hi, lo) bf16 limbs in one C++ Metalium kernel pass (ttnn.generic_op).

The precise paths carry float32 matmul inputs as hi = bf16(x), lo = bf16(x - hi); ttnn spells that as
typecast, typecast back, subtract, typecast (four memory-bound passes and three float32 intermediates).
kernels/split_bf16.cpp reads x once and writes both limbs (packer narrowing for hi, exact SFPU float32
x - hi, packer narrowing for lo). Same precision class, not bit-identical (e2e PCC 0.96017 -> 0.96055).
Plugged into the ports' split helpers (SPLIT_FN).
"""
from __future__ import annotations

import os

import ttnn

# the kernels live next to this module; the stock reader is found under TT_METAL_HOME
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TT_METAL = os.environ.get("TT_METAL_HOME", "")


def fused_split_bf16(x):
    """x float32 TILE (interleaved) -> (hi, lo) bf16 TILE in x's memory config."""
    device = x.device()
    mem = x.memory_config()
    hi = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    lo = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    vol = 1
    for d in x.padded_shape:
        vol *= int(d)
    tiles = vol // 1024
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, tiles)

    def _cb(index, dtype, page):
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    cbs = [_cb(0, ttnn.float32, 4096), _cb(16, ttnn.bfloat16, 2048), _cb(17, ttnn.bfloat16, 2048), _cb(24, ttnn.bfloat16, 2048)]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    rd_rt[cx][cy] = [x.buffer_address(), per, start]
                    wr_rt[cx][cy] = [hi.buffer_address(), lo.buffer_address(), per, start]
                    cp_rt[cx][cy] = [per]
                    start += per
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 32
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(
                _TT_METAL, "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
            ),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=list(ttnn.TensorAccessorArgs(x).get_compile_time_args()),
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "writer_two_outputs.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=list(ttnn.TensorAccessorArgs(hi).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(lo).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "split_bf16.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    ttnn.generic_op([x, hi, lo], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return hi, lo
