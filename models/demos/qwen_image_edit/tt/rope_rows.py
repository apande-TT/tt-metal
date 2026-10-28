# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The float32 rotate_half RoPE x * cos + rotate_half(x) * sin as one C++ Metalium kernel (ttnn.generic_op).

Replaces the chain slice / slice / neg / concat / multiply / multiply / add (_stubs/attention.rotate_half and the
LM attention's _rope): kernels/reader_rope.cpp streams each output tile's x tile, its half-swapped x tile (D / 2 is
tile aligned) and the cos / sin tiles; kernels/rope_swap.cpp makes the same SFPU float32 multiplies, negation and
round-to-nearest-even add in DEST, so the result is bit-identical and none of the five float32 intermediates of
x's size is written.
"""
from __future__ import annotations

import os

import ttnn

# the kernels live next to this module; the stock writer is found under TT_METAL_HOME
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TT_METAL = os.environ.get("TT_METAL_HOME", "")
_WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"


def fused_rope(x, cos, sin, mem=None):
    """x [B, H, S, D] float32 TILE (interleaved), cos / sin [1 or B, 1, S, D] float32 TILE -> x * cos +
    rotate_half(x) * sin, bit-identical to the ttnn chain; None when the shapes do not fit (D / 2 not tile
    aligned, padding, other layouts) so the caller runs the chain."""
    if len(x.shape) != 4 or any(t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT for t in (x, cos, sin)):
        return None
    if x.is_sharded() or cos.is_sharded() or sin.is_sharded():
        return None
    B, H, S, D = (int(d) for d in x.shape)
    if D % 64 or S % 32 or list(x.padded_shape) != list(x.shape):
        return None
    cb = int(cos.shape[0])
    if list(cos.shape) != [cb, 1, S, D] or list(sin.shape) != [cb, 1, S, D] or cb not in (1, B):
        return None
    Dt, St = D // 32, S // 32
    device = x.device()
    out = ttnn.allocate_tensor_on_device(x.shape, ttnn.float32, ttnn.TILE_LAYOUT, device, mem or ttnn.DRAM_MEMORY_CONFIG)
    n_tiles = B * H * St * Dt
    reader_ct = [Dt, St, H, 1 if cb > 1 else 0]
    for t in (x, cos, sin):
        reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, n_tiles)
    page = 4096  # one float32 tile
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=page)],
        )
        for i, pages in ((0, 8), (16, 2))  # the four input tiles of an output tile double-buffered
    ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [x.buffer_address(), cos.buffer_address(), sin.buffer_address()]
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    rd_rt[cx][cy] = addrs + [per, start]
                    wr_rt[cx][cy] = [out.buffer_address(), per, start]
                    cp_rt[cx][cy] = [per, start]
                    start += per
    # as binary_ng / unary on float32: inputs unpacked to DEST as float32, float32 DEST, precise SFPU
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 32
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_rope.cpp"),
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
            kernel_source=os.path.join(_KERNELS, "rope_swap.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[Dt],
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    return ttnn.generic_op([x, cos, sin, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
