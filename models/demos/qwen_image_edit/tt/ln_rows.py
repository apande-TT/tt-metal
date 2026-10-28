# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The transformer blocks' float32 LayerNorm (no affine) + modulation as one C++ Metalium program (ttnn.generic_op).

Replaces _stubs/encoder_stack.py _TtBlock._modulate(x, shift, scale) = LN(x) * (1 + scale) + shift, spelled as
mean / subtract / multiply / mean / add / rsqrt / multiply / multiply / add with float32 intermediates of x's size:
kernels/ln_modulate.cpp makes each tile row's mean and rsqrt(var + eps) and the modulated output with the stock
ops' SFPU calls in the same order (see its header), from one read of the row kept in L1 (kernels/reader_ln_mod.cpp), so the
result is bit-identical and none of the intermediates is written.
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


def fused_ln_modulate(x, scale1, shift, eps, mem=None):
    """x [B, S, W] float32 TILE (interleaved, S and W tile multiples), scale1 = 1 + scale and shift [B or 1, 1, W]
    float32 TILE -> LN(x) * scale1 + shift, bit-identical to the ttnn chain; None when the shapes do not fit."""
    ts = (x, scale1, shift)
    if any(t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded() for t in ts):
        return None
    if len(x.shape) != 3 or len(scale1.shape) != 3 or list(scale1.shape) != list(shift.shape):
        return None
    B, S, W = (int(d) for d in x.shape)
    if S % 32 or W % 32 or list(x.padded_shape) != list(x.shape):
        return None
    sB = int(scale1.shape[0])
    if sB not in (1, B) or int(scale1.shape[1]) != 1 or int(scale1.shape[2]) != W:
        return None
    Wt, St = W // 32, S // 32
    rows = B * St
    device = x.device()
    out = ttnn.allocate_tensor_on_device(x.shape, ttnn.float32, ttnn.TILE_LAYOUT, device, mem or ttnn.DRAM_MEMORY_CONFIG)
    batch = next(b for b in (4, 2, 1) if Wt % b == 0)
    reader_ct = [Wt, St, 1 if sB > 1 else 0, batch]
    for t in ts:
        reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, rows)
    page = 4096  # one float32 tile
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=page)],
        )
        for i, pages in ((0, 2 * Wt), (1, 1), (2, 1), (3, 4), (16, 2))  # x: two whole rows (one in use, one read)
    ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [x.buffer_address(), scale1.buffer_address(), shift.buffer_address()]
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    rd_rt[cx][cy] = addrs + [per, start]
                    wr_rt[cx][cy] = [out.buffer_address(), per * Wt, start * Wt]
                    cp_rt[cx][cy] = [per]
                    start += per
    # as binary_ng / unary / the accurate float32 reduce: float32 inputs unpacked to DEST, float32 DEST, precise SFPU
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 32
    for i in (0, 1, 2, 3):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_ln_mod.cpp"),
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
            kernel_source=os.path.join(_KERNELS, "ln_modulate.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[Wt, _f32_bits(1.0 / W), _f32_bits(eps)],
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    return ttnn.generic_op([x, scale1, shift, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
