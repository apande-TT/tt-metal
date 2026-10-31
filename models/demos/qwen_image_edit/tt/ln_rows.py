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


def fused_ln_modulate(x, scale1, shift, eps, mem=None, split=False):
    """x [B, S, W] float32 TILE (interleaved, S and W tile multiples), scale1 = 1 + scale and shift [B or 1, 1, W]
    float32 TILE -> LN(x) * scale1 + shift, bit-identical to the ttnn chain; None when the shapes do not fit.
    split: -> (y, (hi, lo)) with y's bf16 limbs as limb_split.fused_split_bf16 makes them (DRAM), and y itself
    allocated but NOT written -- for a caller whose only use of y is that split."""
    ts = (x, scale1, shift)
    if any(t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded() for t in ts):
        return None
    if len(x.shape) != 3 or len(scale1.shape) != 3 or list(scale1.shape) != list(shift.shape):
        return None
    B, S, W = (int(d) for d in x.shape)
    if S % 32 or W % 64 or list(x.padded_shape) != list(x.shape):  # an even tile count per row (column pairs)
        return None
    sB = int(scale1.shape[0])
    if sB not in (1, B) or int(scale1.shape[1]) != 1 or int(scale1.shape[2]) != W:
        return None
    Wt, St = W // 32, S // 32
    rows = B * St
    device = x.device()
    out = ttnn.allocate_tensor_on_device(x.shape, ttnn.float32, ttnn.TILE_LAYOUT, device, mem or ttnn.DRAM_MEMORY_CONFIG)
    if split:
        hi = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        lo = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    batch = next(b for b in (4, 2, 1) if Wt % b == 0)
    reader_ct = [Wt, St, 1 if sB > 1 else 0, batch]
    for t in ts:
        reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    # fewer rows than cores: each row on `parts` cores, every one making the row's mean and rsqrt (the same ops,
    # no exchange) and applying / writing its own 1 / parts of the columns (whole column pairs)
    parts = max(p for p in (1, 2, 4) if p == 1 or (rows * p <= grid.x * grid.y and Wt % (2 * p) == 0))
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, rows * parts)
    page = 4096  # one float32 tile
    f32, b16 = (ttnn.float32, page), (ttnn.bfloat16, page // 2)
    # x: two whole rows (one in use, one read); the apply pass works on column pairs, so the others hold two pairs
    cb_list = [(0, 2 * Wt, f32), (1, 1, f32), (2, 1, f32), (3, 8, f32)]
    if split:  # hi, lo, scratch hi, scratch y
        cb_list += [(16, 4, b16), (17, 4, b16), (24, 2, b16), (25, 2, f32)]
    else:
        cb_list += [(16, 4, f32)]
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * fmt[1],
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt[0], page_size=fmt[1])],
        )
        for i, pages, fmt in cb_list
    ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [x.buffer_address(), scale1.buffer_address(), shift.buffer_address()]
    start = 0
    cw = Wt // parts  # columns per part
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    if parts == 1:  # rows [start, start + per), every column
                        row, n, c0, c1 = start, per, 0, Wt
                    else:  # one (row, part) unit: row start // parts, columns of part start % parts
                        row, n = start // parts, 1
                        c0 = (start % parts) * cw
                        c1 = c0 + cw
                    t0, nt = row * Wt + c0, n * (c1 - c0)
                    rd_rt[cx][cy] = addrs + [n, row, c0, c1]
                    wr_rt[cx][cy] = (
                        [hi.buffer_address(), lo.buffer_address(), nt, t0]
                        if split
                        else [out.buffer_address(), nt, t0]
                    )
                    cp_rt[cx][cy] = [n, c0, c1]
                    start += per
    # as binary_ng / unary / the accurate float32 reduce: float32 inputs unpacked to DEST, float32 DEST, precise SFPU
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 32
    for i in (0, 1, 2, 3, 25):
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
            kernel_source=os.path.join(_KERNELS, "writer_two_outputs.cpp") if split else os.path.join(_TT_METAL, _WRITER),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=(
                list(ttnn.TensorAccessorArgs(hi).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(lo).get_compile_time_args())
                if split
                else [16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
            ),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "ln_modulate.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[Wt, _f32_bits(1.0 / W), _f32_bits(eps), 1 if split else 0],
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    if split:
        ttnn.generic_op([x, scale1, shift, hi, lo], prog)
        return out, (hi, lo)
    return ttnn.generic_op([x, scale1, shift, out], prog)
