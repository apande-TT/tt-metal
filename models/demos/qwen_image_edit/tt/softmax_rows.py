# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The transformer attention's explicit float32 softmax as one C++ Metalium kernel (ttnn.generic_op).

Replaces the chain s + mask, max, subtract, exp, sum, divide (_stubs/attention.py SOFTMAX_FN): one read of
the logits and one write of the probabilities instead of six ops over [B, H, Sq, Sk] float32. Each core takes
whole tile rows (Wt = ceil(Sk / 32) tiles) and keeps a row in L1 while it runs the five steps over it.

Bit-identical: kernels/softmax_rows.cpp makes, per step, the LLK calls of the stock program this fork picks
for these float32 operands, with its configuration: binary_ng's SFPU path (both operands float32) with the LLK
broadcast (unary_bcast<ROW> for the [B, 1, 1, Sk] mask, unary_bcast<COL> for the row max / sum) and
add / sub_binary_tile<NearestEven>, div_binary_tile; the accurate float32 reduce W (fast_and_approximate_mode
off, fp32 DEST: copy + binary_max_tile / add_binary_tile fold over the row's tiles, sfpu_reduce REDUCE_ROW,
REDUCE_ROW packer edge mask); the unary exp_tile<false>. Every buffer the compute kernel unpacks is
UnpackToDestFp32 (as those programs set their inputs), DEST is float32 and math_approx_mode is off, so no
operand is narrowed to tf32 and the intermediates the stock chain stores as float32 tensors stay float32 in
DEST or L1. Implicit tile padding (Sq or Sk not a multiple of 32) is handled as stock does: ttnn.max / ttnn.sum
fill their input's padding in place (FillPad -inf / 0 on a view of the same buffer) before reducing, so the
subtract sees -inf and the divide 0 there; the kernel applies the same exact select (fill_pad's masks,
fill_tile_bitcast + where_tile) to x's and e's border tiles, so p's padding matches too.
"""
from __future__ import annotations

import os

import ttnn

# the kernels live next to this module; the stock writer is found under TT_METAL_HOME
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TT_METAL = os.environ.get("TT_METAL_HOME", "")


def fused_softmax_rows(s, mask_add, mem, split_mem=None):
    """s [N, H, Sq, Sk] float32 TILE (interleaved), mask_add [N or 1, 1, 1, Sk] float32 TILE or None
    -> softmax(s + mask_add, dim=-1) float32 TILE in `mem`, bit-identical to the ttnn op chain (padding too).
    split_mem: -> (p, (hi, lo)) with p's bf16 limbs in split_mem, made as limb_split.fused_split_bf16 makes them,
    and p itself allocated but NOT written -- for a caller whose only use of p is that split."""
    N, H, Sq, Sk = (int(d) for d in s.shape)
    assert s.dtype == ttnn.float32 and s.layout == ttnn.TILE_LAYOUT, "float32 TILE logits"
    Wt, Ht = int(s.padded_shape[-1]) // 32, int(s.padded_shape[-2]) // 32
    assert Wt == -(-Sk // 32) and Ht == -(-Sq // 32), "only implicit tile padding"
    has_mask = mask_add is not None
    mask_batches = 1
    if has_mask:
        m = tuple(int(d) for d in mask_add.shape)
        assert len(m) == 4 and m[1] == 1 and m[2] == 1 and m[3] == Sk and m[0] in (1, N), m
        assert mask_add.dtype == ttnn.float32 and mask_add.layout == ttnn.TILE_LAYOUT
        mask_batches = m[0]
    batch = next(b for b in (4, 2, 1) if Wt % b == 0)  # logit tiles per reader barrier
    out = ttnn.allocate_tensor_on_device(s.shape, ttnn.float32, ttnn.TILE_LAYOUT, s.device(), mem)
    split = split_mem is not None
    if split:
        hi = ttnn.allocate_tensor_on_device(s.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, s.device(), split_mem)
        lo = ttnn.allocate_tensor_on_device(s.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, s.device(), split_mem)

    device = s.device()
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, N * H * Ht)
    page = 4096  # one float32 tile
    pages = [(0, 2 * batch), (2, Wt), (3, 1), (4, 1)]  # s stream, x / e row, max, sum
    pages += [(17, 2), (24, 2), (25, 2)] if split else [(16, 4)]  # split: lo, scratch hi, scratch p; else out (two pairs)
    if has_mask:
        pages.append((1, Wt))  # the current batch's mask row
    w_mod, h_mod = Sk % 32, Sq % 32
    num_pad = (w_mod != 0) + (h_mod != 0) + (w_mod != 0 and h_mod != 0)
    if num_pad:
        pages.append((5, num_pad))  # padding masks: right, bottom, corner (those present)
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=page)],
        )
        for i, n in pages
    ]
    if split:  # bf16 hi, lo and scratch hi (float32 scratch p above)
        bf = {16: 2, 17: 2, 24: 2}
        cbs = [c for c, (i, _) in zip(cbs, pages) if i not in bf] + [
            ttnn.CBDescriptor(
                total_size=n * 2048,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
            )
            for i, n in bf.items()
        ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    mask_addr = mask_add.buffer_address() if has_mask else 0
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    rd_rt[x][y] = [s.buffer_address(), mask_addr, per, start]
                    wr_rt[x][y] = (
                        [hi.buffer_address(), lo.buffer_address(), per * Wt, start * Wt]
                        if split
                        else [out.buffer_address(), per * Wt, start * Wt]
                    )
                    cp_rt[x][y] = [per, start]
                    start += per
    shape_ct = [Wt, Ht, H, mask_batches, int(has_mask), w_mod, h_mod]
    reader_ct = shape_ct + [batch] + list(ttnn.TensorAccessorArgs(s).get_compile_time_args())
    shape_ct = shape_ct + [int(split)]
    reader_ct += list(ttnn.TensorAccessorArgs(mask_add if has_mask else s).get_compile_time_args())
    # as binary_ng / unary / the accurate fp32 reduce: float32 inputs unpacked to DEST, float32 DEST, precise SFPU
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 32
    for i in (0, 1, 2, 3, 4, 5, 25):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_softmax_rows.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "writer_two_outputs.cpp")
            if split
            else os.path.join(
                _TT_METAL,
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            ),
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
            kernel_source=os.path.join(_KERNELS, "softmax_rows.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=shape_ct,
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    if split:
        ttnn.generic_op([s] + ([mask_add] if has_mask else []) + [hi, lo], prog)
        return out, (hi, lo)
    return ttnn.generic_op([s] + ([mask_add] if has_mask else []) + [out], prog)
