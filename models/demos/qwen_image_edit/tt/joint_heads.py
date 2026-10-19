# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The joint attention's head split as one C++ Metalium data-movement program (ttnn.generic_op).

The transformer's text and image streams each project to [B, S, W] (q | k | v column blocks, heads side by
side). The attention needs every head of q / k / v over the joint [text, image] sequence: [B, H, L + S, D].
The stock path takes it per stream by H slices + reshapes + a concat, then concatenates the two streams
along the sequence. This program writes the joint tensor straight from the two projections: each output
tile is a copy of one source tile (kernels/reader_joint_heads.cpp into the stock interleaved writer), so the
result is bit-identical.
"""
from __future__ import annotations

import os

import ttnn

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TT_METAL = os.environ.get("TT_METAL_HOME", "")
_BATCH = 8  # tiles per read barrier


def fused_joint_heads(txt, img, off, heads, head_dim, mem):
    """txt [B, L, W], img [B, S, W] TILE (same dtype) -> [B, heads, L + S, head_dim]: head h is the column block
    off + h * head_dim of either stream, the text rows first. None when the shapes are not whole tiles."""
    tb, ib = [int(d) for d in txt.padded_shape], [int(d) for d in img.padded_shape]
    ts, is_ = [int(d) for d in txt.shape], [int(d) for d in img.shape]
    if txt.dtype != img.dtype or txt.dtype not in (ttnn.float32, ttnn.bfloat16):
        return None
    if len(tb) != 3 or len(ib) != 3 or tb[0] != ib[0] or tb[2] != ib[2]:
        return None
    if ts != tb or is_ != ib or off % 32 or head_dim % 32 or off + heads * head_dim > tb[2]:
        return None
    B, L, S, W = tb[0], tb[1], ib[1], tb[2]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, heads, L + S, head_dim]), txt.dtype, ttnn.TILE_LAYOUT, txt.device(), mem
    )
    reader_ct = [_BATCH] + list(ttnn.TensorAccessorArgs(txt).get_compile_time_args())
    reader_ct += list(ttnn.TensorAccessorArgs(img).get_compile_time_args())
    shape_rt = [L // 32, S // 32, W // 32, heads, head_dim // 32, off // 32]
    return _copy_program([txt, img], out, "reader_joint_heads.cpp", reader_ct, shape_rt)


def _copy_program(srcs, out, reader, reader_ct, shape_rt):
    """`reader` streams out's tiles in order (one source tile each) into cb 16; the stock interleaved writer stores
    them. Reader runtime args: the source addresses, (tile count, first tile), then shape_rt (shapes and offsets are
    runtime args so every call of a reader shares one program binary in the trace)."""
    n_tiles = 1
    for d in out.padded_shape:
        n_tiles *= int(d)
    n_tiles //= 1024
    grid = out.device().compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, n_tiles)
    page = 4096 if out.dtype == ttnn.float32 else 2048
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * _BATCH * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=out.dtype, page_size=page)],
        )
    ]
    rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in srcs]
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    rd_rt[x][y] = addrs + [per, start] + shape_rt
                    wr_rt[x][y] = [out.buffer_address(), per, start]
                    start += per
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, reader),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(
                _TT_METAL,
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            ),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op(srcs + [out], prog)
