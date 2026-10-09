# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sum of the n equal source blocks of a float32 tensor as one C++ Metalium kernel (ttnn.generic_op).

Used for the local part of the transformer's exact all_to_all TP reduce (_ccl.A2A_SUM_FN): one pass
over the blocks instead of n slices + n - 1 ttnn.add calls. The compute kernel unpacks the float32 tiles
to DEST (UnpackToDestFp32) and adds them with the SFPU float32 add (round to nearest even) in source
order, the same adds as the ttnn.add chain, so the result is bit-identical.
"""
from __future__ import annotations

import os

import ttnn

# the kernels live next to this module; the stock writer is found under TT_METAL_HOME
_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TT_METAL = os.environ.get("TT_METAL_HOME", "")


def fused_block_sum(g, n, part, mem):
    """g [n * part[0], ...] float32 TILE -> sum of its n blocks of shape `part` (block i = source i)."""
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(part), ttnn.float32, ttnn.TILE_LAYOUT, g.device(), mem)
    block_tiles = _tiles(out)
    reader_ct = [block_tiles, n] + list(ttnn.TensorAccessorArgs(g).get_compile_time_args())
    return _sum_program([g], out, n, "reader_sum_blocks.cpp", reader_ct)


def fused_sum_bias(a, b, bias_rows, mem):
    """(a + b) + bias for float32 TILE a, b of one shape [..., N] and bias_rows [32, N] float32 (the bias row
    replicated over a tile's rows): the limb products' sum and the bias add of a precise linear in one pass,
    the same two SFPU float32 adds in the same order as ttnn.add(ttnn.add(a, b), bias)."""
    out = ttnn.allocate_tensor_on_device(a.shape, ttnn.float32, ttnn.TILE_LAYOUT, a.device(), mem)
    reader_ct = [int(a.padded_shape[-1]) // 32]
    for t in (a, b, bias_rows):
        reader_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    return _sum_program([a, b, bias_rows], out, 3, "reader_sum_bias.cpp", reader_ct)


def _tiles(t):
    vol = 1
    for d in t.padded_shape:
        vol *= int(d)
    return vol // 1024


def _sum_program(srcs, out, n, reader, reader_ct):
    """kernels/sum_blocks.cpp over the n source tiles `reader` streams per output tile of `out`; the reader's
    runtime args are the source addresses, then (tile count, first tile)."""
    device = out.device()
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, _tiles(out))
    page = 4096  # one float32 tile
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=page)],
        )
        for i, pages in ((0, 2 * n), (16, 2))  # the n source tiles of an output tile double-buffered
    ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in srcs]
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    rd_rt[x][y] = addrs + [per, start]
                    wr_rt[x][y] = [out.buffer_address(), per, start]
                    cp_rt[x][y] = [per]
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
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "sum_blocks.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[n],
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op(srcs + [out], prog)
