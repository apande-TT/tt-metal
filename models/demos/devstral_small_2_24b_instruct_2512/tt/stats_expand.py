# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill norm's gathered stats rebuilt from the all_gather of tt/add_stats' compacted stats (COMPACT_STATS), as
ONE data-movement program (ttnn.generic_op, kernels/stats_expand.cpp): the (..., S, ND * 32) tensor the all_gather of
the whole stats tiles produced -- tile (k, d) = device d's stats tile k, its 32 row sums in column 0 and exact zeros
elsewhere -- so the sum and the rsqrt chain after it run on the same values. The gather moves 1/32 of the bytes."""

from __future__ import annotations

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_TILE = 32


def expand(g, x):
    """g: (1, 1, rt, ND * 32) fp32 gathered compact stats of the residual x (rt = x's tile rows); returns the
    (x.shape[:-1], ND * 32) fp32 gathered stats tiles, in g's memory config."""
    device = g.device()
    shape = [int(s) for s in x.shape]
    rt = int(g.shape[-2])
    nd = int(g.shape[-1]) // _TILE
    pages = rt * nd
    grid = device.compute_with_storage_grid_size()
    ncores = min(int(grid.x) * int(grid.y), pages)
    nmax = -(-pages // ncores)
    ncores = -(-pages // nmax)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape[:-1] + [nd * _TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, g.memory_config()
    )

    def cb(index, size):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)
        return ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[fmt])

    acc = [a for t in (g, out) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    ga, oa = g.buffer_address(), out.buffer_address()
    rt_args = []
    for c in range(ncores):
        p0 = c * nmax
        cy, cx = divmod(c, int(grid.x))
        rt_args.append(((cx, cy), [ga, oa, p0, min(nmax, pages - p0)]))
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "stats_expand.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[nd, nmax] + acc,
        runtime_args=rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    ttnn.generic_op(
        [g, out], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb(0, nmax * 128), cb(1, 2 * 4096)], semaphores=[])
    )
    return out
