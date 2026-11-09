# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The latents' 2 x 2 pack / unpack as one C++ data-movement kernel (kernels/latent_pack_rows.cpp, ttnn.generic_op).

On ROW_MAJOR 32-bit tensors the 6-D permute of _pack_latents / _unpack_latents keeps its size-2 innermost dim, so
ttnn moves 8-byte rows one at a time (~2.6 ms per call for 32 x [16, 32, 32] float32 latents). The kernel takes one
(b, i) pair of image rows per unit, reads its whole rows, rearranges the words in L1 and writes whole rows: the same
values in the same places (pure data movement).
"""
from __future__ import annotations

import os

import ttnn

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def latent_pack_rows(x, B, C, H, W, unpack=False):
    """pack: ROW_MAJOR float32 x [B, C, H, W] -> [B, (H / 2)(W / 2), 4C]; unpack (unpack=True): the inverse, x
    [B, (H / 2)(W / 2), 4C] -> [B, C, H, W]. None when the shapes do not fit (the caller permutes)."""
    if x.layout != ttnn.ROW_MAJOR_LAYOUT or x.dtype != ttnn.float32 or x.is_sharded() or H % 2 or W % 2:
        return None
    if (W * 4) % 32 or (4 * C * 4) % 32:  # whole-row NOC transfers stay 32-byte aligned
        return None
    shape = [B, C, H, W] if unpack else [B, (H // 2) * (W // 2), 4 * C]
    if [int(d) for d in x.shape] != ([B, (H // 2) * (W // 2), 4 * C] if unpack else [B, C, H, W]):
        return None
    device = x.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, device, x.memory_config())
    unit_bytes = 2 * C * W * 4  # a unit's rows, either side
    grid = device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, B * (H // 2))
    cbs = [
        ttnn.CBDescriptor(
            total_size=unit_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=unit_bytes)],
        )
        for i in (0, 1)
    ]
    ct = [1 if unpack else 0, C, H, W]
    for t in (x, out):
        ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    rt[cx][cy] = [x.buffer_address(), out.buffer_address(), per, start]
                    start += per
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "latent_pack_rows.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    return ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
