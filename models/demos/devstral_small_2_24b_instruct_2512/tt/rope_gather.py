# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode RoPE tables as ONE data-movement program (ttnn.generic_op, kernels/rope_gather.cpp).

The fused q / k decode RoPE reads (1, 2B, 1, head_dim) cos / sin tables -- q's batch rows, then k's -- height-sharded
one row per core over its 2B cores. Each step derived them on device (position x inv_freq, cos / sin, scale, cast:
seven ops) and then concatenated and resharded them (four more). Here the cos / sin of EVERY position below the KV
capacity are computed ONCE, by those same ops (TtMinistral3RotaryEmbedding.table, so each row is bit-identical to the
step's), and each step core i reads cur_pos[i % B] and copies that position's rows into row 0 of its shard's tiles --
the only row the RoPE's row broadcast reads."""

from __future__ import annotations

from pathlib import Path

import ttnn

_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "rope_gather.cpp")


class RopeGather:
    CB = 0

    def __init__(self, device, batch, head_dim, grid_x):
        """Cores: the first 2 * batch of a grid_x-wide grid, row-major (the fused RoPE's q cores, then its k cores)."""
        self.device, self.batch, self.head_dim = device, batch, head_dim
        n = 2 * batch
        self.core_xy = [(i % grid_x, i // grid_x) for i in range(n)]
        self.cores = ttnn.num_cores_to_corerangeset(
            n, ttnn.CoreCoord(grid_x, device.compute_with_storage_grid_size().y), row_wise=True
        )

    def __call__(self, cur_pos, cos_tab, sin_tab, memory_config):
        """cur_pos: (B,) int32 row-major; cos_tab / sin_tab: (1, 1, P, head_dim) bf16 row-major. Returns the (1, 2B, 1,
        head_dim) bf16 cos / sin tiles in memory_config (height-sharded one row per core on this program's cores)."""
        shape = ttnn.Shape((1, 2 * self.batch, 1, self.head_dim))
        cos, sin = (
            ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config)
            for _ in range(2)
        )
        ct = [self.head_dim, self.CB]
        for t in (cur_pos, cos_tab, sin_tab):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        addrs = [cur_pos.buffer_address(), cos_tab.buffer_address(), sin_tab.buffer_address()]
        addrs += [cos.buffer_address(), sin.buffer_address()]
        size = 128 + 4 * self.head_dim  # positions + one cos and one sin row
        fmt = ttnn.CBFormatDescriptor(buffer_index=self.CB, data_format=ttnn.uint32, page_size=size)
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=self.cores, format_descriptors=[fmt])]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=ct,
                runtime_args=[(xy, addrs + [i % self.batch]) for i, xy in enumerate(self.core_xy)],
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        io = [cur_pos, cos_tab, sin_tab, cos, sin]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return cos, sin
