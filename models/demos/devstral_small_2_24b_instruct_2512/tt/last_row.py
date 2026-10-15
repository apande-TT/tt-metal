# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill's last prompt row of every user as ONE data-movement program (ttnn.generic_op, kernels/last_row.cpp):
row R of each user's (S, D) slice of the tiled hidden state (B = 32, S, D), emitted as the tiled (1, 1, 32, D) LM-head
input.

Replaces the tile-aligned 32-row block slice, its untilize and the row-major embedding gather (~125 us of three ops,
10.5 MB through DRAM twice): output tile j is column tile j of all 32 users, so each RISC reads, per user, the two
16-element face rows of row R straight out of that user's tile (a 64-byte DRAM-aligned row pair each), places them in
row b of the output tile and writes it. A pure copy, so bit-identical."""

from __future__ import annotations

from pathlib import Path

import ttnn

_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "last_row.cpp")
_USERS = 32
_SLOT_BYTES = _USERS * 2 * 64 + 2048  # a RISC's row-pair scratch + its output tile


class LastRow:
    RISCS = 2

    def __init__(self, device):
        self.device = device
        g = device.compute_with_storage_grid_size()
        self.core_xy = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    @staticmethod
    def supports(hidden) -> bool:
        b, s, d = (int(x) for x in hidden.shape)
        return (
            b == _USERS
            and hidden.dtype == ttnn.bfloat16
            and hidden.layout == ttnn.TILE_LAYOUT
            and not hidden.is_sharded()
            and s % ttnn.TILE_SIZE == 0
            and d % ttnn.TILE_SIZE == 0
        )

    def __call__(self, hidden, r):
        """hidden: (32, S, D) bf16 tiles, interleaved; r: the row. Returns (1, 1, 32, D) bf16 tiles in DRAM."""
        b, s, d = (int(x) for x in hidden.shape)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, b, d)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        dt, st = d // ttnn.TILE_SIZE, s // ttnn.TILE_SIZE
        stride = self.RISCS * len(self.core_xy)
        acc = [a for t in (hidden, out) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        args = lambda c, h: [
            hidden.buffer_address(),
            out.buffer_address(),
            self.RISCS * c + h,
            (r // ttnn.TILE_SIZE) * dt,
            r % ttnn.TILE_SIZE,
            h,
        ]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[dt, st * dt, stride] + acc,
                runtime_args=[(xy, args(c, h)) for c, xy in enumerate(self.core_xy)],
                config=cfg,
            )
            for h, cfg in enumerate(
                (
                    dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                    dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
                )
            )
        ]
        size = self.RISCS * _SLOT_BYTES + 64
        fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=size)
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=self.cores, format_descriptors=[fmt])]
        ttnn.generic_op([hidden, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out
