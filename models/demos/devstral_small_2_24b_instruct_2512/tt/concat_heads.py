# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill head concat as ONE data-movement program (ttnn.generic_op, kernels/concat_heads.cpp): SDPA's head-major
output (B, H, S, hd) -> (B, 1, S, H * hd), the layout nlp_concat_heads emits.

The input is L1-interleaved, so every one of its tiles already sits in some core's L1. Each data-movement RISC of
every core pushes the tiles of its own core's bank(s) straight to their concatenated slots: one local-source NOC write
a tile, no read round trip and no circular buffer (nlp_concat_heads reads every tile into a circular buffer on 64
cores and writes it back out). A pure tile permutation, so bit-identical."""

from __future__ import annotations

from pathlib import Path

import ttnn

_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "concat_heads.cpp")
_PAGE = 2048  # bf16 tile


class ConcatHeads:
    RISCS = 2  # data-movement RISCs a core (alternate pages of its banks)

    def __init__(self, device):
        self.device = device
        g = device.compute_with_storage_grid_size()
        self.core_xy = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    @staticmethod
    def supports(x) -> bool:
        b, h, s, hd = (int(d) for d in x.shape)
        mem = x.memory_config()
        return (
            x.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and mem.buffer_type == ttnn.BufferType.L1
            and not x.is_sharded()
            and s % ttnn.TILE_SIZE == 0
            and hd % ttnn.TILE_SIZE == 0
        )

    def __call__(self, x, memory_config):
        """x: (B, H, S, hd) bf16 tiles, L1-interleaved -> (B, 1, S, H * hd) bf16 tiles in memory_config."""
        b, h, s, hd = (int(d) for d in x.shape)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((b, 1, s, h * hd)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        dt, st = hd // ttnn.TILE_SIZE, s // ttnn.TILE_SIZE
        acc = [a for t in (x, out) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        rt = [(xy, [x.buffer_address(), out.buffer_address()]) for xy in self.core_xy]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[dt, st, h, b * h * st * dt, _PAGE, j, self.RISCS] + acc,
                runtime_args=rt,
                config=cfg,
            )
            for j, cfg in enumerate(
                (
                    dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                    dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
                )
            )
        ]
        ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=[], semaphores=[]))
        return out
