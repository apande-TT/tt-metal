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


_DECODE_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "concat_heads_decode.cpp")


class ConcatHeadsDecode:
    """Decode head concat as ONE data-movement program (kernels/concat_heads_decode.cpp): SDPA-decode's L1-interleaved
    (1, B, H, hd) output -- batch row b's heads in rows 0..H-1 of its tile row -- -> the (1, 1, B, H * hd) bf16
    L1-interleaved o_proj input. Replaces the reshard onto the head shards, nlp_concat_heads_decode and the unshard
    (three ops, ~6.5 us). Output tile q = h * DT + d takes, in row b, row h of input tile b * DT + d: one core an output
    tile, each of its data-movement RISCs gathering 16 rows (two 32-byte face rows each) into a local scratch laid out
    as its two output faces, then writing them out at once. A pure copy, so bit-identical."""

    CB = 0

    def __init__(self, device):
        self.device = device
        self.grid = device.compute_with_storage_grid_size()

    def supports(self, x) -> bool:
        if len(x.shape) != 4:
            return False
        one, b, h, hd = (int(d) for d in x.shape)
        mem = x.memory_config()
        return (
            one == 1
            and b <= ttnn.TILE_SIZE
            and h <= ttnn.TILE_SIZE
            and x.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and mem.buffer_type == ttnn.BufferType.L1
            and not x.is_sharded()
            and hd % ttnn.TILE_SIZE == 0
            and h * hd // ttnn.TILE_SIZE <= self.grid.x * self.grid.y
        )

    def __call__(self, x):
        """x: (1, B, H, hd) bf16 tiles, L1-interleaved -> (1, 1, B, H * hd) bf16 tiles, L1-interleaved."""
        _, b, h, hd = (int(d) for d in x.shape)
        dt = hd // ttnn.TILE_SIZE
        n_out = h * dt
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, b, h * hd)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG
        )
        gx = self.grid.x
        core_xy = [(q % gx, q // gx) for q in range(n_out)]  # the row-major order num_cores_to_corerangeset takes
        cores = ttnn.num_cores_to_corerangeset(n_out, self.grid, row_wise=True)
        fmt = ttnn.CBFormatDescriptor(buffer_index=self.CB, data_format=ttnn.bfloat16, page_size=_PAGE)
        cbs = [ttnn.CBDescriptor(total_size=_PAGE, core_ranges=cores, format_descriptors=[fmt])]
        acc = [a for t in (x, out) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        rt = [(xy, [x.buffer_address(), out.buffer_address(), q]) for q, xy in enumerate(core_xy)]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_DECODE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[dt, b, half, self.CB] + acc,
                runtime_args=rt,
                config=cfg,
            )
            for half, cfg in enumerate(
                (
                    dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                    dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
                )
            )
        ]
        ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out
