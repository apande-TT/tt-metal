# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode token embedding as ONE program (ttnn.generic_op, kernels/embed_decode_*.cpp), straight into the decoder's
fp32 width-sharded residual layout.

The step's embedding ran as ttnn.embedding into a tiled bf16 row -- one core, since the row is one tile tall (~25
us) -- then a typecast to fp32 and a reshard onto the first norm's width shards. Here, on the residual's shard grid,
core b < B reads token b's whole row of the row-major bf16 table (one DRAM read) and signals every core; each core
then gathers its own column slice of every row from those cores' L1, tilizes the block into fp32 tiles (bf16 -> fp32
is exact) and writes them into its own shard: the same values, in the layout the first layer reads."""

from __future__ import annotations

from pathlib import Path

import ttnn

_DIR = Path(__file__).resolve().parent / "kernels"
CB_TOK, CB_RM, CB_ROW, CB_OUT = 0, 1, 2, 16
SEM_ARRIVED = 0


class EmbedDecode:
    def __init__(self, device, memory_config, dim):
        """memory_config: the decode residual's fp32 WIDTH-sharded L1 layout (one (32, dim / cores) shard a core,
        row-major over its grid); dim: the hidden size."""
        spec = memory_config.shard_spec
        self.device, self.mem, self.dim = device, memory_config, dim
        self.cores = spec.grid
        self.wt = int(spec.shape[1]) // ttnn.TILE_SIZE
        bbox = self.cores.bounding_box()
        gx = bbox.end.x - bbox.start.x + 1
        n = self.cores.num_cores()
        self.core_xy = [(bbox.start.x + c % gx, bbox.start.y + c // gx) for c in range(n)]
        self.noc_xy = []  # every core's NOC coordinates, in shard order (the reader's signal and gather targets)
        for x, y in self.core_xy:
            w = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            self.noc_xy += [w.x, w.y]
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        self.cfg = cfg

    @staticmethod
    def supports(memory_config, dim):
        try:
            spec = memory_config.shard_spec
            return (
                memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                and spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
                and int(spec.shape[0]) == ttnn.TILE_SIZE
                and int(spec.shape[1]) % ttnn.TILE_SIZE == 0
                and int(spec.shape[1]) * spec.grid.num_cores() == dim
                and len(spec.grid.ranges()) == 1
            )
        except Exception:  # noqa: BLE001
            return False

    def _cb(self, index, dtype, page, pages):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=self.cores, format_descriptors=[fmt])

    def __call__(self, tok, table):
        """tok: (1, B) uint32 row-major ids; table: (V, dim) bf16 row-major. Returns (1, 1, B, dim) fp32 tiles in the
        residual's width-sharded layout."""
        b = int(tok.shape[-1])
        if b > len(self.core_xy):
            raise ValueError(f"EmbedDecode holds one token row a core: {b} rows > {len(self.core_xy)} cores")
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, b, self.dim)), ttnn.float32, ttnn.TILE_LAYOUT, self.device, self.mem
        )
        acc = [a for t in (tok, table) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        wt, n, row_bytes = self.wt, len(self.core_xy), self.dim * 2
        kernel = lambda name, ct, rt, config: ttnn.KernelDescriptor(
            kernel_source=str(_DIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        ta, wa, oa = tok.buffer_address(), table.buffer_address(), out.buffer_address()
        kernels = [
            kernel(
                "embed_decode_reader.cpp",
                [b, wt, CB_TOK, CB_RM, CB_ROW, SEM_ARRIVED, n, row_bytes] + acc,
                [(xy, [ta, wa, c] + self.noc_xy) for c, xy in enumerate(self.core_xy)],
                ttnn.ReaderConfigDescriptor(),
            ),
            kernel("embed_decode_compute.cpp", [wt, CB_RM, CB_OUT], [], self.cfg),
            kernel(
                "embed_decode_writer.cpp",
                [wt, CB_OUT],
                [(xy, [oa]) for xy in self.core_xy],
                ttnn.WriterConfigDescriptor(),
            ),
        ]
        cbs = [
            self._cb(CB_TOK, ttnn.uint32, 128, 1),
            self._cb(CB_RM, ttnn.bfloat16, 2048, wt),
            self._cb(CB_ROW, ttnn.bfloat16, row_bytes, 1),
            self._cb(CB_OUT, ttnn.float32, 4096, wt),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=SEM_ARRIVED, core_ranges=self.cores, initial_value=0)]
        ttnn.generic_op([tok, table, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return out
