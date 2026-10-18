# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode (one tile row) projection as a DRAM-bank-local generic_op (kernels/dram_mm).

A few cores per DRAM bank (its optimal worker and the nearest free cores in that row) stream only tiles held by that
bank, each its own contiguous part, in large reads (the access pattern of ttnn's DRAM-sharded matmul), while the arithmetic stays that of the 1-D multicast
matmul it replaces: every output tile is one uninterrupted fp32 DEST accumulation over K in order (that factory
reloads its partials straight into DEST between K blocks), packed to bf16.

The weight is stored DRAM interleaved with its tiles permuted so that bank b's pages, in address order, are exactly
the tiles bank b's workers consume: interleaved page p lives in bank p % banks at offset (p // banks) * page, so page
banks * q + b is tile q of bank b's sequence (output-column groups of group_w tiles, K-major inside a group).
"""
from __future__ import annotations

import os

import torch

import ttnn

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "dram_mm")
_TILE = 32
_TILE_BYTES = 2 * _TILE * _TILE  # bf16
_TXN_TILES = 8  # 16 KB: one NoC packet on Blackhole
_BLOCK_TILES = 16  # weight tiles per reader block (two transactions)
_SLOTS = 3
_IN0_CHUNK = 8
_CB_IN0, _CB_IN1, _CB_OUT = 0, 1, 16
_WORKERS_PER_BANK = 2  # cores streaming disjoint contiguous parts of one bank (at most; must divide its groups)


def _mesh_kw(device, **kw):
    kw["device"] = device
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except AttributeError:
        pass
    return kw


class DramMatmul:
    """y[1, 32, N] = x[.., 32, K] @ w[K, N] for a bf16 one-tile-row x (None-able: check `ok`)."""

    def __init__(self, device, w, math_fidelity, split_n=None):
        """split_n: write output columns [0, split_n) and [split_n, N) as two tensors (two projections sharing
        one activation row run as one program)."""
        self.device = device
        self.k, self.n = w.shape
        self.split_n = split_n or self.n
        # The weight reader runs on NCRISC (ReaderConfigDescriptor: RISCV_1, its default NoC).
        workers = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.RISCV_1_default)
        banks = len(workers)
        k_t, n_t = self.k // _TILE, self.n // _TILE
        self.ok = self.k % _TILE == 0 and self.n % _TILE == 0 and n_t % banks == 0 and self.split_n % _TILE == 0
        if not self.ok:
            return
        cols = n_t // banks
        group_w = max(g for g in (4, 2, 1) if cols % g == 0)
        k_block = _BLOCK_TILES // group_w
        self.ok = k_t % k_block == 0 and (k_t % _IN0_CHUNK == 0)
        if not self.ok:
            return
        bank_groups = cols // group_w
        wpb = min(_WORKERS_PER_BANK, bank_groups)
        # Groups per worker of a bank (the first ones take one more when they do not divide evenly).
        base, extra = divmod(bank_groups, wpb)
        worker_groups = [base + (1 if i < extra else 0) for i in range(wpb)]

        # [k_t, n_t, 32, 32] tiles -> per bank, per group, K-major rows of group_w tiles.
        tiles = w.reshape(k_t, _TILE, n_t, _TILE).permute(0, 2, 1, 3)
        seq = (
            tiles.reshape(k_t, banks, bank_groups, group_w, _TILE, _TILE)
            .permute(1, 2, 0, 3, 4, 5)  # [bank, group, kt, j, 32, 32]
            .reshape(banks, -1, _TILE, _TILE)
        )
        pages = seq.permute(1, 0, 2, 3).reshape(-1, _TILE)  # page banks * q + b = seq[b, q]
        self.weight = ttnn.from_torch(
            pages.contiguous(),
            **_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )

        # Extra workers of a bank: the nearest free cores in its optimal worker's row.
        grid = device.compute_with_storage_grid_size()
        used = {(c.x, c.y) for c in workers}
        bank_cores = []
        for c in workers:
            mine = [c]
            for d in range(1, grid.x):
                for x in (c.x + d, c.x - d):
                    if len(mine) < wpb and 0 <= x < grid.x and (x, c.y) not in used:
                        used.add((x, c.y))
                        mine.append(ttnn.CoreCoord(x, c.y))
            self.ok = self.ok and len(mine) == wpb
            bank_cores.append(mine)
        if not self.ok:
            return
        all_cores = [c for mine in bank_cores for c in mine]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_cores])
        self.cores = cores
        rt = ttnn.RuntimeArgs()
        vchannels = {}
        group_bytes = k_t * group_w * _TILE_BYTES
        for b, mine in enumerate(bank_cores):
            g0 = 0
            for w_i, c in enumerate(mine):
                # Different virtual channels for workers sharing a row (deepseek matmul_wo's assignment).
                vc = (b * wpb + w_i) & 0x3
                row = [v for (x, y), v in vchannels.items() if y == c.y]
                for _ in range(4):
                    if vc not in row:
                        break
                    vc = (vc + 1) & 0x3
                vchannels[(c.x, c.y)] = vc
                ng = worker_groups[w_i]
                rt[c.x][c.y] = [b, vc, b * cols + g0 * group_w, g0 * group_bytes, ng, ng * (k_t // k_block)]
                g0 += ng
        self.rt = rt
        self.cbs = [
            ttnn.CBDescriptor(
                total_size=k_t * _TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_IN0, ttnn.bfloat16, _TILE_BYTES)],
            ),
            ttnn.CBDescriptor(
                total_size=_SLOTS * _BLOCK_TILES * _TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_IN1, ttnn.bfloat16, _TILE_BYTES)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * group_w * _TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_OUT, ttnn.bfloat16, _TILE_BYTES)],
            ),
        ]
        self.reader_ct = [_CB_IN1, _BLOCK_TILES, _BLOCK_TILES // _TXN_TILES, _TXN_TILES * _TILE_BYTES]
        self.writer_ct = [k_t, group_w, _IN0_CHUNK, _CB_IN0, _CB_OUT, self.split_n // _TILE]
        self.compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "compute_dram_mm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[k_t, group_w, k_block, _CB_IN0, _CB_IN1, _CB_OUT],
            runtime_args=rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )

    def accepts(self, x):
        if not self.ok or x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or x.memory_config().is_sharded():
            return False
        ps = list(x.padded_shape)
        rows = 1
        for d in ps[:-1]:
            rows *= d
        return ps[-1] == self.k and rows == _TILE and x.shape[-1] == self.k

    def __call__(self, x, memory_config):
        widths = [self.split_n] + ([self.n - self.split_n] if self.split_n < self.n else [])
        outs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(x.shape)[:-1] + [n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
            )
            for n in widths
        ]
        out = outs[0]
        common = [self.weight.buffer_address(), x.buffer_address(), out.buffer_address(), outs[-1].buffer_address()]
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "reader_dram_mm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=self.reader_ct,
            runtime_args=self.rt,
            common_runtime_args=common,
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "writer_dram_mm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=self.writer_ct
            + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=self.rt,
            common_runtime_args=common,
            config=ttnn.WriterConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, self.compute], semaphores=[], cbs=self.cbs)
        ttnn.generic_op([x, self.weight, *outs], program)
        return outs[0] if len(outs) == 1 else tuple(outs)
