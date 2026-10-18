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
import struct

import torch

import ttnn

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "dram_mm")
_TILE = 32
_TILE_BYTES = 2 * _TILE * _TILE  # bf16
_TXN_TILES = 8  # 16 KB of bf16 (one NoC packet on Blackhole)
_BLOCK_TILES = 16  # weight tiles per reader block (two transactions)
_SLOTS = 3
_IN0_CHUNK = 8
_CB_IN0, _CB_IN1, _CB_OUT = 0, 1, 16
_WORKERS_PER_BANK = 2  # cores streaming disjoint contiguous parts of one bank (at most; must divide its groups)
_NORM_CB_BLOCKS = 10  # with a fused norm: weight CB the stream can run ahead into (20 measured slower)
_CB_XIN, _CB_SCALER, _CB_EPS, _CB_GAMMA, _CB_NORMOUT = 2, 3, 4, 5, 17
_CB_X2, _CB_EX2, _CB_EX2PE, _CB_FUSION, _CB_X2ALL, _CB_EX2PE_SEND = 24, 25, 26, 27, 28, 29
_SEM_GATHER, _SEM_BCAST, _SEM_ALL, _SEM_START, _SEM_EP = 0, 1, 2, 3, 4
_CB_EP_LHS, _CB_EP_RHS, _CB_EP_OUT = 0, 1, 16  # on the epilogue helper cores only
_NO_SEM = 0xFFFFFFFF


def _mesh_kw(device, **kw):
    kw["device"] = device
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except AttributeError:
        pass
    return kw


def _core_rects(cores):
    """The logical cores as few rectangles as possible (x-runs per row, stacked when rows repeat): a program's
    kernels go to each CoreRange of its set, so rectangles beat a list of single cores."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []  # (x0, x1, y)
    for y in sorted(rows):
        xs = sorted(rows[y])
        x0 = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                runs.append((x0, prev, y))
                x0 = x
            prev = x
        runs.append((x0, prev, y))
    rects = []  # [x0, x1, y0, y1]
    for x0, x1, y in runs:
        for r in rects:
            if r[0] == x0 and r[1] == x1 and r[3] == y - 1:
                r[3] = y
                break
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
    )


class DramMatmul:
    """y[1, 32, N] = x[.., 32, K] @ w[K, N] for a bf16 one-tile-row x (None-able: check `ok`)."""

    def __init__(
        self,
        device,
        w,
        math_fidelity,
        split_n=None,
        dtype=ttnn.bfloat16,
        workers_per_bank=None,
        norm=None,
        cb_blocks=None,
        prebuilt=None,
        group_w=None,
        residual_add=False,
    ):
        """split_n: write output columns [0, split_n) and [split_n, N) as two tensors (two projections sharing
        one activation row run as one program). norm: an r_m_s_norm.TtRMSNorm applied to x inside the program
        (its shared-statistic kernel's arithmetic, then an all-gather of the normalized row) -- only for a norm and
        projection of the same math fidelity. cb_blocks: weight CB depth in reader blocks. prebuilt: a DramMatmul
        of the same weight (and group_w) whose device copy is reused (w is then ignored). group_w: output tiles per
        DEST accumulation (default the largest of 4, 2, 1 dividing a bank's columns). residual_add: the program also
        returns residual + projection (ttnn.add's arithmetic, on helper cores that take each output tile as soon as
        its writer has stored it); __call__ then needs residual= and returns the sum."""
        self.device = device
        self.norm = norm
        if prebuilt is not None:
            w = torch.empty(prebuilt.k, prebuilt.n, device="meta")
        cb_blocks = cb_blocks or (_NORM_CB_BLOCKS if norm is not None else _SLOTS)
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
        group_w = group_w or max(g for g in (4, 2, 1) if cols % g == 0)
        self.ok = cols % group_w == 0
        if not self.ok:
            return
        k_block = _BLOCK_TILES // group_w
        self.ok = k_t % k_block == 0 and (k_t % _IN0_CHUNK == 0)
        if not self.ok:
            return
        bank_groups = cols // group_w
        wpb = min(workers_per_bank or _WORKERS_PER_BANK, bank_groups)
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
        if prebuilt is not None:
            self.weight = prebuilt.weight
        else:
            # Whole tiles are permuted, so a block-float dtype quantizes every tile as in the unpermuted weight.
            self.weight = ttnn.from_torch(
                pages.contiguous(),
                **_mesh_kw(device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            )
        w_tile_bytes = self.weight.tile.get_tile_size(dtype)

        # Extra workers of a bank: the nearest free cores to its optimal worker (same row first), taken round-robin
        # over the banks so neighbouring banks share their surroundings.
        grid = device.compute_with_storage_grid_size()
        used = {(c.x, c.y) for c in workers}
        bank_cores = [[c] for c in workers]
        for _ in range(wpb - 1):
            for mine in bank_cores:
                c = mine[0]
                free = [  # nearest first; same row, then the +x side (measured faster than -x for 2 per bank)
                    (abs(x - c.x) + abs(y - c.y), abs(y - c.y), 0 if x > c.x else 1, x, y)
                    for x in range(grid.x)
                    for y in range(grid.y)
                    if (x, y) not in used
                ]
                if not free:
                    self.ok = False
                    return
                *_, x, y = min(free)
                used.add((x, y))
                mine.append(ttnn.CoreCoord(x, y))
        all_cores = [c for mine in bank_cores for c in mine]
        cores = _core_rects(all_cores)
        self.cores = cores
        rt = ttnn.RuntimeArgs()
        vchannels = {}
        group_bytes = k_t * group_w * w_tile_bytes
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
        self.semaphores = []
        self.residual_add = residual_add and self.split_n == self.n
        self.ep = None
        if self.residual_add:
            free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in used]
            ep_tpc = -(-n_t // len(free))
            helpers = free[: -(-n_t // ep_tpc)]
            ep_rt = ttnn.RuntimeArgs()
            ep_xy = []
            for i, c in enumerate(helpers):
                ep_rt[c.x][c.y] = [i * ep_tpc, min(ep_tpc, n_t - i * ep_tpc)]
                p = device.worker_core_from_logical_core(c)
                ep_xy += [p.x, p.y]
            ep_cores = _core_rects(helpers)
            self.ep = dict(tpc=ep_tpc, cores=ep_cores, rt=ep_rt, xy=ep_xy)
        if norm is not None:
            n_cores = len(all_cores)
            self.ok = norm.consts is not None
            if not self.ok:
                return
            # The norm runs on the first n_norm cores (the most that split its Kt tiles evenly), all receive its row.
            n_norm = max(d for d in range(1, n_cores + 1) if k_t % d == 0)
            tpc = k_t // n_norm
            self.noc_xy = []
            for i, c in enumerate(all_cores):  # all_cores[0] is the reducer
                args = list(rt[c.x][c.y]) + [i * tpc, 1 if i == 0 else 0, 1 if i < n_norm else 0]
                rt[c.x][c.y] = args
                p = device.worker_core_from_logical_core(c)
                self.noc_xy += [p.x, p.y]
            self.semaphores = [
                ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0)
                for i in (_SEM_GATHER, _SEM_BCAST, _SEM_ALL, _SEM_START)
            ]
        self.cbs = [
            ttnn.CBDescriptor(
                total_size=k_t * _TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_IN0, ttnn.bfloat16, _TILE_BYTES)],
            ),
            ttnn.CBDescriptor(
                total_size=cb_blocks * _BLOCK_TILES * w_tile_bytes,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_IN1, dtype, w_tile_bytes)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * group_w * _TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_OUT, ttnn.bfloat16, _TILE_BYTES)],
            ),
        ]
        self.reader_ct = [_CB_IN1, _BLOCK_TILES, _BLOCK_TILES // _TXN_TILES, _TXN_TILES * w_tile_bytes, cb_blocks]
        self.reader_ct += [_SEM_START if norm is not None else _NO_SEM]
        if norm is None:
            self.writer_src = "writer_dram_mm.cpp"
            self.writer_ct = [k_t, group_w, _IN0_CHUNK, _CB_IN0, _CB_OUT, self.split_n // _TILE]
            ep_off = 4
            compute_src = "compute_dram_mm.cpp"
            compute_ct = [k_t, group_w, k_block, _CB_IN0, _CB_IN1, _CB_OUT]
        else:
            f32_tile = 4 * _TILE * _TILE

            def cb(idx, n, fmt, size):
                return ttnn.CBDescriptor(
                    total_size=n * size, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(idx, fmt, size)]
                )

            bf, f32 = ttnn.bfloat16, ttnn.float32
            self.cbs += [
                cb(_CB_XIN, tpc, bf, _TILE_BYTES),
                cb(_CB_SCALER, 1, bf, _TILE_BYTES),
                cb(_CB_EPS, 1, bf, _TILE_BYTES),
                cb(_CB_GAMMA, tpc, bf, _TILE_BYTES),
                cb(_CB_NORMOUT, tpc, bf, _TILE_BYTES),
                cb(_CB_X2, tpc, f32, f32_tile),
                cb(_CB_EX2, 1, f32, f32_tile),
                cb(_CB_EX2PE, 1, f32, f32_tile),
                cb(_CB_FUSION, tpc, f32, f32_tile),
                cb(_CB_X2ALL, k_t, f32, f32_tile),
                cb(_CB_EX2PE_SEND, 1, f32, f32_tile),
            ]
            self.writer_src = "writer_dram_mm_norm.cpp"
            self.writer_ct = [k_t, group_w, _CB_IN0, _CB_OUT, self.split_n // _TILE, tpc]
            self.writer_ct += [_CB_XIN, _CB_SCALER, _CB_EPS, _CB_GAMMA, _CB_X2, _CB_X2ALL, _CB_EX2PE, _CB_EX2PE_SEND]
            self.writer_ct += [_CB_NORMOUT, _SEM_GATHER, _SEM_BCAST, _SEM_ALL, n_cores, n_norm, _SEM_START]
            ep_off = 8 + 2 * n_cores
            inv_w_bits = struct.unpack("<I", struct.pack("<f", 1.0 / self.k))[0]
            compute_src = "compute_dram_mm_norm.cpp"
            compute_ct = [k_t, group_w, k_block, _CB_IN0, _CB_IN1, _CB_OUT, inv_w_bits, tpc]
            compute_ct += [_CB_XIN, _CB_SCALER, _CB_EPS, _CB_GAMMA, _CB_X2, _CB_EX2, _CB_EX2PE, _CB_FUSION]
            compute_ct += [_CB_X2ALL, _CB_EX2PE_SEND, _CB_NORMOUT, math_fidelity.value]
            math_fidelity = ttnn.MathFidelity.HiFi4  # the norm's (as r_m_s_norm's kernel); the matmul keeps its own
        self.writer_ct += [1, _SEM_EP, self.ep["tpc"], ep_off] if self.ep else [0, 0, 1, 0]
        if self.ep:
            everywhere = cores.merge(self.ep["cores"])
            self.semaphores.append(ttnn.SemaphoreDescriptor(id=_SEM_EP, core_ranges=everywhere, initial_value=0))
            self.cbs += [
                ttnn.CBDescriptor(
                    total_size=self.ep["tpc"] * _TILE_BYTES,
                    core_ranges=self.ep["cores"],
                    format_descriptors=[ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, _TILE_BYTES)],
                )
                for idx in (_CB_EP_LHS, _CB_EP_RHS, _CB_EP_OUT)
            ]
            self.ep["compute"] = ttnn.KernelDescriptor(
                kernel_source=os.path.join(_KERNEL_DIR, "compute_ep_add.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.ep["cores"],
                compile_time_args=[_CB_EP_LHS, _CB_EP_RHS, _CB_EP_OUT],
                runtime_args=self.ep["rt"],
                config=ttnn.ComputeConfigDescriptor(  # ttnn.add's (binary_ng) for bf16 operands
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    dst_full_sync_en=False,
                ),
            )
        self.compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, compute_src),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=compute_ct,
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

    def __call__(self, x, memory_config, out_shape=None, residual=None):
        """out_shape: the (single) output's shape when it is not [.., 32, N] but has the same tile pages in order."""
        widths = [self.split_n] + ([self.n - self.split_n] if self.split_n < self.n else [])
        shapes = [out_shape] if out_shape is not None else [list(x.shape)[:-1] + [n] for n in widths]
        outs = [
            ttnn.allocate_tensor_on_device(ttnn.Shape(sh), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config)
            for sh in shapes
        ]
        out = outs[0]
        common = [self.weight.buffer_address(), x.buffer_address(), out.buffer_address(), outs[-1].buffer_address()]
        writer_ct = self.writer_ct + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        writer_ct += list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        inputs = [x, self.weight]
        if self.norm is not None:
            g, c = self.norm.weight, self.norm.consts
            common += [g.buffer_address(), c.buffer_address()] + self.noc_xy[:2] + self.noc_xy
            writer_ct += list(ttnn.TensorAccessorArgs(g).get_compile_time_args())
            writer_ct += list(ttnn.TensorAccessorArgs(c).get_compile_time_args())
            inputs += [g, c]
        kernels = []
        final = None
        if self.ep is not None:
            assert residual is not None and list(residual.padded_shape) == list(out.padded_shape)
            final = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(out.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
            )
            common += self.ep["xy"]
            ep_common = [residual.buffer_address(), out.buffer_address(), final.buffer_address()]
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=os.path.join(_KERNEL_DIR, "reader_ep_add.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.ep["cores"],
                    compile_time_args=[_CB_EP_LHS, _CB_EP_RHS, _SEM_EP]
                    + list(ttnn.TensorAccessorArgs(residual).get_compile_time_args())
                    + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
                    runtime_args=self.ep["rt"],
                    common_runtime_args=ep_common,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=os.path.join(_KERNEL_DIR, "writer_ep_add.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.ep["cores"],
                    compile_time_args=[_CB_EP_OUT] + list(ttnn.TensorAccessorArgs(final).get_compile_time_args()),
                    runtime_args=self.ep["rt"],
                    common_runtime_args=ep_common,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                self.ep["compute"],
            ]
            inputs += [residual]
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
            kernel_source=os.path.join(_KERNEL_DIR, self.writer_src),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=writer_ct,
            runtime_args=self.rt,
            common_runtime_args=common,
            config=ttnn.WriterConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(
            kernels=[reader, writer, self.compute] + kernels, semaphores=self.semaphores, cbs=self.cbs
        )
        if final is not None:
            ttnn.generic_op([*inputs, out, final], program)
            ttnn.deallocate(out)  # the projection alone, read only by the helpers
            return final
        ttnn.generic_op([*inputs, *outs], program)
        return outs[0] if len(outs) == 1 else tuple(outs)
