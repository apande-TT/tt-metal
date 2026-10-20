# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode MLP down projection as ONE program (ttnn.generic_op): y = m @ W_down for the single tile row of decode
tokens (m: the fused gate / up's silu(gate) * up product), bit-identical to ttnn's 1D-multicast matmul.

The same streaming scheme as tt/gate_up_decode.py: a spare core reads the [32, K] product once and multicasts it to
the compute cores (kernels/gate_up_sender.cpp); each compute core owns two output columns and each of its
data-movement RISCs streams one of them (kernels/gate_up_stream.cpp) from a BANK-MAJOR weight copy, RB K rows a push
= one contiguous read; compute (kernels/down_compute.cpp) accumulates both columns over the whole K in fp32 DEST."""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.demos.devstral_small_2_24b_instruct_2512.tt.gate_up_decode import (
    _TILE_BYTES,
    bank_major_order,
    dual_noc_sender_kernels,
    stream_order,
)

_KDIR = Path(__file__).resolve().parent / "kernels"


class StreamDownDecode:
    NC = 2  # output columns a compute core (one a data-movement RISC)
    RB = 16  # K rows a weight push (bf16)
    # K rows a push by weight dtype: 8 tiles (~9 KB block-float8, ~4.6 KB block-float4) stream fastest (bf8 down,
    # measured: 35 KB pushes 120 us, 17 KB 117 us, 9 KB 110.5 us, 4 KB 111 us; bf4 at 8 rows gained on the trace)
    RB_BY_DTYPE = {ttnn.bfloat8_b: 8, ttnn.bfloat4_b: 8}
    CB_BYTES = 112 * 1024  # per weight CB (DEPTH = pushes that fit)
    GRID_X = 8  # compute cores: an 8-wide rectangle (as the 1D down's 8 x 10)
    # SPREAD: the columns over SPREAD_CORES cores (the grid's first, row-major; the sender stays the last core) instead
    # of NT / NC: every bank serves the same number of columns, from its cores c % NB -- as many with two columns (one
    # a RISC) as that takes, the rest with one, whose pushes alternate between the two RISCs (GU_PSTEP 2,
    # DOWN_ONE_COL). More readers stream the weight faster (the gate / up's bf8 stream: 64 -> 104 cores, 415 -> 466
    # GB/s); each column still accumulates over K in order in one DEST tile: bit-identical.
    SPREAD = True
    SPREAD_CORES = 104
    # spread: the sender core multicasts half the activation on each NOC (gate_up_sender.cpp X_SLICE): one RISC
    # multicasting the whole 512 KB row of the down landed it ~40 us (row 9: ~60 us) into the program
    DUAL_NOC = True
    # ar_call: RISCV_0's share of a two-column core's second column, in pushes of its Kt / RB (the rest is RISCV_1's)
    SPLIT_B_SHARE = 20 / 32

    @classmethod
    def rb_for(cls, dtype) -> int:
        return cls.RB_BY_DTYPE.get(dtype, cls.RB)

    @classmethod
    def supports(cls, device, k, n, dtype) -> bool:
        """k x n: one device's down weight."""
        kt, nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        g = device.compute_with_storage_grid_size()
        n_cores = nt // cls.NC
        return (
            dtype in _TILE_BYTES
            and k % ttnn.TILE_SIZE == 0
            and nt % cls.NC == 0
            and n_cores % cls.GRID_X == 0
            and cls.GRID_X < g.x
            and n_cores // cls.GRID_X <= g.y
            and nt % device.dram_grid_size().x == 0
            and kt % cls.rb_for(dtype) == 0
        )

    def __init__(self, device, w_down, dtype, math_fidelity, put, tp=1, rb=None, ar_fold=False):
        """w_down: [tp * K, N] torch weight (device d multiplies rows d * K .. (d + 1) * K); put(t, dtype) uploads a
        torch tensor sharded over the mesh on its second-to-last dim. rb: K rows a push (default: rb_for(dtype)); a
        short K takes fewer rows a push, so each column still streams as several pushes. ar_fold: lay the columns out
        for ar_call (the all_reduce and the residual add folded in) instead of __call__."""
        import torch

        self.device = device
        k, n = int(w_down.shape[0]) // tp, int(w_down.shape[1])
        self.kt, self.nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        self.RB = int(rb) if rb and self.kt % int(rb) == 0 else self.rb_for(dtype)
        self.nb = nb = device.dram_grid_size().x
        g = device.compute_with_storage_grid_size()
        rng = lambda a, b, c, d: ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
        phys = lambda x, y: device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        spread = self.SPREAD and self._spread_ok(g)
        self.spread = spread
        if spread:
            n_cores = self.SPREAD_CORES
            self.core_xy = [(c % g.x, c // g.x) for c in range(n_cores)]
            # per bank: `two` cores with two columns, the rest with one
            per_bank, cores_bank = self.nt // nb, n_cores // nb
            two = per_bank - cores_bank
            self.widths = [2 if c // nb < two else 1 for c in range(n_cores)]
            self.banks = [c % nb for c in range(n_cores)]
            full, rem = divmod(n_cores, g.x)
            rects = [(0, 0, g.x - 1, full - 1)] + ([(0, full, rem - 1, full)] if rem else [])
        else:
            n_cores = self.nt // self.NC
            gx, gy = self.GRID_X, n_cores // self.GRID_X
            self.core_xy = [(c % gx, c // gx) for c in range(n_cores)]
            self.widths = [self.NC] * n_cores
            self.banks = None
            rects = [(0, 0, gx - 1, gy - 1)]
        # the columns of core c: c0[c] .. c0[c] + widths[c] - 1; ar_fold (see ar_call): core c's first column is c, a
        # two-column core's second n_cores + c (each core's first column is done a wave before the second ones)
        self.ar_fold = bool(ar_fold) and spread
        self.c0, self.cols, j = [], [], 0
        for c, w in enumerate(self.widths):
            self.c0.append(j)
            self.cols.append([c] + ([n_cores + c] if w == 2 else []) if self.ar_fold else list(range(j, j + w)))
            j += w
        assert j == self.nt, (j, self.nt)
        owner = {col: c for c, cs in enumerate(self.cols) for col in cs}
        assert sorted(owner) == list(range(self.nt))
        # the column streams (nc = 1): column j from its core's bank
        streams = [(col, 1, self.banks[owner[col]] if spread else col % nb) for col in range(self.nt)]
        order, self.page0s = stream_order(self.kt, self.nt, nb, streams, self.RB)
        self.w = put(torch.cat([bank_major_order(w_down[d * k : (d + 1) * k], order) for d in range(tp)], dim=0), dtype)
        del order
        self.cores = ttnn.CoreRangeSet([rng(*r) for r in rects])
        one = lambda xy: rng(*xy, *xy)
        self.groups = [  # (width, core indices, core set): each width's kernels are built for it
            (w, idx, ttnn.CoreRangeSet([one(self.core_xy[c]) for c in idx]))
            for w in sorted(set(self.widths))
            for idx in [[c for c in range(n_cores) if self.widths[c] == w]]
        ]
        self.sender = (g.x - 1, g.y - 1)
        self.sender_set = ttnn.CoreRangeSet([rng(*self.sender, *self.sender)])
        self.all = self.cores.merge(self.sender_set)
        snd = phys(*self.sender)
        self.sender_xy = [snd.x, snd.y]
        self.rects = [len(rects)]  # the sender's multicast targets (NOC 0 order corners) and core counts
        for a, b, c, d in rects:
            lo, hi = phys(a, b), phys(c, d)
            self.rects += [lo.x, lo.y, hi.x, hi.y, (c - a + 1) * (d - b + 1)]
        self.w_tile = _TILE_BYTES[dtype]
        self.depth = max(2, self.CB_BYTES // (self.RB * self.w_tile))
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = math_fidelity
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        self.cfg = cfg

    def _n_send(self):
        return 2 if self.DUAL_NOC and self.spread else 1

    def _sender_kernels(self, m, sem_ready, sem_valid, dm0, x_acc):
        kt, n_cores = self.kt, len(self.core_xy)
        if self.DUAL_NOC and self.spread:
            return dual_noc_sender_kernels(
                self.device, self.sender, kt, 2048, n_cores, sem_ready, sem_valid, m, self.rects, x_acc
            )
        return [
            self._kernel(
                "gate_up_sender.cpp",
                self.sender_set,
                [kt, 2048, n_cores, sem_ready, sem_valid] + x_acc,
                [(self.sender, [m.buffer_address()] + self.rects)],
                dm0,
            )
        ]

    def _spread_ok(self, g) -> bool:
        n, nb = self.SPREAD_CORES, self.nb
        per_bank = self.nt // nb
        return (
            n < g.x * g.y  # and a spare last core for the sender
            and n % nb == 0
            and self.nt % nb == 0
            and n // nb <= per_bank <= 2 * (n // nb)
            and self.kt % (2 * self.RB) == 0
        )

    def _cb(self, index, tiles, page, dtype):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
        return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=self.all, format_descriptors=[fmt])

    def _kernel(self, name, cores, ct, rt, config, defines=()):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
            defines=list(defines),
        )

    def __call__(self, m, memory_config=None):
        """m: (1, 1, 32, K) bf16 tiles (interleaved); returns m @ W_down, (1, 1, 32, N) bf16 in L1 (memory_config:
        an L1 layout to write it in instead, e.g. the width shards the decode all_reduce reads)."""
        shape = tuple(int(d) for d in m.shape)
        if shape[:-1] != (1, 1, ttnn.TILE_SIZE) or m.dtype != ttnn.bfloat16 or m.is_sharded():
            raise ValueError(f"StreamDownDecode takes one interleaved bf16 tile row, got {shape} {m.dtype}")
        if self.ar_fold:
            raise NotImplementedError("an ar_fold StreamDownDecode lays its columns out for ar_call")
        kt, nt, rb, w_tile, depth, nb = self.kt, self.nt, self.RB, self.w_tile, self.depth, self.nb
        x_tile, o_tile = 2048, 2048
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            memory_config or ttnn.L1_MEMORY_CONFIG,
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        wdt = self.w.dtype
        cbs = [  # x first: the sender writes it at its own address of it, the same L1 address on every core
            self._cb(0, kt, x_tile, ttnn.bfloat16),
            self._cb(1, depth * rb, w_tile, wdt),
            self._cb(2, depth * rb, w_tile, wdt),
            self._cb(16, self.NC, o_tile, ttnn.bfloat16),
        ]
        SEM_READY, SEM_VALID = 0, 1
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all, initial_value=0) for i in (SEM_READY, SEM_VALID)]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        # one weight column a push (nc = 1); the OWNER (RISCV_1) writes the core's output tiles
        kernels = self._sender_kernels(m, SEM_READY, SEM_VALID, dm0, acc(m))
        nsend = [("GU_NSEND", str(self._n_send()))]
        for w, idx, cores in self.groups:
            # two columns: RISC j streams column j; one column: RISC j streams its pushes j, j + 2, ...
            ct = lambda cb, owner: [kt, nb, 1, rb, w_tile, cb, depth, owner, SEM_READY, SEM_VALID, o_tile, w]
            page0 = lambda c, j: self.page0s[self.c0[c] + j] if w == 2 else self.page0s[self.c0[c]] + j * rb * nb
            rt = lambda j: [
                (
                    self.core_xy[c],
                    [self.w.buffer_address(), self.c0[c], out.buffer_address()] + self.sender_xy + [page0(c, j)],
                )
                for c in idx
            ]
            one = [] if w == 2 else [("GU_PSTEP", "2")]
            kernels += [
                self._kernel("gate_up_stream.cpp", cores, ct(1, 0) + acc(self.w, out), rt(0), dm0, one + nsend),
                self._kernel("gate_up_stream.cpp", cores, ct(2, 1) + acc(self.w, out), rt(1), dm1, one + nsend),
                self._kernel(
                    "down_compute.cpp", cores, [kt, rb], [], self.cfg, [] if w == 2 else [("DOWN_ONE_COL", "1")]
                ),
            ]
        ttnn.generic_op([m, self.w, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return out

    # ar_call: the two fabric-link workers (free cores of the spread grid: the stream cores fill rows 0.. row-major,
    # the sender is the last core)
    AR_WORKERS = ((8, 9), (9, 9))

    def ar_ok(self, m, x, ar) -> bool:
        """Whether ar_call can take (m, x, ar): x the fp32 decode residual in ar's width shards (WT tiles a core, every
        residual core a two-column core), ar the DecodeAllReduce whose scratch and global semaphore it uses."""
        if not self.ar_fold or ar is None or ar.tp % 2 or ar.num_links != len(self.AR_WORKERS):
            return False
        g = self.device.compute_with_storage_grid_size()
        used = {self.sender} | set(self.core_xy)
        if any(c in used or c[0] >= g.x or c[1] >= g.y for c in self.AR_WORKERS):
            return False
        xs, ss = x.memory_config().shard_spec if x.is_sharded() else None, ar.mem.shard_spec
        if xs is None or x.dtype != ttnn.float32 or len(xs.grid.ranges()) != 1 or xs.grid != ss.grid:
            return False
        wt = int(ss.shape[1]) // ttnn.TILE_SIZE
        r0 = xs.grid.ranges()[0]
        rw = r0.end.x - r0.start.x + 1
        n_res = xs.grid.num_cores()
        res = [(r0.start.x + i % rw, r0.start.y + i // rw) for i in range(n_res)]
        idx = {xy: c for c, xy in enumerate(self.core_xy)}
        return (
            tuple(xs.shape) == tuple(ss.shape)
            and n_res * wt == self.nt
            and all(xy in idx and self.widths[idx[xy]] == 2 for xy in res)
            and (len(self.core_xy) % wt == 0)
            and tuple(int(d) for d in m.shape) == (1, 1, ttnn.TILE_SIZE, self.kt * ttnn.TILE_SIZE)
        )

    def ar_call(self, m, x, ar):
        """m: (1, 1, 32, K) bf16 tiles (interleaved, the gate / up product); x: the fp32 decode residual in ar.mem's
        width shards. Returns the new residual x + all_reduce(m @ W_down) (fp32 in x's shards): the down projection,
        ttnn.experimental.all_reduce_async of its bf16 partials and the residual add, bit for bit, in one program.

        Each core's columns stream one at a time (kernels/down_ar_stream.cpp): its first column -- wave 1, the shards
        of columns 0 .. n_cores - 1 -- is done about two thirds of the way through, and the fabric-link workers
        (kernels/ar_send_waves.cpp, shards alternating between them) send those shards to every device while the
        two-column cores' second columns (wave 2) still stream; only wave 2 is exchanged after the projection. Then
        each residual core sums its shard's TP slots and adds them to x (kernels/down_compute.cpp RES_TAIL)."""
        kt, nt, rb, w_tile, depth, nb = self.kt, self.nt, self.RB, self.w_tile, self.depth, self.nb
        x_tile, o_tile, f32_tile, ar_tile = 2048, 2048, 4096, 2048
        tp, links = ar.tp, ar.num_links
        n_cores = len(self.core_xy)
        spec = x.memory_config().shard_spec
        wt = int(spec.shape[1]) // ttnn.TILE_SIZE
        shard_bytes = wt * ar_tile
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ar.mem
        )
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape(tuple(int(d) for d in x.shape)), ttnn.float32, ttnn.TILE_LAYOUT, self.device, x.memory_config()
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        rng = lambda a, b, c, d: ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
        one = lambda xy: rng(*xy, *xy)
        phys = lambda c: self.device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        # residual shard i on core i of x's shard grid (row-major); its columns are i * wt .. i * wt + wt - 1
        r0 = spec.grid.ranges()[0]
        rw = r0.end.x - r0.start.x + 1
        n_shards = spec.grid.num_cores()
        res_logical = [(r0.start.x + i % rw, r0.start.y + i // rw) for i in range(n_shards)]
        res_xy = [phys(c) for c in res_logical]
        shard_of = {xy: i for i, xy in enumerate(res_logical)}
        w1_shards = n_cores // wt  # columns 0 .. n_cores - 1: wave 1
        # link l sends shards i % links == l: wave 1 first, then wave 2
        link_shards = [
            [i for i in range(w1_shards) if i % links == l] + [i for i in range(w1_shards, n_shards) if i % links == l]
            for l in range(links)
        ]
        link_n1 = [sum(1 for i in ls if i < w1_shards) for ls in link_shards]
        workers = ttnn.CoreRangeSet([one(c) for c in self.AR_WORKERS])
        link_xy = [v for c in self.AR_WORKERS for v in (phys(c).x, phys(c).y)]
        hdr_bytes = ttnn.get_tt_fabric_packet_header_size_bytes()
        is_res = [self.core_xy[c] in shard_of for c in range(n_cores)]
        groups = []  # (columns, residual?, core indices)
        for w in sorted(set(self.widths)):
            for r in (False, True):
                idx = [c for c in range(n_cores) if self.widths[c] == w and is_res[c] == r]
                if idx:
                    groups.append((w, r, idx, ttnn.CoreRangeSet([one(self.core_xy[c]) for c in idx])))
        res_set = ttnn.CoreRangeSet([one(c) for c in res_logical])
        everyone = self.all.merge(workers)
        wdt = self.w.dtype
        mk_cb = lambda index, tiles, page, dtype, cores: ttnn.CBDescriptor(
            total_size=tiles * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        cbs = [  # x first: the sender writes it at its own address of it, the same L1 address on every core
            mk_cb(0, kt, x_tile, ttnn.bfloat16, self.all),
            mk_cb(1, depth * rb, w_tile, wdt, self.cores),
            mk_cb(2, depth * rb, w_tile, wdt, self.cores),
            mk_cb(16, 2, o_tile, ttnn.bfloat16, self.cores),
            mk_cb(11, wt, f32_tile, ttnn.float32, res_set),
            mk_cb(12, tp * wt, ar_tile, ttnn.bfloat16, res_set),
            mk_cb(13, wt, ar_tile, ttnn.bfloat16, res_set),
            mk_cb(14, wt, f32_tile, ttnn.float32, res_set),
            mk_cb(17, max(len(ls) for ls in link_shards), shard_bytes, ttnn.bfloat16, workers),
            mk_cb(18, 2, hdr_bytes, ttnn.uint32, workers),
        ]
        SEM_READY, SEM_VALID, SEM_W1, SEM_W2 = 0, 1, 2, 3
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        scratch_addr, sem_addr = ar.buffer.buffer_address(), ttnn.get_global_semaphore_address(ar.sem)
        kernels = self._sender_kernels(m, SEM_READY, SEM_VALID, dm0, acc(m))
        nsend = [("GU_NSEND", str(self._n_send()))]
        n_push = kt // rb
        n0 = max(1, min(n_push - 1, round(n_push * self.SPLIT_B_SHARE)))
        # (count, step) of each RISC's pushes: first column alternating, second split at n0
        pattern = {0: (n_push // 2, 2, n0, 1), 1: (n_push // 2, 2, n_push - n0, 1)}
        for w, r, idx, cores in groups:
            ct = lambda cb, owner: [
                kt, nb, rb, w_tile, cb, depth, owner, SEM_READY, SEM_VALID, o_tile, w, SEM_W1, SEM_W2, n_cores,
                int(r), tp, wt, ar_tile, f32_tile,
            ] + list(pattern[owner]) + acc(self.w, out)
            tail = lambda c: [x.buffer_address(), scratch_addr, h.buffer_address(), sem_addr] if r else []
            rt = lambda f: [
                (
                    self.core_xy[c],
                    [self.w.buffer_address(), out.buffer_address()]
                    + self.sender_xy
                    + [self.page0s[col] + (f if i == 0 else f * n0) * rb * nb for i, col in enumerate(self.cols[c])]
                    + [0] * (2 - w)
                    + self.cols[c]
                    + [0] * (2 - w)
                    + link_xy
                    + tail(c),
                )
                for c in idx
            ]
            defines = [("DOWN_SEQ", str(w)), ("SPLIT_B", str(n0))] + ([("RES_TAIL", "1")] if r else [])
            kernels += [
                self._kernel("down_ar_stream.cpp", cores, ct(1, 0), rt(0), dm0, nsend),
                self._kernel("down_ar_stream.cpp", cores, ct(2, 1), rt(1), dm1, nsend),
                self._kernel("down_compute.cpp", cores, [kt, rb, tp, wt], [], self.cfg, defines),
            ]
        node = lambda d: self.device.get_fabric_node_id(ttnn.MeshCoordinate(0, d))
        worker_rt = lambda l: [out.buffer_address(), scratch_addr, sem_addr] + [
            v for i in link_shards[l] for v in (res_xy[i].x, res_xy[i].y)
        ]
        mpd = ttnn.MeshProgramDescriptor()
        for d in range(tp):
            # ring neighbours and line-multicast reach (all_reduce_async's Ring split, alternating by device parity)
            fwd_n = -(-(tp - 1) // 2)
            bwd_n = tp - 1 - fwd_n
            if d % 2 == 0:
                fwd_n, bwd_n = bwd_n, fwd_n
            route = lambda n: [1, n, 0, 0, 0, 0]  # 1D fabric: start one hop out, reach n devices
            link_kernels = [
                self._kernel(
                    "ar_send_waves.cpp",
                    ttnn.CoreRangeSet([one(c)]),
                    [d, 18, 17, shard_bytes, link_n1[l], len(link_shards[l]) - link_n1[l], SEM_W1, SEM_W2, wt]
                    + route(fwd_n)
                    + route(bwd_n),
                    [(c, worker_rt(l))],
                    dm0,
                )
                for l, c in enumerate(self.AR_WORKERS)
            ]
            sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=everyone, initial_value=0) for i in range(4)]
            pd = ttnn.ProgramDescriptor(kernels=kernels + link_kernels, cbs=cbs, semaphores=sems)
            for l, c in enumerate(self.AR_WORKERS):
                args = pd.kernels[len(kernels) + l].runtime_args[c[0]][c[1]]
                for nbr, n in (((d + 1) % tp, fwd_n), ((d - 1) % tp, bwd_n)):
                    args.append(int(n > 0))
                    if n > 0:
                        args.extend(ttnn.setup_fabric_connection(node(d), node(nbr), l, pd, ttnn.CoreCoord(*c)))
            coord = ttnn.MeshCoordinate(0, d)
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = pd
        ttnn.generic_op([m, self.w, out, x, ar.buffer, h], mpd)
        ttnn.deallocate(out)
        return h
