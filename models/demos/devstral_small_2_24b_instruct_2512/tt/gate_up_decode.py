# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode MLP front half as ONE program (ttnn.generic_op, kernels/gate_up_*.cpp): silu(x @ W_gate) * (x @ W_up) for
the single tile row of decode tokens, replacing two 1D-multicast ttnn.linear calls and the silu-multiply.

ttnn's 1D-multicast matmul makes every core wait on an activation multicast (~1.3 us) for each K block, so its
weight reader only runs a block ahead of compute: the block-float4 gate / up streamed at ~275 GB/s (86 us each).
Here a spare core reads the [32, K] activation once and multicasts it whole, and on each compute core the two
data-movement RISCs stream the weights independently of it -- gate on RISCV_0 (NOC 0), up on RISCV_1 (NOC 1, which
also receives x and writes the output) -- each into a CB DEPTH pushes deep. Each core owns NC gate and NC up columns
and accumulates both over the whole K in fp32 DEST (full-sync DEST: 2 * NC tiles), applies silu(gate) * up on the
SFPU in fp32 and rounds once to bf16 (ttnn's chain rounds gate and up to bf16 first).

The weights are BANK-MAJOR copies (bank_major): an interleaved DRAM buffer puts page p in bank p % NB at slot p // NB,
so the tiles are ordered such that each push of a core (RB K rows x its NC columns, row-major) occupies consecutive
slots of one bank, and the push is ONE contiguous read instead of RB * NC tile reads (per-tile reads of the 576-byte
block-float4 tiles left the stream at ~350 GB/s). Core c reads bank c % NB, so every bank serves n_cores / NB cores."""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def bank_major(w: torch.Tensor, nb: int, nc: int, rb: int) -> torch.Tensor:
    """One device's [K, N] weight -> the same [K, N] shape with its tiles permuted so that the interleaved pages
    (row-major tile order) put each stream's pushes in consecutive slots of bank stream % nb: stream s = q * nb + b
    (the stream reading columns s * nc .. s * nc + nc - 1), push p, tile t (row r = t // nc, column j = t % nc, i.e.
    weight tile (p * rb + r, s * nc + j)) sits at page ((q * P + p) * T + t) * nb + b, T = rb * nc tiles a push."""
    k, n = w.shape
    kt, nt = k // 32, n // 32
    n_streams, pushes, t_push = nt // nc, kt // rb, rb * nc
    c = torch.arange(n_streams).view(-1, 1, 1)
    p = torch.arange(pushes).view(1, -1, 1)
    t = torch.arange(t_push).view(1, 1, -1)
    page = ((c // nb * pushes + p) * t_push + t) * nb + c % nb
    tile = (p * rb + t // nc) * nt + c * nc + t % nc
    order = torch.empty(kt * nt, dtype=torch.long)
    order[page.flatten()] = tile.flatten()
    tiles = w.reshape(kt, 32, nt, 32).permute(0, 2, 1, 3).reshape(kt * nt, 32, 32)
    return tiles[order].reshape(kt, nt, 32, 32).permute(0, 2, 1, 3).reshape(k, n)


def bank_major_page0(stream: int, kt: int, nb: int, nc: int, rb: int) -> int:
    """First page of a stream's first push in bank_major's layout (each next push starts rb * nc * nb pages later)."""
    return (stream // nb) * (kt // rb) * rb * nc * nb + stream % nb


class FusedGateUpDecode:
    NC = 4  # output columns a compute core (gate and up each): 2 * NC tiles fill the full-sync fp32 DEST
    RB = 8  # K rows a weight push (bf16)
    # K rows a push by weight dtype: ~4.5 KB pushes (one K row of the NC columns: 4 block-float8 / 8 block-float4
    # tiles), many in flight per CB, stream faster than few large ones (bf8 gate / up, measured: 35 KB pushes 225 us,
    # 17 KB 220 us, 9 KB 218 us, 4.4 KB 216 us; the bf4 layers gained on the decode trace too)
    RB_BY_DTYPE = {ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 2}
    CB_BYTES = 256 * 1024  # per weight CB (DEPTH = pushes that fit; 112 KB: 218.5 us, 256 KB: 214.7 us at bf8)

    @classmethod
    def rb_for(cls, dtype) -> int:
        return cls.RB_BY_DTYPE.get(dtype, cls.RB)

    @classmethod
    def supports(cls, device, k, n, dtype) -> bool:
        """k x n: one device's gate (= up) weight."""
        kt, nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        g = device.compute_with_storage_grid_size()
        n_cores = nt // cls.NC
        return (
            dtype in _TILE_BYTES
            and k % ttnn.TILE_SIZE == 0
            and nt % cls.NC == 0
            and n_cores < g.x * g.y  # and a spare core for the sender
            and n_cores % device.dram_grid_size().x == 0
            and kt % cls.rb_for(dtype) == 0
        )

    def __init__(self, device, w_gate, w_up, dtype, math_fidelity, put, tp=1):
        """w_gate, w_up: [K, tp * N] torch weights (device d multiplies columns d * N .. (d + 1) * N); put(t, dtype)
        uploads a torch tensor sharded over the mesh on its last dim."""
        self.device = device
        self.RB = self.rb_for(dtype)
        k, n = int(w_gate.shape[0]), int(w_gate.shape[1]) // tp
        self.kt, self.nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        self.nb = nb = device.dram_grid_size().x
        bm = lambda w: torch.cat(
            [bank_major(w[:, d * n : (d + 1) * n], nb, self.NC, self.RB) for d in range(tp)], dim=-1
        )
        self.wg, self.wu = put(bm(w_gate), dtype), put(bm(w_up), dtype)
        n_cores = self.nt // self.NC
        g = device.compute_with_storage_grid_size()
        # Compute cores: the grid's first n_cores, row-major; the sender: the grid's last core.
        self.core_xy = [(c % g.x, c // g.x) for c in range(n_cores)]
        full, rem = divmod(n_cores, g.x)
        rects = [(0, 0, g.x - 1, full - 1)] if full else []
        if rem:
            rects.append((0, full, rem - 1, full))
        rng = lambda a, b, c, d: ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
        self.cores = ttnn.CoreRangeSet([rng(*r) for r in rects])
        sender = (g.x - 1, g.y - 1)
        self.sender = sender
        self.sender_set = ttnn.CoreRangeSet([rng(*sender, *sender)])
        self.all = self.cores.merge(self.sender_set)
        snd = device.worker_core_from_logical_core(ttnn.CoreCoord(*sender))
        self.sender_xy = [snd.x, snd.y]
        self.rects = [len(rects)]  # the sender's multicast targets: NOC 0 order corners and core count
        for a, b, c, d in rects:
            lo = device.worker_core_from_logical_core(ttnn.CoreCoord(a, b))
            hi = device.worker_core_from_logical_core(ttnn.CoreCoord(c, d))
            self.rects += [lo.x, lo.y, hi.x, hi.y, (c - a + 1) * (d - b + 1)]
        self.w_tile = _TILE_BYTES[dtype]
        self.depth = max(2, self.CB_BYTES // (self.RB * self.NC * self.w_tile))
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = math_fidelity
        cfg.fp32_dest_acc_en = True
        cfg.dst_full_sync_en = True
        cfg.math_approx_mode = False
        self.cfg = cfg

    def _cb(self, index, tiles, page, dtype):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
        return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=self.all, format_descriptors=[fmt])

    def _kernel(self, name, cores, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    def __call__(self, x):
        """x: (1, 1, 32, K) bf16 tiles (interleaved); returns silu(x W_gate) * (x W_up), (1, 1, 32, N) bf16 in L1."""
        shape = tuple(int(d) for d in x.shape)
        if shape[:-1] != (1, 1, ttnn.TILE_SIZE) or x.dtype != ttnn.bfloat16 or x.is_sharded():
            raise ValueError(f"FusedGateUpDecode takes one interleaved bf16 tile row, got {shape} {x.dtype}")
        kt, nt, nc, rb, w_tile, depth, nb = self.kt, self.nt, self.NC, self.RB, self.w_tile, self.depth, self.nb
        x_tile, o_tile = 2048, 2048
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.L1_MEMORY_CONFIG,
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        wdt = self.wg.dtype
        # x first: it is written remotely at the sender's own address of it, so it sits at the same L1 address on
        # every core
        cbs = [
            self._cb(0, kt, x_tile, ttnn.bfloat16),
            self._cb(1, depth * rb * nc, w_tile, wdt),
            self._cb(2, depth * rb * nc, w_tile, wdt),
            self._cb(16, nc, o_tile, ttnn.bfloat16),
        ]
        SEM_READY, SEM_VALID = 0, 1
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all, initial_value=0) for i in (SEM_READY, SEM_VALID)]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        stream_ct = lambda cb, owner, w: [
            kt,
            nb,
            nc,
            rb,
            w_tile,
            cb,
            depth,
            owner,
            SEM_READY,
            SEM_VALID,
            o_tile,
            nc,
        ] + acc(w, out)
        page0 = lambda c: bank_major_page0(c, kt, nb, nc, rb)
        rt = lambda w: [
            ((cx, cy), [w.buffer_address(), c * nc, out.buffer_address()] + self.sender_xy + [page0(c)])
            for c, (cx, cy) in enumerate(self.core_xy)
        ]
        kernels = [
            self._kernel(
                "gate_up_sender.cpp",
                self.sender_set,
                [kt, x_tile, len(self.core_xy), SEM_READY, SEM_VALID] + acc(x),
                [(self.sender, [x.buffer_address()] + self.rects)],
                dm0,
            ),
            self._kernel("gate_up_stream.cpp", self.cores, stream_ct(1, 0, self.wg), rt(self.wg), dm0),
            self._kernel("gate_up_stream.cpp", self.cores, stream_ct(2, 1, self.wu), rt(self.wu), dm1),
            self._kernel("gate_up_compute.cpp", self.cores, [kt, nc, rb], [], self.cfg),
        ]
        ttnn.generic_op([x, self.wg, self.wu, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return out



class NormGateUpDecode(FusedGateUpDecode):
    """FusedGateUpDecode with the post-attention RMSNorm folded into its prologue: it takes the decode residual h itself
    (fp32, width-sharded, WT tiles a shard) instead of the normed bf16 row, so the weight streams start while the row
    is still being normed (a separate rms_norm + unshard ran ~8.6 us first).

    NP norm cores on rows the program leaves free (NORM_ORIGIN, NORM_SHAPE; kernels/norm_shard_*.cpp) each read one
    residual shard, square and row-sum it in fp32, swap those row sums with the other norm cores, add all of them in
    fp32, scale by 1 / hidden, add eps, take rsqrt and multiply their shard by it (FPU column broadcast, as rms_norm's
    sharded kernel does), then write the bf16 result into their K slot of the activation sender's x CB and count it
    there; the sender (gate_up_sender.cpp, X_GATHER) waits for all NP slices and multicasts the row exactly as it
    multicasts a row it read itself. Stream cores, weights and compute are FusedGateUpDecode's, unchanged."""

    NORM_ORIGIN = (0, 6)
    NORM_SHAPE = (10, 4)  # NP = 40 cores, 4 K tiles each at K = 160

    @classmethod
    def supports_norm(cls, device, k, n, dtype) -> bool:
        if not cls.supports(device, k, n, dtype):
            return False
        g = device.compute_with_storage_grid_size()
        (ox, oy), (nx, ny) = cls.NORM_ORIGIN, cls.NORM_SHAPE
        n_cores = (n // ttnn.TILE_SIZE) // cls.NC
        used_rows = -(-n_cores // g.x)  # the stream cores fill rows 0.. row-major; the sender is the last core
        return (
            k % (nx * ny * ttnn.TILE_SIZE) == 0
            and oy >= used_rows
            and ox + nx <= g.x
            and oy + ny <= g.y
            and (ox + nx < g.x or oy + ny < g.y)  # not the sender (g.x - 1, g.y - 1)
        )

    def __init__(self, device, w_gate, w_up, dtype, math_fidelity, put, tp, eps, hidden):
        super().__init__(device, w_gate, w_up, dtype, math_fidelity, put, tp)
        import struct

        (ox, oy), (nx, ny) = self.NORM_ORIGIN, self.NORM_SHAPE
        self.np = nx * ny
        self.wt = self.kt // self.np  # tiles of a residual shard
        self.norm_xy_logical = [(ox + i % nx, oy + i // nx) for i in range(self.np)]
        rng = lambda a, b, c, d: ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
        self.norm_set = ttnn.CoreRangeSet([rng(ox, oy, ox + nx - 1, oy + ny - 1)])
        self.n_all = self.all.merge(self.norm_set)
        phys = lambda x, y: device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        # the gather core (norm core 0, the origin corner) and the other norm cores as rectangles it multicasts to
        gc = phys(ox, oy)
        self.gather_xy = [gc.x, gc.y]
        others = [(ox + 1, oy, ox + nx - 1, oy), (ox, oy + 1, ox + nx - 1, oy + ny - 1)]
        self.others = [len(others)]
        for a, b, c, d in others:
            lo, hi = phys(a, b), phys(c, d)
            self.others += [lo.x, lo.y, hi.x, hi.y, (c - a + 1) * (d - b + 1)]
        bits = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]
        self.inv_n_bits, self.eps_bits = bits(1.0 / hidden), bits(float(eps))
        ncfg = ttnn.ComputeConfigDescriptor()
        ncfg.math_fidelity = ttnn.MathFidelity.HiFi4
        ncfg.fp32_dest_acc_en = True
        ncfg.math_approx_mode = False
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[4] = ttnn.UnpackToDestMode.UnpackToDestFp32  # cb_h: fp32 straight to DEST for the SFPU square
        ncfg.unpack_to_dest_mode = modes
        self.ncfg = ncfg

    def accepts(self, h) -> bool:
        """h: the (1, 1, 32, K) fp32 decode residual, width-sharded into self.np shards of self.wt tiles."""
        if tuple(int(d) for d in h.shape) != (1, 1, ttnn.TILE_SIZE, self.kt * ttnn.TILE_SIZE):
            return False
        if h.dtype != ttnn.float32 or not h.is_sharded():
            return False
        mem = h.memory_config()
        spec = mem.shard_spec
        return (
            mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            and spec is not None
            and tuple(spec.shape) == (ttnn.TILE_SIZE, self.wt * ttnn.TILE_SIZE)
        )

    def _ncb(self, index, tiles, page, dtype, cores):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
        return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=cores, format_descriptors=[fmt])

    def norm_call(self, h):
        """h: the decode residual as accepts() requires; returns silu(norm(h) W_gate) * (norm(h) W_up), (1, 1, 32, N)
        bf16 in L1 (unit-gamma RMSNorm: its gamma is folded into the weights)."""
        kt, nt, nc, rb, w_tile, depth, nb = self.kt, self.nt, self.NC, self.RB, self.w_tile, self.depth, self.nb
        x_tile, o_tile, f32_tile = 2048, 2048, 4096
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.L1_MEMORY_CONFIG,
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        wdt = self.wg.dtype
        norm, everyone = self.norm_set, self.n_all
        # x first, on every core (the norm cores write into the sender's at their own address of it)
        cbs = [
            self._ncb(0, kt, x_tile, ttnn.bfloat16, everyone),
            self._ncb(3, self.np, f32_tile, ttnn.float32, norm),
            self._ncb(1, depth * rb * nc, w_tile, wdt, self.all),
            self._ncb(2, depth * rb * nc, w_tile, wdt, self.all),
            self._ncb(16, nc, o_tile, ttnn.bfloat16, self.all),
            self._ncb(4, self.wt, f32_tile, ttnn.float32, norm),
            self._ncb(5, self.wt, f32_tile, ttnn.float32, norm),
            self._ncb(6, self.wt, f32_tile, ttnn.float32, norm),
            self._ncb(7, 1, f32_tile, ttnn.float32, norm),
            self._ncb(8, 1, f32_tile, ttnn.float32, norm),
            self._ncb(9, 1, f32_tile, ttnn.float32, norm),
            self._ncb(10, self.wt, x_tile, ttnn.bfloat16, norm),
        ]
        SEM_READY, SEM_VALID, SEM_PART, SEM_GATHER, SEM_R = 0, 1, 2, 3, 4
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=everyone, initial_value=0) for i in range(5)]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        stream_ct = lambda cb, owner, w: [
            kt,
            nb,
            nc,
            rb,
            w_tile,
            cb,
            depth,
            owner,
            SEM_READY,
            SEM_VALID,
            o_tile,
            nc,
        ] + acc(w, out)
        page0 = lambda c: bank_major_page0(c, kt, nb, nc, rb)
        rt = lambda w: [
            ((cx, cy), [w.buffer_address(), c * nc, out.buffer_address()] + self.sender_xy + [page0(c)])
            for c, (cx, cy) in enumerate(self.core_xy)
        ]
        norm_rt = [
            (xy, [i, h.buffer_address()] + self.sender_xy + self.gather_xy + self.others)
            for i, xy in enumerate(self.norm_xy_logical)
        ]
        kern = lambda name, cores, ct, rt_, cfg, defines=(): ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt_,
            config=cfg,
            defines=list(defines),
        )
        kernels = [
            kern(
                "gate_up_sender.cpp",
                self.sender_set,
                [kt, x_tile, len(self.core_xy), SEM_READY, SEM_VALID, SEM_GATHER],
                [(self.sender, [0] + self.rects)],
                dm0,
                [("X_GATHER", str(self.np))],
            ),
            kern("gate_up_stream.cpp", self.cores, stream_ct(1, 0, self.wg), rt(self.wg), dm0),
            kern("gate_up_stream.cpp", self.cores, stream_ct(2, 1, self.wu), rt(self.wu), dm1),
            kern("gate_up_compute.cpp", self.cores, [kt, nc, rb], [], self.cfg),
            kern(
                "norm_shard_dm.cpp",
                norm,
                [self.wt, self.np, x_tile, f32_tile, SEM_PART, SEM_GATHER, SEM_R] + acc(h),
                norm_rt,
                dm0,
            ),
            kern(
                "norm_shard_compute.cpp",
                norm,
                [self.wt, self.np, self.inv_n_bits, self.eps_bits],
                [(xy, [int(i == 0)]) for i, xy in enumerate(self.norm_xy_logical)],
                self.ncfg,
            ),
        ]
        ttnn.generic_op([h, self.wg, self.wu, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return out

    # The attention's o_proj all_reduce folded in (ar_norm_call): one fabric-link worker core a link, on cores the
    # program leaves free (stream cores: rows 0..5 up to column 8; norm cores: columns 0..9 of rows 6..9; the sender is
    # the last core).
    AR_WORKERS = ((10, 6), (10, 7))

    def ar_accepts(self, x, p, ar) -> bool:
        """Whether ar_norm_call can fold the all_reduce `ar` (tt/decode_all_reduce.py) of the partials p in here."""
        if ar is None or ar.tp % 2 or len(self.AR_WORKERS) != ar.num_links or not self.accepts(x):
            return False
        g = self.device.compute_with_storage_grid_size()
        used = {self.sender} | set(self.core_xy) | set(self.norm_xy_logical)
        if any(c in used or c[0] >= g.x or c[1] >= g.y for c in self.AR_WORKERS):
            return False
        xs, ps, ss = x.memory_config().shard_spec, p.memory_config().shard_spec, ar.mem.shard_spec
        return (
            p.dtype == ttnn.bfloat16
            and p.is_sharded()
            and tuple(p.shape) == tuple(x.shape)
            and ps is not None
            and ps.grid == xs.grid == ss.grid
            and tuple(ps.shape) == tuple(xs.shape) == tuple(ss.shape)
            and len(xs.grid.ranges()) == 1
            and xs.grid.num_cores() == self.np
            and self.np % ar.num_links == 0
        )

    def ar_norm_call(self, x, p, ar):
        """x: the decode residual (fp32, accepts()); p: this device's o_proj partials (bf16, in x's width shards);
        ar: the DecodeAllReduce whose scratch and global semaphore the exchange uses (ar_accepts). Returns (h, m): the new
        residual h = x + all_reduce(p) (fp32 in x's shards; all_reduce_async's sum + ttnn.add, bit for bit) and
        m = silu(norm(h) W_gate) * (norm(h) W_up) as norm_call returns it.

        Each device's link workers (kernels/ar_send.cpp) multicast its partial shards into slot `device` of every
        device's scratch at program launch, so the exchange runs under the weight streams instead of before them; norm
        core i waits for the TP slots of residual shard i, sums them, adds them to x (kernels/norm_shard_ar_*.cpp) and
        norms the result as norm_call does."""
        kt, nt, nc, rb, w_tile, depth, nb = self.kt, self.nt, self.NC, self.RB, self.w_tile, self.depth, self.nb
        x_tile, o_tile, f32_tile, ar_tile = 2048, 2048, 4096, 2048
        tp, wt, links = ar.tp, self.wt, ar.num_links
        shard_bytes = wt * ar_tile
        n_res = self.np // links
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.L1_MEMORY_CONFIG,
        )
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape(tuple(int(d) for d in x.shape)), ttnn.float32, ttnn.TILE_LAYOUT, self.device, x.memory_config()
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        wdt = self.wg.dtype
        norm, everyone = self.norm_set, self.n_all
        rng = lambda a, b, c, d: ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
        workers = ttnn.CoreRangeSet([rng(*c, *c) for c in self.AR_WORKERS])
        hdr_bytes = ttnn.get_tt_fabric_packet_header_size_bytes()
        cbs = [
            self._ncb(0, kt, x_tile, ttnn.bfloat16, everyone),
            self._ncb(3, self.np, f32_tile, ttnn.float32, norm),
            self._ncb(1, depth * rb * nc, w_tile, wdt, self.all),
            self._ncb(2, depth * rb * nc, w_tile, wdt, self.all),
            self._ncb(16, nc, o_tile, ttnn.bfloat16, self.all),
            self._ncb(4, wt, f32_tile, ttnn.float32, norm),
            self._ncb(5, wt, f32_tile, ttnn.float32, norm),
            self._ncb(6, wt, f32_tile, ttnn.float32, norm),
            self._ncb(7, 1, f32_tile, ttnn.float32, norm),
            self._ncb(8, 1, f32_tile, ttnn.float32, norm),
            self._ncb(9, 1, f32_tile, ttnn.float32, norm),
            self._ncb(10, wt, x_tile, ttnn.bfloat16, norm),
            self._ncb(11, wt, f32_tile, ttnn.float32, norm),
            self._ncb(12, tp * wt, ar_tile, ttnn.bfloat16, norm),
            self._ncb(13, wt, ar_tile, ttnn.bfloat16, norm),
            self._ncb(14, wt, f32_tile, ttnn.float32, norm),
            self._ncb(17, n_res, shard_bytes, ttnn.bfloat16, workers),
            self._ncb(18, 2, hdr_bytes, ttnn.uint32, workers),
        ]
        SEM_READY, SEM_VALID, SEM_PART, SEM_GATHER, SEM_R = 0, 1, 2, 3, 4
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        stream_ct = lambda cb, owner, w: [kt, nb, nc, rb, w_tile, cb, depth, owner, SEM_READY, SEM_VALID, o_tile, nc] + acc(
            w, out
        )
        page0 = lambda c: bank_major_page0(c, kt, nb, nc, rb)
        rt = lambda w: [
            ((cx, cy), [w.buffer_address(), c * nc, out.buffer_address()] + self.sender_xy + [page0(c)])
            for c, (cx, cy) in enumerate(self.core_xy)
        ]
        # residual shard i lives on core i of x's shard grid (row-major); norm core i pairs with it
        r0 = x.memory_config().shard_spec.grid.ranges()[0]
        rw = r0.end.x - r0.start.x + 1
        phys = lambda c: self.device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        res_xy = [phys((r0.start.x + i % rw, r0.start.y + i // rw)) for i in range(self.np)]
        norm_xy = [phys(c) for c in self.norm_xy_logical]
        scratch_addr, sem_addr = ar.buffer.buffer_address(), ttnn.get_global_semaphore_address(ar.sem)
        norm_rt = [
            (
                xy,
                [i, res_xy[i].x, res_xy[i].y, x.buffer_address(), scratch_addr, h.buffer_address(), sem_addr]
                + self.sender_xy
                + self.gather_xy
                + self.others,
            )
            for i, xy in enumerate(self.norm_xy_logical)
        ]
        kern = lambda name, cores, ct, rt_, cfg, defines=(): ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt_,
            config=cfg,
            defines=list(defines),
        )
        worker_rt = lambda l: [p.buffer_address(), scratch_addr, sem_addr] + [
            v
            for i in range(l * n_res, (l + 1) * n_res)
            for v in (res_xy[i].x, res_xy[i].y, norm_xy[i].x, norm_xy[i].y)
        ]
        node = lambda d: self.device.get_fabric_node_id(ttnn.MeshCoordinate(0, d))
        mpd = ttnn.MeshProgramDescriptor()
        for d in range(tp):
            # ring neighbours and line-multicast reach (all_reduce_async's Ring split, alternating by device parity)
            fwd_n = -(-(tp - 1) // 2)
            bwd_n = tp - 1 - fwd_n
            if d % 2 == 0:
                fwd_n, bwd_n = bwd_n, fwd_n
            route = lambda n: [1, n, 0, 0, 0, 0]  # 1D fabric: start one hop out, reach n devices
            kernels = [
                kern(
                    "gate_up_sender.cpp",
                    self.sender_set,
                    [kt, x_tile, len(self.core_xy), SEM_READY, SEM_VALID, SEM_GATHER],
                    [(self.sender, [0] + self.rects)],
                    dm0,
                    [("X_GATHER", str(self.np))],
                ),
                kern("gate_up_stream.cpp", self.cores, stream_ct(1, 0, self.wg), rt(self.wg), dm0),
                kern("gate_up_stream.cpp", self.cores, stream_ct(2, 1, self.wu), rt(self.wu), dm1),
                kern("gate_up_compute.cpp", self.cores, [kt, nc, rb], [], self.cfg),
                kern(
                    "norm_shard_ar_dm.cpp",
                    norm,
                    [wt, self.np, x_tile, f32_tile, SEM_PART, SEM_GATHER, SEM_R, tp, ar_tile],
                    norm_rt,
                    dm0,
                ),
                kern(
                    "norm_shard_ar_compute.cpp",
                    norm,
                    [wt, self.np, self.inv_n_bits, self.eps_bits, tp],
                    [(xy, [int(i == 0)]) for i, xy in enumerate(self.norm_xy_logical)],
                    self.ncfg,
                ),
                kern(
                    "ar_send.cpp",
                    workers,
                    [d, 18, 17, shard_bytes, n_res] + route(fwd_n) + route(bwd_n),
                    [(c, worker_rt(l)) for l, c in enumerate(self.AR_WORKERS)],
                    dm0,
                ),
            ]
            sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=everyone, initial_value=0) for i in range(5)]
            pd = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)
            for l, c in enumerate(self.AR_WORKERS):
                args = pd.kernels[len(kernels) - 1].runtime_args[c[0]][c[1]]
                for nbr, n in (((d + 1) % tp, fwd_n), ((d - 1) % tp, bwd_n)):
                    args.append(int(n > 0))
                    if n > 0:
                        args.extend(ttnn.setup_fabric_connection(node(d), node(nbr), l, pd, ttnn.CoreCoord(*c)))
            coord = ttnn.MeshCoordinate(0, d)
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = pd
        ttnn.generic_op([x, p, ar.buffer, self.wg, self.wu, h, out], mpd)
        return h, out
