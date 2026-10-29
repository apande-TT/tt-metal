# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode QKV projection as ONE program (ttnn.generic_op): qkv = x @ [W_q | W_k | W_v] for the single tile row of
decode tokens, x the input norm's bf16 width shards, replacing the DRAM-sharded matmul (~21.7 us a layer -> ~19).

The stock op runs 24 workers (3 a DRAM bank), each 2 output columns over the whole K = 160 tiles: 320 tile products
a core, math-bound. Here each of the N = 48 columns gets its own core -- 160 products a core -- on the 6 x 8
rectangle at (0, 0) whose 1-tile width shards nlp_create_qkv_heads_decode reads in place (2.2 -> 1.8 us):
  * both data-movement RISCs stream the core's column from a BANK-MAJOR weight copy, pushes of RB K rows (one
    contiguous read each), two in flight a RISC: RISCV_1 (NOC 1) the first W1 odd pushes, RISCV_0 (NOC 0, whose reads
    land faster) the rest (kernels/qkv_stream.cpp);
  * a spare SENDER core reads x out of the norm's shards and multicasts it to the rectangle CH tiles a chunk,
    alternate chunks over the two NOCs from its two RISCs, each chunk counted into a valid semaphore as it goes out
    (kernels/qkv_xsend.cpp); RISCV_1 of each core hands compute the chunks in K order as they are counted, while it
    waits for its own weight reads (cpp try 4 had every core pull the whole row itself: 15 MB of unicast reads,
    ~50 us; with one weight RISC a core the stream ran at ~337 GB/s);
  * compute (kernels/qkv_compute.cpp) accumulates the column over the whole K in fp32 DEST at HiFi2, in K order, and
    packs it to bf16 straight into the core's output shard. The stock op instead packs its fp32 partial after every
    in0_block_w (20) K tiles and reloads it through srcA, which keeps 19 bits: SPILL_KB = 20 replays that exactly
    (text PCC 0.995063 as stock) but the image gate fell to 0.992770; the unbroken fp32 accumulation is the more exact
    sum (text 0.995063 -> 0.99513, image 0.995239 unchanged)."""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.devstral_small_2_24b_instruct_2512.tt.gate_up_decode import (
    _TILE_BYTES,
    bank_major_order,
    stream_order,
)

_KDIR = Path(__file__).resolve().parent / "kernels"


class StreamQKVDecode:
    # compute cores = output shard cores, one column each (row-major from (0, 0)); grid try (run 12): 6 wide x 8 tall
    # 19.96-20.19 us vs 8 x 6 20.6-21.3 (the NOC 1 weight stream ends ~1.5 us sooner); centred 8 x 6 at (3, 2): ~22 us
    GRID = (6, 8)
    RB = 10  # K rows a weight push (~10.9 KB block-float8)
    CH = 10  # x tiles a multicast chunk
    SPILL_KB = 0  # K tiles between fp32 partial spills / srcA reloads (0: none, see the module docstring)
    W1 = 7  # pushes of the 16 RISCV_1 streams (8 each: the NOC 1 stream ended ~4 us after the NOC 0 one)
    SENDER = (6, 0)  # next to the rectangle and to the norm's shards (8 x 5 at (0, 0))
    # cpp try (run 12, NLPCreateQKVHeadsDecode): __call__(heads=...) writes q / k / v in the head layouts itself
    HEADS_OUT = True

    @classmethod
    def supports(cls, device, k, n, dtype) -> bool:
        """k x n: one device's [q | k | v] weight."""
        kt, nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        g = device.compute_with_storage_grid_size()
        gx, gy = cls.GRID
        sx, sy = cls.SENDER
        return (
            dtype in _TILE_BYTES
            and k % ttnn.TILE_SIZE == 0
            and n % ttnn.TILE_SIZE == 0
            and nt == gx * gy
            and gx <= g.x
            and gy <= g.y
            and (sx >= gx or sy >= gy)
            and sx < g.x
            and sy < g.y
            and nt % device.dram_grid_size().x == 0
            and kt % (cls.SPILL_KB or kt) == 0
            and (cls.SPILL_KB or kt) % cls.RB == 0
            and cls.RB % cls.CH == 0
        )

    def __init__(self, device, wqkv, dtype, math_fidelity, put, tp):
        """wqkv: [K, tp * N] torch weight (device d multiplies columns d * N .. (d + 1) * N); put(t, dtype) uploads a
        torch tensor sharded over the mesh on its last dim."""
        self.device = device
        k, n = int(wqkv.shape[0]), int(wqkv.shape[1]) // tp
        self.kt, self.nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        self.nb = nb = device.dram_grid_size().x
        gx, gy = self.GRID
        # column j on core (j % gx, j // gx), streamed from bank j % nb
        order, self.page0s = stream_order(self.kt, self.nt, nb, [(j, 1, j % nb) for j in range(self.nt)], self.RB)
        self.w = put(
            torch.cat([bank_major_order(wqkv[:, d * n : (d + 1) * n], order) for d in range(tp)], dim=-1), dtype
        )
        del order
        self.core_xy = [(j % gx, j // gx) for j in range(self.nt)]
        rect = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))
        self.cores = ttnn.CoreRangeSet([rect])
        one = lambda xy: ttnn.CoreRange(ttnn.CoreCoord(*xy), ttnn.CoreCoord(*xy))
        self.sender_set = ttnn.CoreRangeSet([one(self.SENDER)])
        self.all = self.cores.merge(self.sender_set)
        phys = lambda xy: device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        snd, lo, hi = phys(self.SENDER), phys((0, 0)), phys((gx - 1, gy - 1))
        self.sender_xy = [snd.x, snd.y]
        self.rect_xy = [lo.x, lo.y, hi.x, hi.y]
        self.out_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.cores, (ttnn.TILE_SIZE, ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.w_tile = _TILE_BYTES[dtype]
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = math_fidelity
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        self.cfg = cfg

    def accepts(self, x) -> bool:
        mem = x.memory_config()
        return (
            tuple(int(d) for d in x.shape) == (1, 1, ttnn.TILE_SIZE, self.kt * ttnn.TILE_SIZE)
            and x.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and mem.buffer_type == ttnn.BufferType.L1
            and mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )

    def _kernel(self, name, ct, rt, config, defines=(), cores=None):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores or self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
            defines=list(defines),
        )

    def heads_ok(self, heads_mem, users, n_q, n_kv, head_dim) -> bool:
        """Whether __call__ can split the heads itself into nlp_create_qkv_heads_decode(overlap_qk_coregrid=False,
        memory_config=heads_mem)'s outputs: heads_mem one rectangle at (0, 0) whose first `users` cores (whole rows)
        take q and v and the next `users` k, and N = (n_q + 2 n_kv) head_dim."""
        spec = heads_mem.shard_spec
        if not self.HEADS_OUT or spec is None or len(spec.grid.ranges()) != 1:
            return False
        r = spec.grid.ranges()[0]
        w = r.end.x - r.start.x + 1
        return (
            (r.start.x, r.start.y) == (0, 0)
            and 0 < users <= ttnn.TILE_SIZE
            and users % w == 0
            and spec.grid.num_cores() >= 2 * users
            and n_q <= ttnn.TILE_SIZE
            and n_kv <= ttnn.TILE_SIZE
            and head_dim % ttnn.TILE_SIZE == 0
            and (n_q + 2 * n_kv) * head_dim == self.nt * ttnn.TILE_SIZE
        )

    def _head_outputs(self, heads_mem, users, n_q, n_kv, head_dim):
        """q (1, B, n_q, hd), k / v (1, B, n_kv, hd): bf16 tiles, one (32, hd) shard a user -- q and v on the head
        grid's first B cores, k on the next B (nlp_create_qkv_heads_decode's output specs) -- and each one's cores."""
        r = heads_mem.shard_spec.grid.ranges()[0]
        w = r.end.x - r.start.x + 1
        rows = users // w
        rect = lambda y0: ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, y0), ttnn.CoreCoord(w - 1, y0 + rows - 1))]
        )
        mem = lambda grid: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (ttnn.TILE_SIZE, head_dim), ttnn.ShardOrientation.ROW_MAJOR),
        )
        alloc = lambda heads, grid: ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, users, heads, head_dim)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, mem(grid)
        )
        q, k, v = alloc(n_q, rect(0)), alloc(n_kv, rect(rows)), alloc(n_kv, rect(0))
        phys = lambda u, y0: self.device.worker_core_from_logical_core(ttnn.CoreCoord(u % w, y0 + u // w))
        qv_xy = [c for u in range(users) for c in (phys(u, 0).x, phys(u, 0).y)]
        k_xy = [c for u in range(users) for c in (phys(u, rows).x, phys(u, rows).y)]
        return q, k, v, qv_xy, k_xy

    def __call__(self, x, heads=None):
        """x: (1, 1, 32, K) bf16 width-sharded in L1 (the input norm's output); returns x @ W, (1, 1, 32, N) bf16 in
        1-tile width shards over GRID. heads = (heads_mem, users, n_q, n_kv, head_dim) (heads_ok): instead returns
        (q, k, v) as nlp_create_qkv_heads_decode(.., overlap_qk_coregrid=False, memory_config=heads_mem) splits them
        -- each core writes its column's rows straight into the head shards (kernels/qkv_stream.cpp HEADS_OUT)."""
        if not self.accepts(x):
            raise ValueError(f"StreamQKVDecode cannot take {tuple(x.shape)} {x.dtype} {x.memory_config()}")
        kt, nt, rb, ch, w_tile, nb = self.kt, self.nt, self.RB, self.CH, self.w_tile, self.nb
        x_tile, o_tile = 2048, 2048
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        fmt = lambda index, page, dtype: [
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
        ]
        cb = lambda index, tiles, page, dtype, cores=None: ttnn.CBDescriptor(
            total_size=tiles * page, core_ranges=cores or self.cores, format_descriptors=fmt(index, page, dtype)
        )
        CB_X, CB_W0, CB_W1, CB_OUT, CB_PART = 0, 1, 2, 16, 24
        if heads is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.device,
                self.out_mem,
            )
            outs, out_cb, heads_rt = [out], ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out), None
        else:
            heads_mem, users, n_q, n_kv, hd = heads
            q, k, v, qv_xy, k_xy = self._head_outputs(heads_mem, users, n_q, n_kv, hd)
            outs, out_cb = [q, k, v], cb(CB_OUT, 1, o_tile, ttnn.bfloat16)
            dt = hd // ttnn.TILE_SIZE  # tiles across a head
            face_row = lambda h: h * 32 if h < 16 else 1024 + (h - 16) * 32  # row h's first face row in a bf16 tile

            def heads_rt(j):  # column j: [q | k | v], head-major, dt columns a head
                part, h = (
                    (0, j // dt)
                    if j < n_q * dt
                    else ((1, (j - n_q * dt) // dt) if j < (n_q + n_kv) * dt else (2, (j - (n_q + n_kv) * dt) // dt))
                )
                t = j % dt
                buf = (q, k, v)[part].buffer_address()
                return [buf + t * o_tile + face_row(h), users] + (k_xy if part == 1 else qv_xy)

        SEM_READY, SEM_V0, SEM_V1 = 0, 1, 2
        w1 = self.W1
        w0 = kt // rb - w1  # each weight CB holds its RISC's whole share
        cbs = [  # x first: the sender multicasts it at its own address of it, the same L1 address on every core
            cb(CB_X, kt, x_tile, ttnn.bfloat16, self.all),
            cb(CB_W0, w0 * rb, w_tile, self.w.dtype),
            cb(CB_W1, w1 * rb, w_tile, self.w.dtype),
            out_cb,
            cb(CB_PART, 2, 4096, ttnn.float32),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all, initial_value=0) for i in (0, 1, 2)]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        # kernels/qkv_stream.cpp: RISCV_1 streams the first W1 odd pushes of the column and receives x, RISCV_0 the rest
        # (and with heads, scatters the finished tile into the head shards)
        stream_ct = lambda cb, role: [kt, nb, rb, w_tile, cb, ch, SEM_READY, SEM_V0, SEM_V1, w1, role] + acc(self.w)
        stream_rt = lambda role: [
            (
                xy,
                [self.w.buffer_address(), self.page0s[c]]
                + self.sender_xy
                + (heads_rt(c) if heads_rt is not None and role == 0 else []),
            )
            for c, xy in enumerate(self.core_xy)
        ]
        n_cores = len(self.core_xy)
        kernels = [
            self._kernel(
                "qkv_stream.cpp", stream_ct(CB_W0, 0), stream_rt(0), dm0, [] if heads is None else [("HEADS_OUT", "1")]
            ),
            self._kernel("qkv_stream.cpp", stream_ct(CB_W1, 1), stream_rt(1), dm1, [("X_RECV", "1")]),
            self._kernel(
                "qkv_compute.cpp", [kt, self.SPILL_KB or kt, rb, ch, w1], [(xy, []) for xy in self.core_xy], self.cfg
            ),
        ] + [
            self._kernel(
                "qkv_xsend.cpp",
                [kt, ch, x_tile, n_cores, SEM_READY, sem] + acc(x),
                [(self.SENDER, [x.buffer_address(), r] + self.rect_xy)],
                cfg,
                cores=self.sender_set,
            )
            for r, (sem, cfg) in enumerate(((SEM_V0, dm0), (SEM_V1, dm1)))
        ]
        ttnn.generic_op([x, self.w] + outs, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return outs[0] if heads is None else tuple(outs)
