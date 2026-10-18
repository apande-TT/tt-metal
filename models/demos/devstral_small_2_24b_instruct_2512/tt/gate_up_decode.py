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
    RB = 8  # K rows a weight push
    CB_BYTES = 112 * 1024  # per weight CB (DEPTH = pushes that fit)

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
            and kt % cls.RB == 0
        )

    @classmethod
    def bank_major(cls, w: torch.Tensor, nb: int) -> torch.Tensor:
        return bank_major(w, nb, cls.NC, cls.RB)

    def __init__(self, device, w_gate, w_up, dtype, math_fidelity, put, tp=1):
        """w_gate, w_up: [K, tp * N] torch weights (device d multiplies columns d * N .. (d + 1) * N); put(t, dtype)
        uploads a torch tensor sharded over the mesh on its last dim."""
        self.device = device
        k, n = int(w_gate.shape[0]), int(w_gate.shape[1]) // tp
        self.kt, self.nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        self.nb = nb = device.dram_grid_size().x
        bm = lambda w: torch.cat([self.bank_major(w[:, d * n : (d + 1) * n], nb) for d in range(tp)], dim=-1)
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
