# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The one-tile-row lm_head (x @ W for 32 rows: decode, and prefill's last prompt rows) as ONE streaming program
(ttnn.generic_op, kernels/gate_up_sender.cpp + lm_head_stream.cpp + lm_head_compute.cpp): the decode gate / up's
weight-streaming design (tt/gate_up_decode.py) without its SwiGLU, over several column passes per core.

ttnn.linear's 1D-multicast matmul streamed the (5120 x vocab/TP) block-float8 weight at ~380 GB/s. Here a spare core
multicasts x once, and each of the N_CORES compute cores streams its columns on both data-movement RISCs (copy A on
RISCV_0 / NOC 0, copy B on RISCV_1 / NOC 1), PASSES groups of 2 * NC columns: in pass p RISC A reads stream
p * N_CORES + c of its BANK-MAJOR copy (tt/gate_up_decode.bank_major), so every pass spreads the cores evenly over the
DRAM banks and each push is one contiguous read. Every output tile accumulates the whole K in K order in fp32 DEST and
is rounded to bf16 once: the 1D-multicast matmul's result (its fp32 partials between K blocks are reloaded exactly)."""
from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.devstral_small_2_24b_instruct_2512.tt.gate_up_decode import _TILE_BYTES, bank_major, bank_major_page0

_KDIR = Path(__file__).resolve().parent / "kernels"


class LMHeadDecode:
    NC = 4  # columns a stream a pass: 2 * NC tiles fill the full-sync fp32 DEST
    N_CORES = 64  # compute cores (a multiple of the DRAM banks)
    RB_BY_DTYPE = {ttnn.bfloat8_b: 1, ttnn.bfloat4_b: 2}  # K rows a push (~4.4 KB pushes, as the decode gate / up)
    CB_BYTES = 256 * 1024  # per weight CB

    @classmethod
    def supports(cls, device, k, n, dtype) -> bool:
        """k x n: one device's weight."""
        kt, nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        g = device.compute_with_storage_grid_size()
        per_core = 2 * cls.NC * cls.N_CORES
        return (
            dtype in cls.RB_BY_DTYPE
            and k % ttnn.TILE_SIZE == 0
            and n % ttnn.TILE_SIZE == 0
            and nt % per_core == 0
            and cls.N_CORES < g.x * g.y  # and a spare core for the sender
            and cls.N_CORES % device.dram_grid_size().x == 0
            and kt % cls.RB_BY_DTYPE[dtype] == 0
        )

    def __init__(self, device, w, dtype, math_fidelity, put, tp=1):
        """w: [K, tp * N] torch weight (device d multiplies columns d * N .. (d + 1) * N); put(t, dtype) uploads a torch
        tensor sharded over the mesh on its last dim."""
        self.device = device
        nc, n_cores = self.NC, self.N_CORES
        self.rb = rb = self.RB_BY_DTYPE[dtype]
        k, n = int(w.shape[0]), int(w.shape[1]) // tp
        self.kt, self.nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        self.passes = self.nt // (2 * nc * n_cores)
        self.nb = nb = device.dram_grid_size().x

        def copies(wd):  # one device's [K, N] -> copy A and copy B, columns ordered by stream p * n_cores + c
            tiles = wd.reshape(k, self.nt, ttnn.TILE_SIZE)
            idx_a, idx_b = [], []
            for p in range(self.passes):
                for c in range(n_cores):
                    first = (c * self.passes + p) * 2 * nc  # core c's pass-p output columns
                    idx_a += range(first, first + nc)
                    idx_b += range(first + nc, first + 2 * nc)
            pick = lambda idx: bank_major(tiles[:, idx, :].reshape(k, -1), nb, nc, rb)
            return pick(idx_a), pick(idx_b)

        per_dev = [copies(w[:, d * n : (d + 1) * n]) for d in range(tp)]
        self.wa = put(torch.cat([a for a, _ in per_dev], dim=-1), dtype)
        self.wb = put(torch.cat([b for _, b in per_dev], dim=-1), dtype)

        g = device.compute_with_storage_grid_size()
        self.core_xy = [(c % g.x, c // g.x) for c in range(n_cores)]
        full, rem = divmod(n_cores, g.x)
        rects = [(0, 0, g.x - 1, full - 1)] if full else []
        if rem:
            rects.append((0, full, rem - 1, full))
        rng = lambda a, b, c, d: ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
        self.cores = ttnn.CoreRangeSet([rng(*r) for r in rects])
        self.sender = sender = (g.x - 1, g.y - 1)
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
        self.depth = max(2, self.CB_BYTES // (rb * nc * self.w_tile))
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = math_fidelity
        cfg.fp32_dest_acc_en = True
        cfg.dst_full_sync_en = True
        cfg.math_approx_mode = False
        self.cfg = cfg

    def accepts(self, x) -> bool:
        shape = tuple(int(d) for d in x.shape)
        return (
            shape == (1, 1, ttnn.TILE_SIZE, self.kt * ttnn.TILE_SIZE)
            and x.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and not x.is_sharded()
        )

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

    def __call__(self, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """x: (1, 1, 32, K) bf16 tiles (interleaved); returns x @ W, (1, 1, 32, N) bf16."""
        kt, nt, nc, rb, w_tile, depth, nb = self.kt, self.nt, self.NC, self.rb, self.w_tile, self.depth, self.nb
        n_cores, passes = self.N_CORES, self.passes
        x_tile, o_tile = 2048, 2048
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        wdt = self.wa.dtype
        # x first: it is written remotely at the sender's own address of it, so it sits at the same L1 address on
        # every core
        cbs = [
            self._cb(0, kt, x_tile, ttnn.bfloat16),
            self._cb(1, depth * rb * nc, w_tile, wdt),
            self._cb(2, depth * rb * nc, w_tile, wdt),
            self._cb(16, 2 * nc, o_tile, ttnn.bfloat16),
        ]
        SEM_READY, SEM_VALID = 0, 1
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all, initial_value=0) for i in (SEM_READY, SEM_VALID)]
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
        dm0 = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0)
        dm1 = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1)
        pass_pages = n_cores * kt * nc  # stream p * n_cores + c starts this many pages after stream c (n_cores % nb == 0)
        stream_ct = lambda cb, owner, w: [
            kt, nb, nc, rb, w_tile, cb, depth, owner, SEM_READY, SEM_VALID, o_tile, passes, pass_pages
        ] + acc(w, out)
        page0 = lambda c: bank_major_page0(c, kt, nb, nc, rb)
        rt = lambda w: [
            ((cx, cy), [w.buffer_address(), c * passes * 2 * nc, out.buffer_address()] + self.sender_xy + [page0(c)])
            for c, (cx, cy) in enumerate(self.core_xy)
        ]
        kernels = [
            self._kernel(
                "gate_up_sender.cpp",
                self.sender_set,
                [kt, x_tile, n_cores, SEM_READY, SEM_VALID] + acc(x),
                [(self.sender, [x.buffer_address()] + self.rects)],
                dm0,
            ),
            self._kernel("lm_head_stream.cpp", self.cores, stream_ct(1, 0, self.wa), rt(self.wa), dm0),
            self._kernel("lm_head_stream.cpp", self.cores, stream_ct(2, 1, self.wb), rt(self.wb), dm1),
            self._kernel("lm_head_compute.cpp", self.cores, [kt, nc, rb, passes], [], self.cfg),
        ]
        ttnn.generic_op([x, self.wa, self.wb, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return out
