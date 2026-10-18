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
from models.demos.devstral_small_2_24b_instruct_2512.tt.gate_up_decode import _TILE_BYTES, bank_major, bank_major_page0

_KDIR = Path(__file__).resolve().parent / "kernels"


class StreamDownDecode:
    NC = 2  # output columns a compute core (one a data-movement RISC)
    RB = 16  # K rows a weight push (bf16)
    # K rows a push by weight dtype: ~9 KB pushes (8 block-float8 / 16 block-float4 tiles) stream fastest (bf8 down,
    # measured: 35 KB pushes 120 us, 17 KB 117 us, 9 KB 110.5 us, 4 KB 111 us)
    RB_BY_DTYPE = {ttnn.bfloat8_b: 8, ttnn.bfloat4_b: 16}
    CB_BYTES = 112 * 1024  # per weight CB (DEPTH = pushes that fit)
    GRID_X = 8  # compute cores: an 8-wide rectangle (as the 1D down's 8 x 10)

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

    def __init__(self, device, w_down, dtype, math_fidelity, put, tp=1):
        """w_down: [tp * K, N] torch weight (device d multiplies rows d * K .. (d + 1) * K); put(t, dtype) uploads a
        torch tensor sharded over the mesh on its second-to-last dim."""
        import torch

        self.device = device
        self.RB = self.rb_for(dtype)
        k, n = int(w_down.shape[0]) // tp, int(w_down.shape[1])
        self.kt, self.nt = k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        self.nb = nb = device.dram_grid_size().x
        # one stream a column (nc = 1): stream s reads column s
        self.w = put(
            torch.cat([bank_major(w_down[d * k : (d + 1) * k], nb, 1, self.RB) for d in range(tp)], dim=0), dtype
        )
        n_cores = self.nt // self.NC
        g = device.compute_with_storage_grid_size()
        gx, gy = self.GRID_X, n_cores // self.GRID_X
        self.core_xy = [(c % gx, c // gx) for c in range(n_cores)]
        rng = lambda a, b, c, d: ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))
        self.cores = ttnn.CoreRangeSet([rng(0, 0, gx - 1, gy - 1)])
        self.sender = (g.x - 1, g.y - 1)
        self.sender_set = ttnn.CoreRangeSet([rng(*self.sender, *self.sender)])
        self.all = self.cores.merge(self.sender_set)
        snd = device.worker_core_from_logical_core(ttnn.CoreCoord(*self.sender))
        self.sender_xy = [snd.x, snd.y]
        lo = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        hi = device.worker_core_from_logical_core(ttnn.CoreCoord(gx - 1, gy - 1))
        self.rects = [1, lo.x, lo.y, hi.x, hi.y, n_cores]  # the sender's multicast target (NOC 0 order corners)
        self.w_tile = _TILE_BYTES[dtype]
        self.depth = max(2, self.CB_BYTES // (self.RB * self.w_tile))
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = math_fidelity
        cfg.fp32_dest_acc_en = True
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

    def __call__(self, m):
        """m: (1, 1, 32, K) bf16 tiles (interleaved); returns m @ W_down, (1, 1, 32, N) bf16 in L1."""
        shape = tuple(int(d) for d in m.shape)
        if shape[:-1] != (1, 1, ttnn.TILE_SIZE) or m.dtype != ttnn.bfloat16 or m.is_sharded():
            raise ValueError(f"StreamDownDecode takes one interleaved bf16 tile row, got {shape} {m.dtype}")
        kt, nt, rb, w_tile, depth, nb = self.kt, self.nt, self.RB, self.w_tile, self.depth, self.nb
        x_tile, o_tile = 2048, 2048
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, ttnn.TILE_SIZE, nt * ttnn.TILE_SIZE)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.L1_MEMORY_CONFIG,
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
        # one weight column a push (nc = 1); the OWNER (RISCV_1) writes the core's NC output tiles
        stream_ct = lambda cb, owner: [kt, nb, 1, rb, w_tile, cb, depth, owner, SEM_READY, SEM_VALID, o_tile, self.NC]
        rt = lambda j: [
            (
                (cx, cy),
                [self.w.buffer_address(), c * self.NC, out.buffer_address()]
                + self.sender_xy
                + [bank_major_page0(c * self.NC + j, kt, nb, 1, rb)],
            )
            for c, (cx, cy) in enumerate(self.core_xy)
        ]
        kernels = [
            self._kernel(
                "gate_up_sender.cpp",
                self.sender_set,
                [kt, x_tile, len(self.core_xy), SEM_READY, SEM_VALID] + acc(m),
                [(self.sender, [m.buffer_address()] + self.rects)],
                dm0,
            ),
            self._kernel("gate_up_stream.cpp", self.cores, stream_ct(1, 0) + acc(self.w, out), rt(0), dm0),
            self._kernel("gate_up_stream.cpp", self.cores, stream_ct(2, 1) + acc(self.w, out), rt(1), dm1),
            self._kernel("down_compute.cpp", self.cores, [kt, rb], [], self.cfg),
        ]
        ttnn.generic_op([m, self.w, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return out
