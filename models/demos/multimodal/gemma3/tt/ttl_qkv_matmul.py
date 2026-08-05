# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-lang kernel for gemma3's DECODE QKV projection -- the tt-lang rung on
``MatmulDeviceOperation 32 x 3840 x 8192``.

The op is the fused wqkv projection on the decode path: a [32, 3840] bf16 activation against a
[3840, 8192] **bfloat4_b** DRAM-width-sharded weight, 288 launches, 14.79 ms, 345 GB/s of a ~372
GB/s achievable stream.

Written rank-2 on purpose. ttl 1.0.1's tile ``@`` rejects rank-4 blocks ("lhs must be rank 2, got
rank 4") while ``make_dataflow_buffer_like`` requires the block rank to match the tensor's, so a
rank-4 matmul has NO legal buffer rank -- the caller reshapes to [32, 3840] / [3840, 8192] first.
The accumulator is handled by PEELING the first k-step (``store(a @ b)`` then
``store(prev + a @ b)``): ttl has no ``ttl.block.fill``, and an ``if kt == 0`` inside the compute
loop would force the k-loop to unroll at trace time.

Partitioning: one core per output N-tile column group, each core streaming all 120 K-tiles of the
single M-tile row and its own K x per_core_N slice of the weight. That is the simple correct
partitioning -- with M = 1 tile there is no output-row axis to re-read A over, so unlike the wide-M
shapes this one does NOT pay the activation-traffic penalty that sank the hand C++ kernels.
"""
from __future__ import annotations

import ttnn

import ttl

TILE = 32


@ttl.operation(grid=(1, 1))
def qkv_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    """out[32, N] = a[32, K] @ b[K, N], one M-tile, accumulating over K in DST."""
    k_tiles = a.shape[1] // TILE
    n_tiles = out.shape[1] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        for nt in range(n_tiles):
            for kt in range(k_tiles):
                with a_dfb.reserve() as blk:
                    ttl.copy(a[0, kt], blk).wait()
                with b_dfb.reserve() as blk:
                    ttl.copy(b[kt, nt], blk).wait()

    @ttl.compute()
    def compute():
        for _nt in range(n_tiles):
            # PEELED first k-step: no accumulator zeroing primitive exists in ttl 1.0.1.
            with a_dfb.wait() as a_blk:
                with b_dfb.wait() as b_blk:
                    with o_dfb.reserve() as res:
                        res.store(a_blk @ b_blk)
            for _kt in range(k_tiles - 1):
                with a_dfb.wait() as a_blk:
                    with b_dfb.wait() as b_blk:
                        with o_dfb.reserve() as res:
                            res.store(res + a_blk @ b_blk)

    @ttl.datamovement()
    def write():
        for nt in range(n_tiles):
            with o_dfb.wait() as blk:
                ttl.copy(blk, out[0, nt]).wait()
