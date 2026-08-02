# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-lang head-split kernels for gemma3's prefill ``nlp_create_qkv_heads`` (the tt-lang rung).

The stock op is pure data movement: it slices a fused ``[1, 1, S, (q + 2kv) * head_dim]`` tensor
into ``q[1, nq, S, hd]`` and ``k/v[1, nkv, S, hd]``. There is no math, so the ttl matmul blocker in
this tree (``mm_block_init`` is not declared here) does not apply -- a tile gather lowers cleanly.

WHY THREE KERNELS: ``@ttl.operation`` hard-rejects ``num_outs != 1``, so the one stock op that
returns q, k and v has to become three single-output gathers. Each reads only the columns it owns,
so the total DRAM traffic is unchanged; the cost is two extra dispatches per call.

The kernels must live in a real module: ttl reads them back with ``inspect.getsource``, which a
heredoc or an exec'd string breaks.
"""
from __future__ import annotations

import ttnn

import ttl

TILE = 32


def _gather(x, out, col0):
    """Copy out[0, h, st, dt] <- x[0, 0, st, col0 + h * d_tiles + dt] for every head/tile."""
    heads = out.shape[1]
    s_tiles = out.shape[2] // TILE
    d_tiles = out.shape[3] // TILE
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1, 1, 1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        for h in range(heads):
            for st in range(s_tiles):
                for dt in range(d_tiles):
                    with x_dfb.reserve() as blk:
                        ttl.copy(x[0, 0, st, col0 + h * d_tiles + dt], blk).wait()

    @ttl.compute()
    def compute():
        for _h in range(heads):
            for _st in range(s_tiles):
                for _dt in range(d_tiles):
                    with x_dfb.wait() as blk:
                        with o_dfb.reserve() as res:
                            res.store(blk)

    @ttl.datamovement()
    def write():
        for h in range(heads):
            for st in range(s_tiles):
                for dt in range(d_tiles):
                    with o_dfb.wait() as blk:
                        ttl.copy(blk, out[0, h, st, dt]).wait()


@ttl.operation(grid=(1, 1))
def split_q(x: ttnn.Tensor, q: ttnn.Tensor) -> None:
    _gather(x, q, 0)


@ttl.operation(grid=(1, 1))
def split_k(x: ttnn.Tensor, k: ttnn.Tensor) -> None:
    _gather(x, k, x.shape[3] // TILE - 2 * (k.shape[1] * (k.shape[3] // TILE)))


@ttl.operation(grid=(1, 1))
def split_v(x: ttnn.Tensor, v: ttnn.Tensor) -> None:
    _gather(x, v, x.shape[3] // TILE - (v.shape[1] * (v.shape[3] // TILE)))


def create_qkv_heads(x, num_heads, num_kv_heads, memory_config):
    """Drop-in for ttnn.experimental.nlp_create_qkv_heads on the prefill (transpose_k_heads=False)."""
    width = int(x.shape[-1])
    head_dim = width // (num_heads + 2 * num_kv_heads)
    seq = int(x.shape[-2])
    batch = int(x.shape[0])
    q = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, num_heads, seq, head_dim]), x.dtype, x.layout, x.device(), memory_config
    )
    k = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, num_kv_heads, seq, head_dim]), x.dtype, x.layout, x.device(), memory_config
    )
    v = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, num_kv_heads, seq, head_dim]), x.dtype, x.layout, x.device(), memory_config
    )
    split_q(x, q)
    split_k(x, k)
    split_v(x, v)
    return q, k, v
