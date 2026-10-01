# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The tt-lang rung on the prefill KV-cache seed: the shared prefix k/v written into every sample's
cache rows by ONE kernel.

The stock seed repeats the batch-1 prefix `[1, H, P, D]` 32x into a `[B, H, P, D]` tensor (a
RepeatCodegen op that writes 5.6 MB) and then `paged_fill_cache` copies that into the resident
`[B, H, C, D]` cache (reading the 5.6 MB back). Here every core of an `H x P/32` grid owns one tile
row of the prefix: it reads it ONCE and writes it to the same tile row of every sample's cache
block, through a zero-copy 2D view of the cache. Nothing is repeated and no intermediate exists.
The values are copied tile for tile (bf8_b in, bf8_b out), so the cache holds the same bits.

On unless VOXTRAL_TTL_KV=0.
"""
from __future__ import annotations

import os

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import ttl_swiglu

try:
    import ttl

    _HAVE_TTL = ttl_swiglu._HAVE_TTL
except ImportError:  # pragma: no cover - depends on the environment
    _HAVE_TTL = False

TILE = 32
_OPS: dict = {}


def enabled() -> bool:
    return _HAVE_TTL and os.environ.get("VOXTRAL_TTL_KV", "1") == "1"


def _make(batch, heads, pt, ct, wt):
    """The fill op for a `[batch, heads, ct*32, wt*32]` cache and a `[1, heads, pt*32, wt*32]` prefix."""

    @ttl.operation(grid=(heads, pt))
    def prefix_fill(src: ttnn.Tensor, cache: ttnn.Tensor) -> None:
        in_dfb = ttl.make_dataflow_buffer_like(src, shape=(1, wt), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(cache, shape=(1, wt), block_count=2)

        @ttl.datamovement()
        def read():
            h, j = ttl.node(dims=2)
            with in_dfb.reserve() as blk:
                ttl.copy(src[h * pt + j : h * pt + j + 1, 0:wt], blk).wait()

        @ttl.compute()
        def compute():
            with in_dfb.wait() as a, out_dfb.reserve() as o:
                o.store(a)

        @ttl.datamovement()
        def write():
            h, j = ttl.node(dims=2)
            with out_dfb.wait() as blk:
                for b in range(batch):
                    r = (b * heads + h) * ct + j
                    ttl.copy(blk, cache[r : r + 1, 0:wt]).wait()

    return prefix_fill


def supports(cache, prefix) -> bool:
    if not enabled():
        return False
    try:
        b, h, c, d = (int(s) for s in cache.shape)
        pb, ph, p, pd = (int(s) for s in prefix.shape)
        return (
            pb == 1
            and ph == h
            and pd == d
            and cache.dtype == prefix.dtype
            and c % TILE == 0
            and p % TILE == 0
            and d % TILE == 0
            and p <= c
            and h * (p // TILE) <= 110
            and not cache.memory_config().is_sharded()
            and not prefix.memory_config().is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def fill_prefix(cache, prefix):
    """Write `prefix [1, H, P, D]` into slots `[0, P)` of every sample of `cache [B, H, C, D]`, in place."""
    b, h, c, d = (int(s) for s in cache.shape)
    p = int(prefix.shape[2])
    key = (b, h, p // TILE, c // TILE, d // TILE)
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = _make(*key)
    op(ttnn.reshape(prefix, [h * p, d]), ttnn.reshape(cache, [b * h * c, d]))
