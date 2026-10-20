# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode rope on the QKV layout as a tt-lang (ttl) kernel (kernel-rung experiment): the first W / 32 tiles of the
one-tile-row QKV output, one tile per core of a 9 x 8 grid; tile c (head column tile dt = c % Dt) becomes
x[c] * cos[dt] + x[rotate-half partner] * signed_sin[dt]."""
from __future__ import annotations

import ttl

import ttnn

_TILE = 32
_GRID = (9, 8)
_N_CORES = _GRID[0] * _GRID[1]

_kernel = None


def _build(d_t):
    gx = _GRID[0]
    half = d_t // 2

    @ttl.operation(grid=_GRID)
    def rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, y: ttnn.Tensor) -> None:
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        xr_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=1)
        c_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), block_count=1)
        s_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), block_count=1)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            cx, cy = ttl.node(dims=2)
            c = cy * gx + cx
            dt = c % d_t
            rot = c - dt + (dt + half) % d_t
            with x_dfb.reserve() as xb, xr_dfb.reserve() as xrb, c_dfb.reserve() as cb, s_dfb.reserve() as sb:
                ttl.copy(x[0, c], xb).wait()
                ttl.copy(x[0, rot], xrb).wait()
                ttl.copy(cos[0, dt], cb).wait()
                ttl.copy(sin[0, dt], sb).wait()

        @ttl.compute()
        def compute():
            with x_dfb.wait() as xb, xr_dfb.wait() as xrb, c_dfb.wait() as cb, s_dfb.wait() as sb:
                with y_dfb.reserve() as yb:
                    yb.store(xb * cb + xrb * sb)

        @ttl.datamovement()
        def write():
            cx, cy = ttl.node(dims=2)
            c = cy * gx + cx
            with y_dfb.wait() as yb:
                ttl.copy(yb, y[0, c]).wait()

    return rope


def ttl_rope_supported(qkv, width):
    return width // _TILE == _N_CORES and qkv.dtype == ttnn.bfloat16 and list(qkv.padded_shape)[-2] == _TILE


def ttl_rope_qkv(device, qkv, cos_rows, sin_rows, width, d_t, memory_config):
    """qkv [1, 32, N] -> roped first `width` columns [1, 32, width]."""
    global _kernel
    if _kernel is None:
        _kernel = _build(d_t)
    b, n = int(qkv.shape[-2]), int(qkv.shape[-1])
    d = int(cos_rows.shape[-1])
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([b, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)
    _kernel(ttnn.reshape(qkv, (b, n)), ttnn.reshape(cos_rows, (b, d)), ttnn.reshape(sin_rows, (b, d)), y)
    return ttnn.reshape(y, (1, b, width))
