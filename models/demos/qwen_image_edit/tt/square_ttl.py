# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""x * x for a bf16 tile tensor, bf16 out, in ONE tt-lang pass (the exact linears' guard bound: |lead row|).

The guard's tolerance takes the row norms of the bf16 lead limb only as a bound (GUARD_SLACK of margin), so the
squares are kept in bf16: half the bytes of float32 squares, written once and read once by the row sum (which
accumulates in float32). Each tile is squared on the SFPU (bf16 in, rounded to bf16 out).

`square(t)` returns t * t, or None for a tensor it does not take (the caller then keeps the ttnn spelling).
"""

from __future__ import annotations

import ttl

import ttnn

TILE = 32
_OP = []
_COMPILED = set()


def _make_op():
    @ttl.operation(grid="auto", options="--no-ttl-fpu-binary-ops")
    def _square(x: ttnn.Tensor, y: ttnn.Tensor):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        rows = x.shape[0] // TILE
        cols = x.shape[1] // TILE
        rows_per_node = -(-rows // grid_rows)
        cols_per_node = -(-cols // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < rows:
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < cols:
                            with x_dfb.wait() as a, y_dfb.reserve() as o:
                                o.store(a * a)

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < rows:
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < cols:
                            with x_dfb.reserve() as a:
                                tx = ttl.copy(x[row : row + 1, col : col + 1], a)
                                tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < rows:
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < cols:
                            with y_dfb.wait() as o:
                                tx = ttl.copy(o, y[row : row + 1, col : col + 1])
                                tx.wait()

    return _square


def square(t):
    """t * t (bf16) for an interleaved bf16 tile tensor [..., M, K] (M, K tile-aligned), or None. t is left as it is."""
    s = list(t.shape)
    ps = list(t.padded_shape)
    if len(s) < 2 or ps[-2] % TILE or ps[-1] % TILE:
        return None
    if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded():
        return None
    if not _OP:
        _OP.append(_make_op())
    rows = 1
    for d in ps[:-1]:
        rows *= d
    if ps != s:  # a ragged tensor: not viewable as a flat padded block
        return None
    x = ttnn.view(t, (rows, s[-1]))
    key = tuple(s)
    first = key not in _COMPILED
    if first:
        x = ttnn.clone(x)  # ttl pins the compiling call's arguments: give it a copy
    y = ttnn.empty_like(x)
    _OP[0](x, y)
    if first:
        _COMPILED.add(key)
        ttnn.deallocate(x)
        out = ttnn.clone(y)
        ttnn.deallocate(y)
        y = out
    return ttnn.view(y, s)
