# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""float32 -> its bf16 rounding, as float32, in ONE tt-lang pass (the VAE exact convs' hi limb).

precise_affine (models/tt_dit/pipelines/qwen_image_edit_vae/_stubs/_resident.py, ROUND_BF16) spells the hi
limb as typecast(typecast(t, bf16), float32): two passes, an input-sized bf16 round trip through DRAM. Here
each tile is typecast into a bf16 dataflow buffer and typecast back out as float32 (widening is exact), so
the tensor is read once and written once.

`prepare(device)` runs once at build time; `round_bf16` returns None for a shape it does not take (or an
unprepared device), and the caller then keeps the ttnn spelling.
"""

from __future__ import annotations

import torch
import ttl

import ttnn

TILE = 32
_OP = []
_PROTO = {}
_COMPILED = set()


def _make_op():
    @ttl.operation(grid="auto", fp32_dest_acc_en=True)
    def _round(x: ttnn.Tensor, proto: ttnn.Tensor, y: ttnn.Tensor):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        rows = x.shape[0] // TILE
        cols = x.shape[1] // TILE
        rows_per_node = -(-rows // grid_rows)
        cols_per_node = -(-cols // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        mid_dfb = ttl.make_dataflow_buffer_like(proto, shape=(1, 1), block_count=2)  # bf16
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
                            with x_dfb.wait() as a, mid_dfb.reserve() as m:
                                m.store(ttl.math.typecast(a, ttnn.bfloat16))  # the rounding
                            with mid_dfb.wait() as m, y_dfb.reserve() as o:
                                o.store(ttl.math.typecast(m, ttnn.float32))  # widened back: exact

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

    return _round


def prepare(device):
    """Upload the one-tile bf16 tensor the kernel takes its bf16 buffer's format from (once, at build time:
    the forward makes no host uploads)."""
    if id(device) not in _PROTO:
        _PROTO[id(device)] = ttnn.from_torch(
            torch.zeros(TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None,
        )


def round_bf16(t):
    """t rounded to bf16, as float32, for an interleaved float32 tile tensor [..., M, C] (M, C tile-aligned);
    or None. t is left as it is (the caller still needs it)."""
    s = list(t.shape)
    if len(s) < 2 or s[-2] % TILE or s[-1] % TILE:
        return None
    if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded():
        return None
    if not _OP:
        _OP.append(_make_op())
    dev = t.device()
    if id(dev) not in _PROTO:  # prepare() was not run for this device: no upload inside a forward
        return None
    rows = 1
    for d in s[:-1]:
        rows *= d
    x = ttnn.view(t, (rows, s[-1]))
    key = tuple(s)
    first = key not in _COMPILED
    if first:
        # ttl keeps the compiling call's arguments for the process lifetime: give it a copy of t to pin, then
        # free that copy; hand back a copy of the output and free the original
        x = ttnn.clone(x)
    y = ttnn.empty_like(x)
    _OP[0](x, _PROTO[id(dev)], y)
    if first:
        _COMPILED.add(key)
        ttnn.deallocate(x)
        out = ttnn.clone(y)
        ttnn.deallocate(y)
        y = out
    return ttnn.view(y, s)
