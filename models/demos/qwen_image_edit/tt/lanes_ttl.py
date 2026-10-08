# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The 8 exact-lane copies of a precise linear's lead limb, and its negation, in ONE tt-lang pass.

The guarded precise linears (models/demos/qwen_image_edit_text_encoder/_stubs/attention.py, LANE_SPLIT) mask
their bf16 lead limb into 8 lanes, lane r keeping the K entries with k % 8 == r: 8 ttnn multiplies, each
reading the whole limb. Since 32 % 8 == 0, lane r's mask is the same 32x32 tile at every K tile (ones in the
columns j with j % 8 == r), so this kernel reads each limb tile once and writes its 8 masked copies. The
masks multiply on the SFPU (x * 1 = x, x * 0 = 0 exactly; a negative x masks to -0, which adds as +0), so
every lane equals the ttnn one. The negated limb (the guard's negated-input product) is the SFPU negation of a
bf16 value, exact as well.

`prepare(device)` uploads the 8 mask tiles once, at build time (the forward makes no host uploads);
`lanes8(lead)` returns (the 8 lanes, -lead), or None for a shape it does not take.
"""

from __future__ import annotations

import torch
import ttl

import ttnn

TILE = 32
LANES = 8
_OP = []
_MASKS = {}
_COMPILED = set()


def _make_op():
    @ttl.operation(grid="auto", options="--no-ttl-fpu-binary-ops")
    def _lanes(
        x: ttnn.Tensor,
        masks: ttnn.Tensor,
        y0: ttnn.Tensor,
        y1: ttnn.Tensor,
        y2: ttnn.Tensor,
        y3: ttnn.Tensor,
        y4: ttnn.Tensor,
        y5: ttnn.Tensor,
        y6: ttnn.Tensor,
        y7: ttnn.Tensor,
        yn: ttnn.Tensor,
    ):
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        rows = x.shape[0] // TILE
        cols = x.shape[1] // TILE
        rows_per_node = -(-rows // grid_rows)
        cols_per_node = -(-cols // grid_cols)

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
        m_dfb = ttl.make_dataflow_buffer_like(masks, shape=(1, 1), block_count=8)
        o0 = ttl.make_dataflow_buffer_like(y0, shape=(1, 1), block_count=2)
        o1 = ttl.make_dataflow_buffer_like(y1, shape=(1, 1), block_count=2)
        o2 = ttl.make_dataflow_buffer_like(y2, shape=(1, 1), block_count=2)
        o3 = ttl.make_dataflow_buffer_like(y3, shape=(1, 1), block_count=2)
        o4 = ttl.make_dataflow_buffer_like(y4, shape=(1, 1), block_count=2)
        o5 = ttl.make_dataflow_buffer_like(y5, shape=(1, 1), block_count=2)
        o6 = ttl.make_dataflow_buffer_like(y6, shape=(1, 1), block_count=2)
        o7 = ttl.make_dataflow_buffer_like(y7, shape=(1, 1), block_count=2)
        on = ttl.make_dataflow_buffer_like(yn, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            # the 8 mask tiles, held for the whole run
            with (
                m_dfb.wait() as m0,
                m_dfb.wait() as m1,
                m_dfb.wait() as m2,
                m_dfb.wait() as m3,
                m_dfb.wait() as m4,
                m_dfb.wait() as m5,
                m_dfb.wait() as m6,
                m_dfb.wait() as m7,
            ):
                for local_row in range(rows_per_node):
                    row = node_row * rows_per_node + local_row
                    if row < rows:
                        for local_col in range(cols_per_node):
                            col = node_col * cols_per_node + local_col
                            if col < cols:
                                with x_dfb.wait() as a:
                                    with o0.reserve() as o:
                                        o.store(a * m0)
                                    with o1.reserve() as o:
                                        o.store(a * m1)
                                    with o2.reserve() as o:
                                        o.store(a * m2)
                                    with o3.reserve() as o:
                                        o.store(a * m3)
                                    with o4.reserve() as o:
                                        o.store(a * m4)
                                    with o5.reserve() as o:
                                        o.store(a * m5)
                                    with o6.reserve() as o:
                                        o.store(a * m6)
                                    with o7.reserve() as o:
                                        o.store(a * m7)
                                    with on.reserve() as o:
                                        o.store(ttl.math.neg(a))

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            for r in range(LANES):
                with m_dfb.reserve() as m:
                    tx = ttl.copy(masks[r : r + 1, 0:1], m)
                    tx.wait()
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
                            with o0.wait() as o:
                                tx = ttl.copy(o, y0[row : row + 1, col : col + 1])
                                tx.wait()
                            with o1.wait() as o:
                                tx = ttl.copy(o, y1[row : row + 1, col : col + 1])
                                tx.wait()
                            with o2.wait() as o:
                                tx = ttl.copy(o, y2[row : row + 1, col : col + 1])
                                tx.wait()
                            with o3.wait() as o:
                                tx = ttl.copy(o, y3[row : row + 1, col : col + 1])
                                tx.wait()
                            with o4.wait() as o:
                                tx = ttl.copy(o, y4[row : row + 1, col : col + 1])
                                tx.wait()
                            with o5.wait() as o:
                                tx = ttl.copy(o, y5[row : row + 1, col : col + 1])
                                tx.wait()
                            with o6.wait() as o:
                                tx = ttl.copy(o, y6[row : row + 1, col : col + 1])
                                tx.wait()
                            with o7.wait() as o:
                                tx = ttl.copy(o, y7[row : row + 1, col : col + 1])
                                tx.wait()
                            with on.wait() as o:
                                tx = ttl.copy(o, yn[row : row + 1, col : col + 1])
                                tx.wait()

    return _lanes


def prepare(device):
    """Upload the 8 lane-mask tiles ((8 x 32, 32) bf16; tile r keeps the columns j % 8 == r), once."""
    if id(device) not in _MASKS:
        j = torch.arange(TILE)
        m = torch.cat([(j % LANES == r).to(torch.float32).expand(TILE, TILE) for r in range(LANES)], 0)
        _MASKS[id(device)] = ttnn.from_torch(
            m,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None,
        )


def lanes8(lead):
    """(the 8 lanes, -lead) of a bf16 interleaved tile tensor [..., M, K] (M, K tile-aligned), or None."""
    s = list(lead.shape)
    if len(s) < 2 or s[-2] % TILE or s[-1] % TILE:
        return None
    if lead.dtype != ttnn.bfloat16 or lead.layout != ttnn.TILE_LAYOUT or lead.is_sharded():
        return None
    dev = lead.device()
    if id(dev) not in _MASKS:
        return None
    if not _OP:
        _OP.append(_make_op())
    rows = 1
    for d in s[:-1]:
        rows *= d
    x = ttnn.view(lead, (rows, s[-1]))
    key = tuple(s)
    first = key not in _COMPILED
    if first:
        # ttl keeps the compiling call's arguments for the process lifetime: give it a copy to pin and free it,
        # and hand back copies of the outputs
        x = ttnn.clone(x)
    ys = [ttnn.empty_like(x) for _ in range(LANES + 1)]  # the lanes, then -lead
    _OP[0](x, _MASKS[id(dev)], *ys)
    if first:
        _COMPILED.add(key)
        ttnn.deallocate(x)
        outs = []
        for y in ys:
            outs.append(ttnn.clone(y))
            ttnn.deallocate(y)
        ys = outs
    ys = [ttnn.view(y, s) for y in ys]
    return ys[:LANES], ys[LANES]
