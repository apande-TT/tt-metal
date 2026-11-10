# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-mesh collectives for the TP=N transformer ports.

The ports shard with ShardTensorToMesh over ALL devices of the mesh (row-major device order), so a
TP collective must span every device. On a 1xN mesh that is one collective on axis 1; on an RxC mesh
it is axis 1 then axis 0. all_gather in that order concatenates the shards back in row-major order,
and a reduce over both axes sums all R*C partials.

Reduction precision: ttnn.all_reduce / ttnn.reduce_scatter round float32 partials. Measured on this
T3K 2x4 mesh: 7e-3..1.1e-2 max abs error on O(3) float32 sums of [2,64,256]..[4,128,3072] per chip.
That is bf16-level rounding, on EVERY residual update of every block. The ports keep their residual
streams in float32 for a reason, so the reduce here gathers the partials (all_gather moves bits
exactly) and adds them in float32. Measured error of that path: 0.0 on the same tensors. The gather
is chunked along the token dim so the N-fold temporary stays bounded.
"""

from __future__ import annotations

import ttnn

_CHUNK_BYTES = 48 * 1024 * 1024  # per-partial bytes gathered at once
# Optional fused kernel for the gathered partials' sum: SUM_PARTS(g, n) -> the left-to-right float32 sum of the n
# slabs of g = [n, ...] (bit-identical to the slice + add chain below), or None for a g it does not take.
SUM_PARTS = None
# Optional exact reduce-scatter + all-gather for an axis of >= 4 devices: each device receives every partial's 1/n
# column slice (all_to_all), sums its slice in the same device order (SUM_PARTS), and the summed slices are gathered
# back -- 2 (n - 1) / n of the tensor over the links instead of the gather's (n - 1), and a 1/n-sized sum.
A2A_REDUCE = False


def mesh_axes(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return []
    if len(shape) != 2:
        return [1] if device.get_num_devices() > 1 else []
    return [ax for ax in (1, 0) if shape[ax] > 1]


def _gather_sum_once(y, ax, n):
    shape = list(y.shape)
    g = ttnn.all_gather(
        ttnn.reshape(y, [1] + shape), dim=0, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear
    )
    out = None if SUM_PARTS is None else SUM_PARTS(g, n)
    if out is not None:
        ttnn.deallocate(g)
        return ttnn.reshape(out, shape)
    for i in range(n):
        part = ttnn.slice(g, [i] + [0] * len(shape), [i + 1] + shape)
        out = part if out is None else ttnn.add(out, part)
    ttnn.deallocate(g)
    return ttnn.reshape(out, shape)


def _a2a_sum(y, ax, n):
    """The reduce as all_to_all (each device: every partial's slice of its 1/n columns) + sum + all_gather of the
    sums, or None for a y it does not take."""
    shape = list(y.shape)
    if SUM_PARTS is None or n < 4 or len(shape) < 2 or shape[-1] % (32 * n) or y.layout != ttnn.TILE_LAYOUT:
        return None
    g = ttnn.experimental.all_to_all_async_generic(
        ttnn.reshape(y, [1] + shape),
        in_dim=0,
        out_dim=len(shape),
        num_links=1,
        topology=ttnn.Topology.Linear,
        cluster_axis=ax,
    )
    part = SUM_PARTS(g, n)
    ttnn.deallocate(g)
    if part is None:
        return None
    out = ttnn.all_gather(part, dim=len(shape) - 1, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear)
    ttnn.deallocate(part)
    return out


def _gather_sum(y, device, ax):
    """Exact float32 reduce over mesh axis `ax`: gather every partial on a new leading dim, then add."""
    n = tuple(device.shape)[ax]
    shape = list(y.shape)
    if A2A_REDUCE:
        out = _a2a_sum(y, ax, n)
        if out is not None:
            return out
    nbytes = 4 if y.dtype == ttnn.float32 else 2
    for s in shape:
        nbytes *= s
    if nbytes <= _CHUNK_BYTES or len(shape) < 2 or shape[-2] <= 32:
        return _gather_sum_once(y, ax, n)
    # chunk along the token (second-to-last) dim in whole tiles
    rows = shape[-2]
    per_row = nbytes // rows
    step = max(32, (_CHUNK_BYTES // per_row) // 32 * 32)
    outs = []
    for lo in range(0, rows, step):
        hi = min(rows, lo + step)
        start = [0] * len(shape)
        end = list(shape)
        start[-2], end[-2] = lo, hi
        outs.append(_gather_sum_once(ttnn.slice(y, start, end), ax, n))
    return ttnn.concat(outs, dim=len(shape) - 2)


def all_reduce(y, device):
    """Sum of the partials over every device of the mesh (exact float32, see module docstring)."""
    for ax in mesh_axes(device):
        y = _gather_sum(y, device, ax)
    return y


def all_gather(y, device, dim=-1):
    dim = dim % len(y.shape)
    for ax in mesh_axes(device):
        y = ttnn.all_gather(y, dim=dim, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear)
    return y


def all_gather_col_major(y, device, dim=-1):
    """all_gather over the mesh axes in the OTHER order (axis 0 first, on the 1/C-wide shards): device (r, c)'s
    block lands at position c * R + r instead of r * C + c. The callers lay their weights out for that order."""
    dim = dim % len(y.shape)
    for ax in reversed(mesh_axes(device)):
        y = ttnn.all_gather(y, dim=dim, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear)
    return y


def col_major_order(device):
    """[d(p) for gathered position p] under all_gather_col_major: the row-major device index whose block lands at
    position p (p = c * R + r -> d = r * C + c), or None when the mesh is not 2D."""
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return None
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        return None
    rows, cols = shape
    return [(p % rows) * cols + p // rows for p in range(rows * cols)]
