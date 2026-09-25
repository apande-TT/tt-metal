# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collectives and weight placement for the TP transformer ports.

The TP axis is a build-time parameter (`tp_axis(...)` context / `get_tp_axis()`), captured by every
port when it is constructed:
  * None (default, the graduated behaviour): the ports shard with ShardTensorToMesh over ALL devices
    of the mesh (row-major device order), so a TP collective spans every device. On a 1xN mesh that is
    one collective on axis 1; on an RxC mesh it is axis 1 then axis 0. all_gather in that order
    concatenates the shards back in row-major order, and a reduce over both axes sums all R*C partials.
  * 0 or 1: TP over that mesh axis only. Weights are ShardTensor2dMesh(dims=(shard_dim, None)) for
    axis 0 (mirrored for axis 1), i.e. replicated over the other axis, and every collective runs on
    the TP axis alone. The other axis is then free for data parallelism (the Qwen-Image-Edit pipeline
    runs TP=8 over axis 0 of an 8x4 Galaxy and splits the batch over axis 1).

Reduction precision: ttnn.all_reduce / ttnn.reduce_scatter round float32 partials. Measured on this
T3K 2x4 mesh: 7e-3..1.1e-2 max abs error on O(3) float32 sums of [2,64,256]..[4,128,3072] per chip.
That is bf16-level rounding, on EVERY residual update of every block. The ports keep their residual
streams in float32 for a reason, so the reduce here gathers the partials (all_gather moves bits
exactly) and adds them in float32. Measured error of that path: 0.0 on the same tensors. The gather
is chunked along the token dim so the N-fold temporary stays bounded.
"""
from __future__ import annotations

import contextlib

import ttnn

_CHUNK_BYTES = 48 * 1024 * 1024  # per-partial bytes gathered at once
_CURRENT = object()  # sentinel: "the current build-time TP axis"
_TP_AXIS = None  # build-time default; ports capture it in __init__


def get_tp_axis():
    return _TP_AXIS


@contextlib.contextmanager
def tp_axis(axis):
    """Build the ports inside this context to put their TP split on mesh axis `axis` (None = all)."""
    global _TP_AXIS
    assert axis in (None, 0, 1), axis
    prev, _TP_AXIS = _TP_AXIS, axis
    try:
        yield
    finally:
        _TP_AXIS = prev


def _shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return None
    return shape if len(shape) == 2 else None


def _is_mesh(device):
    return isinstance(device, ttnn.MeshDevice) and device.get_num_devices() > 1


def tp_size(device, axis=_CURRENT):
    """TP degree of `device` for a port built with TP axis `axis` (default: the current build axis)."""
    axis = _TP_AXIS if axis is _CURRENT else axis
    if not _is_mesh(device):
        return 1
    if axis is None:
        return device.get_num_devices()
    shape = _shape(device)
    return shape[axis] if shape is not None else device.get_num_devices()


def shard_mapper(device, dim, axis=_CURRENT):
    """Mesh mapper that splits `dim` over the TP axis (every device when the axis is None) and
    replicates over the other mesh axis."""
    axis = _TP_AXIS if axis is _CURRENT else axis
    shape = _shape(device)
    if axis is None or shape is None:
        return ttnn.ShardTensorToMesh(device, dim=dim)
    dims = [None, None]
    dims[axis] = dim
    return ttnn.ShardTensor2dMesh(device, mesh_shape=shape, dims=tuple(dims))


def mesh_axes(device, axis=_CURRENT):
    """The mesh axes a TP collective runs over, in order."""
    axis = _TP_AXIS if axis is _CURRENT else axis
    shape = _shape(device)
    if shape is None:
        try:
            return [1] if device.get_num_devices() > 1 else []
        except AttributeError:
            return []
    if axis is not None:
        return [axis] if shape[axis] > 1 else []
    return [ax for ax in (1, 0) if shape[ax] > 1]


def _gather_sum_once(y, ax, n):
    shape = list(y.shape)
    g = ttnn.all_gather(
        ttnn.reshape(y, [1] + shape), dim=0, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear
    )
    out = None
    for i in range(n):
        part = ttnn.slice(g, [i] + [0] * len(shape), [i + 1] + shape)
        out = part if out is None else ttnn.add(out, part)
    ttnn.deallocate(g)
    return ttnn.reshape(out, shape)


def _gather_sum(y, device, ax):
    """Exact float32 reduce over mesh axis `ax`: gather every partial on a new leading dim, then add."""
    n = tuple(device.shape)[ax]
    shape = list(y.shape)
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


def all_reduce(y, device, axis=_CURRENT):
    """Sum of the partials over the TP devices (exact float32, see module docstring). `axis` is the
    port's captured TP axis (default: the current build axis)."""
    for ax in mesh_axes(device, axis):
        y = _gather_sum(y, device, ax)
    return y


def all_gather(y, device, dim=-1, axis=_CURRENT):
    dim = dim % len(y.shape)
    for ax in mesh_axes(device, axis):
        y = ttnn.all_gather(y, dim=dim, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear)
    return y
