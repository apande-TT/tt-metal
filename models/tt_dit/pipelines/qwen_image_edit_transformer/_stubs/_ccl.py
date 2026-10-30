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
# L1 bytes for one gathered chunk of the exact reduce (0: gather into DRAM in _CHUNK_BYTES pieces). With
# it set, each token chunk's gather + slices + float32 adds stay in L1 and only its sum goes to DRAM.
L1_GATHER_BYTES = 0
# with it: each chunk's own partial is also staged in L1, so the gather reads L1 instead of DRAM
L1_GATHER_INPUT = False
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


def _gather_sum_once(y, ax, n, mem=None):
    shape = list(y.shape)
    kw = {} if mem is None else {"memory_config": mem}
    g = ttnn.all_gather(
        ttnn.reshape(y, [1] + shape), dim=0, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear, **kw
    )
    out = None
    for i in range(n):
        part = ttnn.slice(g, [i] + [0] * len(shape), [i + 1] + shape, **kw)
        out = part if out is None else ttnn.add(out, part, **kw)
    ttnn.deallocate(g)
    return ttnn.reshape(out, shape)


def _gather_sum_l1(y, ax, n, nbytes):
    """_gather_sum_once per whole-tile token chunk whose n-fold gather fits L1_GATHER_BYTES; the
    chunk's gather, slices and adds stay in L1 (same adds, same order) and its sum goes to DRAM."""
    shape = list(y.shape)
    rows = shape[-2]
    per_row = n * nbytes // rows
    step = (L1_GATHER_BYTES // per_row) // 32 * 32
    if step >= rows:
        s = _gather_sum_once(y, ax, n, mem=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        return out
    outs = []
    for lo in range(0, rows, step):
        start, end = [0] * len(shape), list(shape)
        start[-2], end[-2] = lo, min(rows, lo + step)
        yc = ttnn.slice(y, start, end, memory_config=ttnn.L1_MEMORY_CONFIG if L1_GATHER_INPUT else None)
        s = _gather_sum_once(yc, ax, n, mem=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(yc)
        outs.append(ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG))
        ttnn.deallocate(s)
    return ttnn.concat(outs, dim=len(shape) - 2)


# exact reduce as an all_to_all (each device receives every partial of its 1/n token slice), local float32
# adds of those 1/n slices, and an all_gather of the sums: ~2(n-1)/n of the tensor on the fabric instead of
# the (n-1)x of gathering every partial, and 1/n of the slices/adds. Same adds in the same order.
A2A_REDUCE = False
A2A_SUM_L1 = False  # the all_to_all's per-source slices and their float32 sum in L1 (all_gathered from there)
# the all_to_all's received partials (the same bytes as its input, read once by the local sum) in L1 when they take at
# most this many bytes (0: DRAM)
A2A_RECV_L1_BYTES = 0


# optional replacement for the all_to_all reduce's local sum of its n source blocks: a callable
# (g, n, part, memory_config) -> sum, e.g. a fused kernel; None: n slices + n - 1 ttnn.add calls
A2A_SUM_FN = None
# the same local sum with the row-parallel layer's bias added after it (all_reduce(..., bias=)): a callable
# (g, n, part, memory_config, bias) -> sum + bias, so the bias rides the all_gather instead of a full-tensor add
# after it; None: the bias is added to the gathered sum
A2A_SUM_BIAS_FN = None
# the local sum (+ bias) with a gated_residual() update in one pass: a callable
# (g, n, part, memory_config, bias_rows or None, gate, residual, split_dim, cluster_axis) -> r_slice + gate * sum,
# or None to run the sum and the update as separate ops (None: separate ops)
A2A_GATED_SUM_FN = None
# the gated residual update r + g * reduce(y) done by the all_to_all reduce on each device's own 1/n slice, before
# the all_gather (which then gathers the updated residual): the same multiply and add per element on 1/n of the
# rows instead of on the whole replicated tensor. Requested per call through gated_residual(); False: never
GATED_RESIDUAL = False
# a gated update split over the tokens may keep its result as this device's own token slice (a Sliced, no
# all_gather of the updated residual) when its caller asks (gated_residual(..., keep=True)); False: always gathered
KEEP_SLICES = False
_PENDING = []  # [GatedResidual] registered by gated_residual(), consumed in order by matching reduces


class Sliced:
    """This device's 1/n slice (dim `dim` of the [B, S, W] stream, over mesh axis `axis`) of a stream of shape
    `shape`, as the all_to_all reduce left it: device k holds slice k, so all_gather restores the stream."""

    def __init__(self, t, dim, axis, shape):
        self.t, self.dim, self.axis, self.shape = t, dim, axis, list(shape)

    def gather_part(self, t):
        """all_gather of a tensor laid out like the slice (e.g. a per-slice result) along the slice's dim."""
        return ttnn.all_gather(
            t,
            dim=self.dim,
            cluster_axis=self.axis,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def gather(self):
        return self.gather_part(self.t)


def unslice(x):
    return x.gather() if isinstance(x, Sliced) else x


class GatedResidual:
    """One pending r + g * reduce(y): `out` holds the updated residual once a reduce has applied it (a Sliced when
    `keep` and the reduce split the tokens)."""

    def __init__(self, gate, residual, keep=False):
        self.gate, self.residual, self.out, self.keep = gate, residual, None, keep


@contextlib.contextmanager
def gated_residual(*pairs, keep=False):
    """Offer (gate, residual) pairs to the reduces run inside the block: a reduce whose output has the
    residual's shape takes the first such pair and returns r + g * sum (entry.out) instead of the sum.
    keep: the updated residual may stay this device's token slice (KEEP_SLICES)."""
    entries = [GatedResidual(g, r, keep and KEEP_SLICES) for g, r in pairs] if GATED_RESIDUAL else []
    _PENDING.extend(entries)
    try:
        yield entries
    finally:
        for e in entries:
            if e in _PENDING:
                _PENDING.remove(e)


def _take_gated(y):
    for e in _PENDING:
        if list(e.residual.shape) == list(y.shape) and len(e.gate.shape) == len(y.shape):
            _PENDING.remove(e)
            return e
    return None


def _a2a_split_dim(lead, n):
    """The 4D dim _a2a_sum splits over the devices: the rows when they split into whole tiles, else the
    batch (dim 1) when it splits evenly; None when neither does."""
    if lead[2] % (32 * n) == 0:
        return 2
    if lead[1] % n == 0:
        return 1
    return None


def _a2a_sum(y, ax, n, bias=None, gated=None):
    """bias (needs A2A_SUM_BIAS_FN): added to the local sum, before the all_gather. gated (GatedResidual): the
    local sum s becomes r + g * s on this device's slice of r (and of g where g spans the split dim)."""
    shape = list(y.shape)
    lead = [1] * (4 - len(shape)) + shape  # 4D for the all_to_all
    d = _a2a_split_dim(lead, n)
    y4 = ttnn.reshape(y, lead)
    # out_dim d is split over the devices and in_dim (0) grows by n: block i of dim 0 = source device i's
    # partial of this device's slice of dim d
    nbytes = 4 if y.dtype == ttnn.float32 else 2
    for v in shape:
        nbytes *= v
    recv_l1 = 0 < nbytes <= A2A_RECV_L1_BYTES
    g = ttnn.experimental.all_to_all_async_generic(
        y4,
        in_dim=0,
        out_dim=d,
        cluster_axis=ax,
        topology=ttnn.Topology.Linear,
        **({"memory_config": ttnn.L1_MEMORY_CONFIG} if recv_l1 else {}),
    )
    part = list(g.shape)
    part[0] //= n
    gated_done = False
    kw = {"memory_config": ttnn.L1_MEMORY_CONFIG} if A2A_SUM_L1 else {}
    out = None
    # a residual already sliced over the tokens is this device's slice; a kept result stays one (in DRAM: it
    # lives until the next block's update)
    local = gated is not None and isinstance(gated.residual, Sliced)
    keep = gated is not None and gated.keep and d == 2
    residual = gated.residual.t if local else gated.residual if gated is not None else None
    if gated is not None and A2A_GATED_SUM_FN is not None:
        mem = ttnn.DRAM_MEMORY_CONFIG if keep else kw.get("memory_config", ttnn.DRAM_MEMORY_CONFIG)
        out = A2A_GATED_SUM_FN(g, n, part, mem, bias, gated.gate, residual, d, ax)
    if out is not None:  # the sum, bias and gated residual update in one pass
        ttnn.deallocate(g)
        gated_done = True
    elif bias is not None:
        out = A2A_SUM_BIAS_FN(g, n, part, kw.get("memory_config", ttnn.DRAM_MEMORY_CONFIG), bias)
    elif A2A_SUM_FN is not None:
        out = A2A_SUM_FN(g, n, part, kw.get("memory_config", ttnn.DRAM_MEMORY_CONFIG))
    else:
        out = None
        for i in range(n):
            p = ttnn.slice(g, [i * part[0], 0, 0, 0], [(i + 1) * part[0]] + part[1:], **kw)
            out = p if out is None else ttnn.add(out, p, **kw)
    if not gated_done:
        ttnn.deallocate(g)
    if gated is not None and not gated_done:
        if local:
            r4 = ttnn.reshape(residual, part)
        else:
            r4 = ttnn.mesh_partition(ttnn.reshape(residual, lead), dim=d, cluster_axis=ax)
        g4 = ttnn.reshape(gated.gate, [1] * (4 - len(shape)) + list(gated.gate.shape))
        if int(g4.shape[d]) > 1:
            g4 = ttnn.mesh_partition(g4, dim=d, cluster_axis=ax)
        s = out
        out = ttnn.add(r4, ttnn.multiply(g4, s), **({"memory_config": ttnn.DRAM_MEMORY_CONFIG} if keep else {}))
        ttnn.deallocate(s)
    if keep:
        gated.out = Sliced(ttnn.reshape(out, shape[:-2] + [int(part[2]), shape[-1]]), len(shape) - 2, ax, shape)
        return gated.out
    out = ttnn.all_gather(
        out, dim=d, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.reshape(out, shape)
    if gated is not None:
        gated.out = out
    return out


def _a2a_ok(y, n):
    shape = list(y.shape)
    return A2A_REDUCE and 2 <= len(shape) <= 4 and _a2a_split_dim([1] * (4 - len(shape)) + shape, n) is not None


def _gather_sum(y, device, ax):
    """Exact float32 reduce over mesh axis `ax`: gather every partial on a new leading dim, then add."""
    n = tuple(device.shape)[ax]
    shape = list(y.shape)
    nbytes = 4 if y.dtype == ttnn.float32 else 2
    for s in shape:
        nbytes *= s
    if _a2a_ok(y, n):
        return _a2a_sum(y, ax, n)
    if L1_GATHER_BYTES and len(shape) >= 2 and shape[-2] % 32 == 0 and n * nbytes * 32 // shape[-2] <= L1_GATHER_BYTES:
        return _gather_sum_l1(y, ax, n, nbytes)
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


def all_reduce(y, device, axis=_CURRENT, bias=None):
    """Sum of the partials over the TP devices (exact float32, see module docstring). `axis` is the
    port's captured TP axis (default: the current build axis). bias: added to the sum (ttnn.add(sum, bias));
    folded into the all_to_all reduce's local sum when A2A_SUM_BIAS_FN can take it."""
    axes = mesh_axes(device, axis)
    fold = (
        bias is not None
        and A2A_SUM_BIAS_FN is not None
        and len(axes) == 1
        and bias.dtype == ttnn.float32
        and y.dtype == ttnn.float32
        and int(bias.padded_shape[-1]) == int(y.padded_shape[-1])
        and _a2a_ok(y, tuple(device.shape)[axes[0]])
    )
    gated = _take_gated(y) if _PENDING and len(axes) == 1 and _a2a_ok(y, tuple(device.shape)[axes[0]]) else None
    if fold:
        from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _precise

        return _a2a_sum(y, axes[0], tuple(device.shape)[axes[0]], bias=_precise._bias_rows(bias), gated=gated)
    if gated is not None and bias is None:
        return _a2a_sum(y, axes[0], tuple(device.shape)[axes[0]], gated=gated)
    if gated is not None:  # a bias the local sum cannot take: give the pair back to the caller
        _PENDING.insert(0, gated)
    for ax in axes:
        y = _gather_sum(y, device, ax)
    return y if bias is None else ttnn.add(y, bias)


def all_gather(y, device, dim=-1, axis=_CURRENT):
    dim = dim % len(y.shape)
    for ax in mesh_axes(device, axis):
        y = ttnn.all_gather(y, dim=dim, cluster_axis=ax, num_links=1, topology=ttnn.Topology.Linear)
    return y
