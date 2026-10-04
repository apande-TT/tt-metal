# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on decode attention's scores `q @ K^T` ([B, n_kv, 32, 128] x [B, n_kv, span, 128]),
through ttnn.generic_op (kernels in tt/cpp_scores_dec_kernels).

The B * n_kv (user, kv head) units are dealt out over the whole grid (some cores take one more).
Per unit a core reads the float32 query's tile row and the unit's bf8_b keys one tile row at a
time, straight from the cache cut (no typecast), and sums `q[d] @ k[j, d]^T` over the head_dim
tiles in fp32 DEST at HiFi4 -- the stock reuse bmm's single-K-block recipe -- writing float32
scores to L1.

On unless VOXTRAL_CPP_SCORES_DEC=0, for every layer built (VOXTRAL_CPP_SCORES_DEC_LAYERS caps it).
Measured 2026-10-02: 14.7 us a call vs 25.4 for the stock MatmulMultiCoreReuse bmm (whose 256
one-tile units ran 3 rounds on 110 cores with per-unit program overhead).
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_scores_dec_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_RB = 4  # key tile rows a read barrier
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SCORES_DEC", "1") == "1"


def packed_enabled() -> bool:
    """The packed [B, 1, 32, span] score layout (see `apply`); off with VOXTRAL_CPP_PACKED_DEC=0."""
    return enabled() and os.environ.get("VOXTRAL_CPP_PACKED_DEC", "1") == "1"


_LAYERS = int(os.environ.get("VOXTRAL_CPP_SCORES_DEC_LAYERS", "1000"))
_claimed = [0]


def claim() -> bool:
    """True for the first VOXTRAL_CPP_SCORES_DEC_LAYERS layers built (default: all)."""
    if not enabled() or _claimed[0] >= _LAYERS:
        return False
    _claimed[0] += 1
    return True


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_scores_dec: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(q, k, span=None) -> bool:
    """`span`: read only the first `span` key rows of `k` (the whole cache, read in place)."""
    try:
        b, h, rows, d = (int(s) for s in q.shape)
        kb, kh, cap, kd = (int(s) for s in k.shape)
        span = cap if span is None else int(span)
        return (
            enabled()
            and rows == _TILE
            and span <= cap
            and cap % _TILE == 0
            and d == kd
            and d % _TILE == 0
            and span % _TILE == 0
            and (b, h) == (kb, kh)
            and q.dtype == ttnn.float32
            and k.dtype in _TILE_BYTES
            and not q.is_sharded()
            and not k.is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def supports_grouped(q, k, span, groups) -> bool:
    """The RoPE output `q [B, 1, n_kv * groups, head_dim]` read in place as the grouped query (unit
    (b, h) takes heads h * groups ..), against the whole cache `k` read to `span`."""
    try:
        b, one, nh, d = (int(s) for s in q.shape)
        kb, kh, cap, kd = (int(s) for s in k.shape)
        return (
            enabled()
            and one == 1
            and nh == kh * int(groups)
            and nh <= _TILE
            and b == kb
            and d == kd
            and d % _TILE == 0
            and int(span) <= cap
            and int(span) % _TILE == 0
            and cap % _TILE == 0
            and q.dtype == ttnn.float32
            and q.layout == ttnn.TILE_LAYOUT
            and k.dtype in _TILE_BYTES
            and not q.is_sharded()
            and not k.is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, fmt, tiles):
    page = _TILE_BYTES[fmt]
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def apply(q, k, span=None, groups=None, packed=False):
    """`q @ k^T` per (user, kv head) over the first `span` key rows (default: all), float32
    `[B, n_kv, 32, span]` in L1. With `groups`, q is the RoPE output `[B, 1, n_kv * groups, head_dim]`,
    regrouped by kv head in the reader. `packed` (with `groups`, n_kv * groups == 32): the scores come back
    `[B, 1, 32, span]` -- kv head h's `groups` real rows at rows h * groups .., every row of every tile real,
    so the mask add / max / exp / sum after it run on n_kv-times fewer tiles (each row's values unchanged)."""
    device = q.device()
    if groups:
        b, _, _, d = (int(s) for s in q.shape)
        h, rows = int(k.shape[1]), _TILE
    else:
        b, h, rows, d = (int(s) for s in q.shape)
    cap = int(k.shape[-2])
    span = cap if span is None else int(span)
    dt, st, ss, units = d // _TILE, span // _TILE, cap // _TILE, b * h
    rb = min(_RB, st)
    pad = -(-st // rb) * rb - st
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    packed = bool(packed and groups and h * int(groups) == _TILE)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([b, 1 if packed else h, rows, span]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    qa, ka, ya = q.buffer_address(), k.buffer_address(), y.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [qa, ka, u0, nu]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [ya, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, st, ss, rb, int(groups or 0), h] + _accessor_args(q) + _accessor_args(k),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, st, pad],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[st, int(packed), int(groups or 0), h] + _accessor_args(y),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # HiFi4 + fp32 DEST, as the stock decode bmm runs.
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cbs = [
        _cb(cores, 0, ttnn.float32, 2 * dt),
        _cb(cores, 1, k.dtype, 2 * rb * dt),
        _cb(cores, 16, ttnn.float32, 2),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    # No custom hash: the default one covers every compile arg (the accessors' buffer types too), the CBs and the
    # cores, and leaves the raw addresses out -- a cache hit re-applies this descriptor's runtime args. Hashing
    # the addresses made a step whose tensors landed elsewhere miss the cache, and a miss inside a trace
    # capture is a compile + binary write the capture refuses.
    ttnn.generic_op([q, k, y], desc)
    return y
