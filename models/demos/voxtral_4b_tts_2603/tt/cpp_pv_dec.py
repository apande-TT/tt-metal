# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on decode attention's P@V `e @ V` ([B, n_kv, 32, span] x [B, n_kv, span, 128]),
through ttnn.generic_op (kernels in tt/cpp_pv_dec_kernels).

The B * n_kv (user, kv head) units are dealt out over the whole grid (some cores take one more).
Per unit a core streams, per span tile, the float32 exp-weight tile and the unit's bf8_b value tile
row straight from the cache cut (no typecast), holding the DT context tiles in fp32 DEST across the
whole span at HiFi4 -- the stock reuse bmm's single-K-block recipe -- and writes float32 to L1.

On unless VOXTRAL_CPP_PV_DEC=0, for every layer built (VOXTRAL_CPP_PV_DEC_LAYERS caps it).
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_pv_dec_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_RB = 4  # span tiles a read barrier
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_PV_DEC", "1") == "1"


_LAYERS = int(os.environ.get("VOXTRAL_CPP_PV_DEC_LAYERS", "1000"))
_claimed = [0]


def claim() -> bool:
    """True for the first VOXTRAL_CPP_PV_DEC_LAYERS layers built (default: all)."""
    if not enabled() or _claimed[0] >= _LAYERS:
        return False
    _claimed[0] += 1
    return True


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_pv_dec: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(e, v, groups=None) -> bool:
    """`v` may be the whole cache: its first `span` rows (e's width) are read in place. With `groups`, e may be
    the packed `[B, 1, 32, span]` (n_kv * groups == 32)."""
    try:
        b, h, rows, span = (int(s) for s in e.shape)
        vb, vh, cap, d = (int(s) for s in v.shape)
        return (
            enabled()
            and rows == _TILE
            and span <= cap
            and cap % _TILE == 0
            and span % _TILE == 0
            and d % _TILE == 0
            and d // _TILE <= 4
            and b == vb
            and (h == vh or (groups and h == 1 and vh * int(groups) == rows))
            and e.dtype == ttnn.float32
            and v.dtype in _TILE_BYTES
            and not e.is_sharded()
            and not v.is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def supports_rows(e, v) -> bool:
    """The full-span multi-row form (the acoustic readout's `[1, H, M, S] @ [1, H, S, D]`): float32 V."""
    try:
        b, h, rows, span = (int(s) for s in e.shape)
        vb, vh, cap, d = (int(s) for s in v.shape)
        return (
            enabled()
            and rows % _TILE == 0
            and span == cap
            and span % _TILE == 0
            and d % _TILE == 0
            and d // _TILE <= 4
            and (b, h) == (vb, vh)
            and e.dtype == ttnn.float32
            and v.dtype == ttnn.float32
            and not e.is_sharded()
            and not v.is_sharded()
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


def apply(e, v, rb=_RB, memory_config=None, groups=None):
    """`e @ v` per (user, kv head), float32 `[B, n_kv, rows, d]` (L1 unless `memory_config`).

    Each query tile row of a (user, kv head) is its own unit; its V tile rows are re-read per unit. With `groups`,
    e is the packed `[B, 1, 32, span]` (cpp_scores_dec packed=True): unit (b, h) gathers its `groups` rows."""
    device = e.device()
    b, h, rows, span = (int(s) for s in e.shape)
    packed = bool(groups) and h == 1 and int(v.shape[1]) * int(groups) == rows
    if packed:
        h = int(v.shape[1])
    d = int(v.shape[-1])
    mt = rows // _TILE
    dt, st, ss, units = d // _TILE, span // _TILE, int(v.shape[-2]) // _TILE, b * h * mt
    rb = min(rb, st)
    pad = -(-st // rb) * rb - st
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([b, h, rows, d]), ttnn.float32, ttnn.TILE_LAYOUT, device, memory_config or ttnn.L1_MEMORY_CONFIG
    )
    ea, va, ya = e.buffer_address(), v.buffer_address(), y.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [ea, va, u0, nu]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [ya, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, st, ss, rb, mt, int(packed), int(groups or 0), h] + _accessor_args(e) + _accessor_args(v),
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
            compile_time_args=[dt] + _accessor_args(y),
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
        _cb(cores, 0, ttnn.float32, 2 * rb),
        _cb(cores, 1, v.dtype, 2 * rb * dt),
        _cb(cores, 16, ttnn.float32, 2 * dt),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    # No custom hash: the default one covers every compile arg (the accessors' buffer types too), the CBs and the
    # cores, and leaves the raw addresses out -- a cache hit re-applies this descriptor's runtime args. Hashing
    # the addresses made a step whose tensors landed elsewhere miss the cache, and a miss inside a trace
    # capture is a compile + binary write the capture refuses.
    ttnn.generic_op([e, v, y], desc)
    return y
