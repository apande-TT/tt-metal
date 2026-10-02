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
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SCORES_DEC", "1") == "1"


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


def supports(q, k) -> bool:
    try:
        b, h, rows, d = (int(s) for s in q.shape)
        kb, kh, span, kd = (int(s) for s in k.shape)
        return (
            enabled()
            and rows == _TILE
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


def _cb(cores, index, fmt, tiles):
    page = _TILE_BYTES[fmt]
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def apply(q, k):
    """`q @ k^T` per (user, kv head), float32 `[B, n_kv, 32, span]` in L1."""
    device = q.device()
    b, h, rows, d = (int(s) for s in q.shape)
    span = int(k.shape[-2])
    dt, st, units = d // _TILE, span // _TILE, b * h
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([b, h, rows, span]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
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
            compile_time_args=[dt, st] + _accessor_args(q) + _accessor_args(k),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, st],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[st] + _accessor_args(y),
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
        _cb(cores, 1, k.dtype, 2 * dt),
        _cb(cores, 16, ttnn.float32, 2),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = hash(("voxtral_cpp_scores_dec", b, h, dt, st, str(k.dtype), qa, ka, ya)) & 0xFFFFFFFFFFFFFFFF
    ttnn.generic_op([q, k, y], desc)
    return y
