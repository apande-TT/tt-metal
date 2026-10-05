# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the prefill head split + RoPE: the fused qkv projection straight to RoPE'd q / k and
plain v in ONE ttnn.generic_op (reader / writer in tt/cpp_qkv_rope_kernels, compute = the stock RoPE kernel).

Stock, a prefill layer runs `nlp_create_qkv_heads` (every section as a q head), three head-range slices (q and k
straight into the RoPE's height-sharded layout) and two `ttnn.experimental.rotary_embedding` ops -- ~65 us a
640-row tail layer, mostly moving the same 8 MB around. Here the reader feeds the STOCK RoPE compute kernel
-- the rotated / sin / input / cos tiles of each head tile row, read straight out of the fused qkv rows -- to a
compute kernel that runs the stock RoPE kernel's ops (`rotary_embedding.cpp`, multi-tile path: FPU mul by the -1
scalar on the first half, FPU mul by sin and by cos, FPU add, HiFi4, 16-bit DEST, bf16 intermediates) over the
row's tiles under one init each (the stock kernel itself, re-initialising per tile, ran 49 us here); the writer
scatters the results into q / k and copies v. The bits are the stock RoPE's.

On unless VOXTRAL_CPP_QKV_ROPE=0.
"""
from __future__ import annotations

import os
import pathlib
import struct

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_qkv_rope_kernels"
_READER = str(_DIR / "reader.cpp")
_WRITER = str(_DIR / "writer.cpp")
_COMPUTE = str(_DIR / "compute.cpp")

_TILE = 32
_BF16_TILE = 2048
_MINUS_ONE_BF16 = struct.unpack("<I", struct.pack("<f", -1.0))[0] >> 16


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_QKV_ROPE", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_qkv_rope: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _plain_bf16(t):
    return t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded()


def _split(device, units):
    grid = device.compute_with_storage_grid_size()
    cap = int(grid.x) * int(grid.y)
    return next(c for c in range(min(cap, units), 0, -1) if units % c == 0)


def supports(qkv, n_heads, n_kv, cos, sin) -> bool:
    """qkv: bf16 TILE interleaved `[1, 1, S, (n_heads + 2 n_kv) * D]`; cos / sin: bf16 `[1, 1, S, D]` tables (the
    stock fused RoPE's prefill form); D an even number of tiles."""
    try:
        if not enabled():
            return False
        s, d = int(cos.shape[-2]), int(cos.shape[-1])
        return (
            all(_plain_bf16(t) for t in (qkv, cos, sin))
            and [int(x) for x in qkv.shape] == [1, 1, s, (n_heads + 2 * n_kv) * d]
            and [int(x) for x in qkv.padded_shape] == [1, 1, s, (n_heads + 2 * n_kv) * d]
            and [int(x) for x in cos.shape] == [1, 1, s, d]
            and [int(x) for x in sin.shape] == [1, 1, s, d]
            and s % _TILE == 0
            and d % (2 * _TILE) == 0
            and _split(qkv.device(), (n_heads + n_kv) * (s // _TILE)) >= 8
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, tiles):
    return ttnn.CBDescriptor(
        total_size=tiles * _BF16_TILE,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=_BF16_TILE)],
    )


def apply(qkv, n_heads, n_kv, cos, sin, v_memory_config=None):
    """`(q, k, v)`: q `[1, n_heads, S, D]` and k `[1, n_kv, S, D]` RoPE'd (L1), v `[1, n_kv, S, D]` (in
    `v_memory_config`, default L1) -- the stock head split + `rotary_embedding` pair, bit for bit."""
    device = qkv.device()
    s, d = int(cos.shape[-2]), int(cos.shape[-1])
    wt, rt = d // _TILE, s // _TILE
    roww = (n_heads + 2 * n_kv) * wt
    l1 = ttnn.L1_MEMORY_CONFIG
    q = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n_heads, s, d]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, l1)
    k = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n_kv, s, d]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, l1)
    v = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, n_kv, s, d]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, v_memory_config or l1
    )
    units = (n_heads + n_kv) * rt  # (tile row, q / k head) units: every core takes the same count
    ncores = _split(device, units)
    per = units // ncores
    vunits = n_kv * rt
    vbase, vextra = divmod(vunits, ncores)
    grid = device.compute_with_storage_grid_size()
    gx = int(grid.x)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    xa, ca, sa = qkv.buffer_address(), cos.buffer_address(), sin.buffer_address()
    qa, ka, va = q.buffer_address(), k.buffer_address(), v.buffer_address()
    rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    w0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nv = vbase + (1 if c < vextra else 0)
        rr[cx][cy] = [xa, ca, sa, c * per, per, w0, nv]
        rw[cx][cy] = [qa, ka, va, c * per, per, w0, nv]
        rc[cx][cy] = [c * per, per]
        w0 += nv
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, wt // 2, n_heads, n_kv, roww, _MINUS_ONE_BF16]
            + _accessor_args(qkv)
            + _accessor_args(cos)
            + _accessor_args(sin),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, n_heads, n_kv, rt] + _accessor_args(q) + _accessor_args(k) + _accessor_args(v),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, n_heads + n_kv],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]
    cfg = kernels[2].config
    # The stock multi-tile factory's ComputeConfigDescriptor{}: HiFi4, 16-bit DEST.
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    cbs = [
        _cb(cores, 0, 2 * wt),  # input
        _cb(cores, 1, 2 * wt),  # rotated input
        _cb(cores, 2, 2 * wt),  # cos
        _cb(cores, 3, 2 * wt),  # sin
        _cb(cores, 4, 1),  # the -1 scalar
        _cb(cores, 24, wt),  # rotated interm (half a row)
        _cb(cores, 25, wt),  # cos interm (a row)
        _cb(cores, 26, wt),  # sin interm (a row)
        _cb(cores, 16, 2 * wt),  # out
        _cb(cores, 17, 2 * wt),  # v pass-through
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([qkv, cos, sin, q, k, v], desc)
    return q, k, v
