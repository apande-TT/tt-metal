# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the decode head split + RoPE: the fused qkv projection of a decode step straight to
RoPE'd q (interleaved) and RoPE'd k / plain v (height-sharded, one user a core -- the cache update's input) in
ONE ttnn.generic_op (reader / writer in tt/cpp_qkv_rope_dec_kernels, compute = tt/cpp_rope_dec's).

Stock, a decode layer step runs nlp_create_qkv_heads_decode (into the shard layout), two sharded-to-interleaved
moves of q and k, the RoPE (tt/cpp_rope_dec) and an interleaved-to-sharded move of k back for the cache update,
~33 us. Here core b takes user b: the reader gathers its q / k / v rows out of the fused tiles (each user is a
ROW of every (head, column) tile there, the decode layout wants the heads as the rows of one tile row a user)
-- pure data movement --, v straight into the core's own v shard; the compute is cpp_rope_dec's (binary_ng's
float32 SFPU multiplies and add in its order); the writer sends q to the interleaved q and k into the core's own
k shard. The same bits.

On unless VOXTRAL_CPP_QKV_ROPE_DEC=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_qkv_rope_dec_kernels"
_READER = str(_DIR / "reader.cpp")
_WRITER = str(_DIR / "writer.cpp")
_COMPUTE = str(pathlib.Path(__file__).resolve().parent / "cpp_rope_dec_kernels" / "compute.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_QKV_ROPE_DEC", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_qkv_rope_dec: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _shard_cores(shard_mem):
    """The shard grid's cores in shard order (row-major)."""
    grid = shard_mem.shard_spec.grid
    return ttnn.corerange_to_cores(grid, row_wise=True), grid


def supports(fused, n_heads, n_kv, cos, sin, shard_mem) -> bool:
    """fused: float32 TILE interleaved `[1, 1, B, (n_heads + 2 n_kv) * D]` with B one tile row; cos / sin: the
    full `[1, 1, 32, D]` float32 rows (cpp_rope_dec.full_rows); shard_mem: the decode shard layout, one user a
    core (one tile row of D a shard); n_heads a full tile of rows, n_kv at most one."""
    try:
        if not enabled() or shard_mem is None or not shard_mem.is_sharded():
            return False
        d = int(cos.shape[-1])
        b = int(fused.shape[-2])
        cores, _ = _shard_cores(shard_mem)
        return (
            fused.dtype == ttnn.float32
            and fused.layout == ttnn.TILE_LAYOUT
            and not fused.is_sharded()
            and [int(x) for x in fused.shape] == [1, 1, b, (n_heads + 2 * n_kv) * d]
            and b == _TILE
            and n_heads == _TILE
            and 0 < n_kv <= _TILE
            and d % (2 * _TILE) == 0
            and all(
                t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded()
                and [int(x) for x in t.shape] == [1, 1, _TILE, d]
                for t in (cos, sin)
            )
            and len(cores) == b
            and list(shard_mem.shard_spec.shape) == [_TILE, d]
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, tiles):
    return ttnn.CBDescriptor(
        total_size=tiles * _FP32_TILE,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=_FP32_TILE)
        ],
    )


def apply(fused, n_heads, n_kv, cos, sin, shard_mem):
    """`(q, k, v)`: q `[B, 1, n_heads, D]` float32 L1 interleaved, RoPE'd; k (RoPE'd) and v `[1, B, n_kv, D]`
    float32 in `shard_mem` -- the stock nlp_create_qkv_heads_decode + moves + RoPE, bit for bit."""
    device = fused.device()
    d = int(cos.shape[-1])
    b = int(fused.shape[-2])
    dt = d // _TILE
    q = ttnn.allocate_tensor_on_device(
        ttnn.Shape([b, 1, n_heads, d]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    k = ttnn.allocate_tensor_on_device(ttnn.Shape([1, b, n_kv, d]), ttnn.float32, ttnn.TILE_LAYOUT, device, shard_mem)
    v = ttnn.allocate_tensor_on_device(ttnn.Shape([1, b, n_kv, d]), ttnn.float32, ttnn.TILE_LAYOUT, device, shard_mem)
    core_list, cores = _shard_cores(shard_mem)
    fa, ca, sa = fused.buffer_address(), cos.buffer_address(), sin.buffer_address()
    qa, ka, va = q.buffer_address(), k.buffer_address(), v.buffer_address()
    rr, rw, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for user, core in enumerate(core_list):
        cx, cy = int(core.x), int(core.y)
        rr[cx][cy] = [fa, ca, sa, user, va]
        rw[cx][cy] = [qa, ka, user]
        rc[cx][cy] = [2]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, n_heads, n_kv] + _accessor_args(fused) + _accessor_args(cos) + _accessor_args(sin),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt] + _accessor_args(q),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # cpp_rope_dec's: binary_ng's float32 SFPU path, fp32 DEST, operands unpacked straight to DEST.
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    for i in (0, 1, 2):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [_cb(cores, 0, 2 * dt), _cb(cores, 1, dt), _cb(cores, 2, dt), _cb(cores, 16, 2 * dt)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([fused, cos, sin, q, k, v], desc)
    return q, k, v
