# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode attention's context normalise + head merge as ONE generic_op (kernels in tt/cpp_ctx_merge_kernels).

After P@V the decode attention holds `pv [B, KV, 32, D]` (each kv head's G grouped query rows padded to a tile)
and the softmax row sums `s [B, KV, 32, 1]`. The stubs then ran `ttnn.divide(pv, s)`, a relabel to the G real
rows, an untilize, a row-major reshape to `[1, 1, B, KV * G * D]` (a real relayout, not a view) and a tilize
for o_proj -- four ops, ~51 us a layer step. Here one unit per merged output tile gathers its 32 batch rows
(query row r of kv head g, d-tile dt of every sample) straight from the P@V tiles, fills each row's sum across
the row as binary_ng's column broadcast does, and divides with the same float32 SFPU divide -- the same bits.

On unless VOXTRAL_CPP_CTX_MERGE=0.
"""

from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_ctx_merge_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_CTX_MERGE", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_ctx_merge: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _cb(cores, index, tiles):
    return ttnn.CBDescriptor(
        total_size=tiles * _FP32_TILE,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=_FP32_TILE)
        ],
    )


def supports(pv, s, groups) -> bool:
    if not enabled():
        return False
    try:
        b, kv, rows, d = (int(v) for v in pv.padded_shape)
        sb, skv, srows, sw = (int(v) for v in s.padded_shape)
        return (
            pv.dtype == ttnn.float32
            and s.dtype == ttnn.float32
            and pv.layout == ttnn.TILE_LAYOUT
            and s.layout == ttnn.TILE_LAYOUT
            and rows == _TILE
            and srows == _TILE
            and sw == _TILE
            and int(s.shape[-1]) == 1
            and sb == b
            and (skv == kv or (skv == 1 and kv * int(groups) == _TILE))
            and 0 < int(groups) <= _TILE
            and d % _TILE == 0
            and not pv.is_sharded()
            and not s.is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def apply(pv, s, groups, memory_config=None):
    """`reshape(untilize(pv / s)[:, :, :groups], [1, 1, B, KV * groups * D])` as float32 TILE, in one pass."""
    device = pv.device()
    batch, kv, _, d = (int(v) for v in pv.shape)
    # packed: the row sums come in cpp_scores_dec's packed [B, 1, 32, 1] layout (kv head g's rows at g * groups ..).
    packed = int(s.shape[1]) == 1 and kv > 1
    dt = d // _TILE
    ct = kv * groups * dt
    bt = -(-batch // _TILE)
    units = bt * ct
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, batch, kv * groups * d]), ttnn.float32, ttnn.TILE_LAYOUT, device, mem
    )
    pa, sa, oa = pv.buffer_address(), s.buffer_address(), out.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [pa, sa, u0, nu]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [oa, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[batch, kv, int(groups), dt, int(packed)] + _accessor_args(pv) + _accessor_args(s),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=_accessor_args(out),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # The stock float32 binary_ng divide: fp32 DEST, operands unpacked straight to DEST.
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, 2),  # gathered P@V rows
        _cb(cores, 1, 2),  # column-filled row sums
        _cb(cores, 2, 1),  # reader scratch: the 32 row-sum segments
        _cb(cores, 16, 2),  # merged output tiles
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    # No custom hash: the default one covers every compile arg (the accessors' buffer types too), the CBs and the
    # cores, and leaves the raw addresses out -- a cache hit re-applies this descriptor's runtime args. Hashing
    # the addresses made a step whose tensors landed elsewhere miss the cache, and a miss inside a trace
    # capture is a compile + binary write the capture refuses.
    ttnn.generic_op([pv, s, out], desc)
    return out
