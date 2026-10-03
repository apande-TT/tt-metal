# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the acoustic attention softmax: `softmax(raw + mask, dim=-1)` over the
float32 scores `[1, H, M, S]`, through ttnn.generic_op (kernels in tt/cpp_softmax_kernels).

The stock chain is five ops, each a full pass over the [1, H, M, S] float32 tensor in DRAM: the mask
add, max, subtract + EXP, sum and divide. Here each unit (one head's 32-row tile row, S/32 tiles)
is read once (raw scores + mask), runs the SAME float32 SFPU primitives in the same order with every
operand unpacked straight to DEST, and its weights are written once -- bit-identical to the stock
chain. The units are dealt out over the whole grid.

On unless VOXTRAL_CPP_SOFTMAX=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_softmax_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64
# The CBs the compute kernel copy_tile()s from: raw, mask, scores, filled max, exp, filled sum.
_UNPACK_TO_DEST = (0, 1, 2, 4, 5, 7)


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SOFTMAX", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_softmax: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(raw, mask) -> bool:
    try:
        b, h, m, s = (int(x) for x in raw.shape)
        mb, mh, mm, ms = (int(x) for x in mask.shape)
        return (
            enabled()
            and b == 1
            and mb == 1
            and mh in (1, h)
            and (mm, ms) == (m, s)
            and m % _TILE == 0
            and s % _TILE == 0
            and raw.dtype == ttnn.float32
            and mask.dtype == ttnn.float32
            and raw.layout == ttnn.TILE_LAYOUT
            and mask.layout == ttnn.TILE_LAYOUT
            and not raw.is_sharded()
            and not mask.is_sharded()
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


def apply(raw, mask, memory_config=None):
    """`softmax(raw + mask, dim=-1)` as float32 `[1, H, M, S]` (DRAM unless `memory_config`)."""
    device = raw.device()
    _, h, m, s = (int(x) for x in raw.shape)
    mh = int(mask.shape[1])
    mt, st = m // _TILE, s // _TILE
    units = h * mt
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, h, m, s]), ttnn.float32, ttnn.TILE_LAYOUT, device, memory_config or ttnn.DRAM_MEMORY_CONFIG
    )
    ra, ma, ya = raw.buffer_address(), mask.buffer_address(), y.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [ra, ma, u0, nu]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [ya, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[st, mt, mh] + _accessor_args(raw) + _accessor_args(mask),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[st],
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
    # The stock reduce and binary_ng float32 SFPU config: HiFi4, fp32 DEST, exact (non-approx) SFPU
    # functions, half-sync DEST, and every operand unpacked straight to DEST (no SrcA tf32 step).
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    for i in _UNPACK_TO_DEST:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, 2),  # raw scores, a tile at a time
        _cb(cores, 1, 2),  # mask, a tile at a time
        _cb(cores, 2, st),  # s = raw + mask
        _cb(cores, 3, 1),  # row max (column 0)
        _cb(cores, 4, 1),  # row max, column-filled
        _cb(cores, 5, st),  # e = exp(s - max)
        _cb(cores, 6, 1),  # row sum (column 0)
        _cb(cores, 7, 1),  # row sum, column-filled
        _cb(cores, 16, 2),  # weights, a tile at a time
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = (
        hash(("voxtral_cpp_softmax", h, mt, st, mh, ra, ma, ya, str(memory_config))) & 0xFFFFFFFFFFFFFFFF
    )
    ttnn.generic_op([raw, mask, y], desc)
    return y
