# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the codec's waveform flatten, through ttnn.generic_op (kernels in
tt/cpp_wave_kernels).

The codec's output projection leaves channels-last float32 TILE `[B, L, C]` (C = patch_size = 240);
the waveform is its row-major flatten `[B, 1, L * C]`. On TILE tensors that reshape re-tiles every
row (~770 us a body at L = 256) and the two batch halves are then concatenated. Here each unit (one
output row, one 32-row tile row) reads that tile row's tiles (one burst each) and gathers their 64-byte
face-row runs L1 -> L1 into a row-major block, and the writer puts the block as one contiguous run into the ROW_MAJOR
output at its final position -- both bodies' rows land in one tensor, so the concat goes
too. Pure data movement: bit-identical. ttnn.to_torch reads the ROW_MAJOR result as before.

On unless VOXTRAL_CPP_WAVE=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_wave_kernels"
_READER = str(_DIR / "reader.cpp")
_WRITER = str(_DIR / "writer.cpp")
_TILE = 32
_FP32_TILE = 4096


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_WAVE", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_wave: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _rows_cols(x):
    shape = [int(d) for d in x.shape]
    return shape[0], shape[-2], shape[-1]


def supports(parts) -> bool:
    if not enabled() or not parts or len(parts) > 2:
        return False
    try:
        dims = [_rows_cols(p) for p in parts]
        _, length, cols = dims[0]
        return all(
            p.dtype == ttnn.float32
            and p.layout == ttnn.TILE_LAYOUT
            and not p.is_sharded()
            and (len(p.shape) == 3 or (len(p.shape) == 4 and int(p.shape[1]) == 1))
            and d[1:] == (length, cols)
            for p, d in zip(parts, dims)
        ) and cols % 16 == 0 and cols <= 8 * _TILE
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def apply(parts):
    """Row-major float32 `[sum B_i, 1, L * C]` from channels-last TILE `[B_i, (1,) L, C]` parts."""
    device = parts[0].device()
    dims = [_rows_cols(p) for p in parts]
    length, cols = dims[0][1], dims[0][2]
    b1 = dims[0][0]
    batch = sum(d[0] for d in dims)
    lt, ct = -(-length // _TILE), -(-cols // _TILE)
    units = batch * lt
    grid = device.compute_with_storage_grid_size()
    gx = int(grid.x)
    ncores = min(units, gx * int(grid.y))
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, 1, length * cols]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    a, b = parts[0], parts[-1]
    aa, ba, ya = a.buffer_address(), b.buffer_address(), y.buffer_address()
    rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [aa, ba, u0, nu]
        rw[cx][cy] = [ya, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[lt, ct, b1 if len(parts) == 2 else batch, length, cols] + _accessor_args(a) + _accessor_args(b),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[lt, length, cols] + _accessor_args(y),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    scratch = _TILE * cols * 4
    cbs = [
        ttnn.CBDescriptor(
            total_size=ct * _FP32_TILE,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=_FP32_TILE)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * scratch,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.float32, page_size=scratch)],
        ),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = (
        hash(("voxtral_cpp_wave", batch, b1, lt, ct, length, cols, len(parts), aa, ba, ya)) & 0xFFFFFFFFFFFFFFFF
    )
    ttnn.generic_op(list(parts) + [y], desc)
    return y
