# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the decode step's input fold: `[B, 1, 1, D]` -> `[1, 1, B, D]` as ONE ttnn.generic_op
(kernel in tt/cpp_fold_rows_kernels).

Each sample's single embedding row sits padded out to its own tile row; the decode stream wants the B rows as
ONE tile row. Stock that is a reshape that re-tiles the whole padded tensor (~33 us a step at B = 32, D = 3072).
Here each core gathers, for its output column tiles, row 0 of every sample's tile into rows 0..B-1 -- just the
B real rows move. Pure data movement: the same bits.

On unless VOXTRAL_CPP_FOLD_ROWS=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_KERNEL = str(pathlib.Path(__file__).resolve().parent / "cpp_fold_rows_kernels" / "fold.cpp")
_TILE = 32
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_FOLD_ROWS", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_fold_rows: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(x) -> bool:
    """x: TILE interleaved float32 / bf16 `[B, 1, 1, D]`, B <= 32, D a tile multiple."""
    try:
        shape = [int(d) for d in x.shape]
        return (
            enabled()
            and len(shape) == 4
            and shape[1] == 1
            and shape[2] == 1
            and 1 <= shape[0] <= _TILE
            and shape[3] % _TILE == 0
            and x.layout == ttnn.TILE_LAYOUT
            and x.dtype in _TILE_BYTES
            and not x.is_sharded()
            and [int(d) for d in x.padded_shape] == [shape[0], 1, _TILE, shape[3]]
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def apply(x, memory_config=None):
    """`x` `[B, 1, 1, D]` as `[1, 1, B, D]` (in `memory_config`, else x's)."""
    device = x.device()
    b, _, _, d = (int(v) for v in x.shape)
    dt = d // _TILE
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(dt, gx * gy)
    base, extra = divmod(dt, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, b, d]), x.dtype, ttnn.TILE_LAYOUT, device, memory_config or x.memory_config()
    )
    xa, ya = x.buffer_address(), y.buffer_address()
    rt = ttnn.RuntimeArgs()
    j0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nj = base + (1 if c < extra else 0)
        rt[cx][cy] = [xa, ya, j0, nj]
        j0 += nj
    size = _TILE_BYTES[x.dtype]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[b, dt, size // 64] + _accessor_args(x) + _accessor_args(y),
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=x.dtype, page_size=size)],
        )
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([x, y], desc)
    return y
