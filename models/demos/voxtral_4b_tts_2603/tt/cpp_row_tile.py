# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on a decode step's one-row tilizes (the RoPE cos / sin rows, the mask row): a row-major
float32 `[..., 1, W]` row as TILE `[1, 1, 1, W]` in ONE small ttnn.generic_op (kernel in tt/cpp_row_tile_kernels).

Stock, `ttnn.to_layout(row, TILE_LAYOUT)` runs tilize-with-padding on ONE core (10-15 us a row). Here each core
copies its tiles' 128-byte slice of the row into row 0 (face 0 / face 1) of a zeroed tile -- the padding rows
are zero, as the stock op's are. Pure data movement: the same bits.

On unless VOXTRAL_CPP_ROW_TILE=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_KERNEL = str(pathlib.Path(__file__).resolve().parent / "cpp_row_tile_kernels" / "row_tile.cpp")
_TILE = 32


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_ROW_TILE", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_row_tile: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(x) -> bool:
    """x: ROW_MAJOR float32 interleaved, one row of W (a tile multiple) -- every leading dim 1."""
    try:
        shape = [int(d) for d in x.shape]
        return (
            enabled()
            and x.layout == ttnn.ROW_MAJOR_LAYOUT
            and x.dtype == ttnn.float32
            and not x.is_sharded()
            and all(d == 1 for d in shape[:-1])
            and shape[-1] % _TILE == 0
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def apply(x, memory_config=None):
    """The row `x` as a float32 TILE `[1, 1, 1, W]` (in `memory_config`, else x's)."""
    device = x.device()
    w = int(x.shape[-1])
    nt = w // _TILE
    grid = device.compute_with_storage_grid_size()
    gx = int(grid.x)
    ncores = min(nt, gx * int(grid.y))
    base, extra = divmod(nt, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, w]), ttnn.float32, ttnn.TILE_LAYOUT, device, memory_config or x.memory_config()
    )
    rt = ttnn.RuntimeArgs()
    j0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nj = base + (1 if c < extra else 0)
        rt[cx][cy] = [x.buffer_address(), y.buffer_address(), j0, nj]
        j0 += nj
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=_accessor_args(x) + _accessor_args(y),
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=4096)],
        )
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([x, y], desc)
    return y
