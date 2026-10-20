# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The patch merger's 2 x 2 spatial unfold as ONE data-movement program (generic_op, kernels/patch_unfold_*.cpp) in tile
layout, replacing to_layout(ROW_MAJOR) + reshape + permute + reshape + to_layout(TILE): 286 -> 40 us for 8 images of
16 x 16 patches (195 of the 286 were the tilize).

x (B, 1, h * w, d) bf16 tiles -> (B, 1, h * w / 4, 4 * d) bf16 tiles, merged token (r, s) = r * w / 2 + s holding
[x(2r, 2s) | x(2r, 2s + 1) | x(2r + 1, 2s) | x(2r + 1, 2s + 1)] -- the (a, b, c) order the merging weight's columns were
permuted to (mistral3_patch_merger.py). A work unit is (image, output tile row, channel tile): the 32 merged tokens of
an output tile row come from 128 consecutive source tokens (4 source tile rows), so a unit reads 4 tiles and writes 4,
moving whole 16-element face rows: bytes copied, nothing recomputed (bit-identical)."""
from __future__ import annotations

from pathlib import Path

import ttnn

_DIR = Path(__file__).resolve().parent / "kernels"
_READER = str(_DIR / "patch_unfold_reader.cpp")
_WRITER = str(_DIR / "patch_unfold_writer.cpp")
_TILE = 32
_TILE_BYTES = 2048  # bf16


def supports(x, h, w, k) -> bool:
    shape = tuple(int(s) for s in x.shape)
    return (
        k == 2
        and len(shape) == 4
        and shape[1] == 1
        and shape[2] == h * w
        and x.layout == ttnn.TILE_LAYOUT
        and x.dtype == ttnn.bfloat16
        and not x.is_sharded()
        and shape[3] % _TILE == 0
        and w % 2 == 0
        and h % 2 == 0
        and _TILE % (w // 2) == 0
        and (h * w) % (4 * _TILE) == 0
    )


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("patch_unfold: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def apply(x, h, w, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """x (B, 1, h * w, d) bf16 tiles, interleaved -> (B, 1, h * w / 4, 4 * d) bf16 tiles in memory_config."""
    device = x.device()
    b, _, n, d = (int(s) for s in x.shape)
    dt, rt, st = d // _TILE, n // 4 // _TILE, n // _TILE
    units = b * rt * dt
    grid = device.compute_with_storage_grid_size()
    gx, ncores = int(grid.x), min(int(grid.x) * int(grid.y), units)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([b, 1, n // 4, 4 * d]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)
    base, extra = divmod(units, ncores)
    rr, rw = [], []
    u0 = 0
    for c in range(ncores):
        nu = base + (1 if c < extra else 0)
        cy, cx = divmod(c, gx)
        rr.append(((cx, cy), [x.buffer_address(), u0, nu]))
        rw.append(((cx, cy), [y.buffer_address(), u0, nu]))
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, rt, w, st] + _accessor_args(x),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, rt] + _accessor_args(y),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cb = lambda index, tiles: ttnn.CBDescriptor(
        total_size=tiles * _TILE_BYTES,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=_TILE_BYTES)],
    )
    ttnn.generic_op([x, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(0, 4), cb(16, 8)]))
    return y
