# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the acoustic semantic argmax, through ttnn.generic_op (kernel in
tt/cpp_argmax_kernels): the row argmax of the float32 TILE logits `[B <= 32, W]` read in place (no
untilize), as uint32 `[B, 1]` ROW_MAJOR in L1 -- ttnn.argmax's output contract.

One core per row; its two RISCs scan half the row's tiles each (two 64-byte face segments a tile,
all reads under one barrier), comparing floats as order-preserving unsigned keys and keeping the
first column of the maximum (torch's tie rule); RISC 1 hands its best to RISC 0 through a CB.

On unless VOXTRAL_CPP_ARGMAX=0 -- a measured rung (stock: untilize + ~250 us ttnn.argmax a call).
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_argmax_kernels"
_KERNEL = str(_DIR / "argmax_row.cpp")
_TILE = 32


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_ARGMAX", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_argmax: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(x) -> bool:
    try:
        rows, w = (int(s) for s in x.shape)
        return (
            enabled()
            and rows <= _TILE
            and x.dtype == ttnn.float32
            and x.layout == ttnn.TILE_LAYOUT
            and not x.is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def argmax(x):
    """Row argmax of `x [B, W]` float32 TILE as uint32 `[B, 1]` ROW_MAJOR in L1."""
    device = x.device()
    rows, w = (int(s) for s in x.shape)
    nt = -(-w // _TILE)
    half = (nt + 1) // 2
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(rows, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([rows, 1]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    xa, ya = x.buffer_address(), y.buffer_address()
    gx = int(grid.x)
    r0, r1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for b in range(rows):
        cy, cx = divmod(b, gx)
        r0[cx][cy] = [xa, ya, b, 0, half]
        r1[cx][cy] = [xa, ya, b, half, nt]

    def kern(role, rt, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[role, w, role, 2] + _accessor_args(x) + _accessor_args(y),
            runtime_args=rt,
            config=cfg,
        )

    scratch = half * 128
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=size)],
        )
        for i, size in ((0, scratch), (1, scratch), (2, 32))
    ]
    desc = ttnn.ProgramDescriptor(
        kernels=[kern(0, r0, ttnn.ReaderConfigDescriptor()), kern(1, r1, ttnn.WriterConfigDescriptor())], semaphores=[], cbs=cbs
    )
    desc.custom_program_hash = hash(("voxtral_cpp_argmax", rows, w, xa, ya)) & 0xFFFFFFFFFFFFFFFF
    ttnn.generic_op([x, y], desc)
    return y
