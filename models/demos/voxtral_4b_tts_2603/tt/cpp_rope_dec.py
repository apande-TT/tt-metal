# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the decode RoPE: q and k through `x * cos + cat(x2, x1) * sin_signed` in ONE
ttnn.generic_op (kernels in tt/cpp_rope_dec_kernels).

Stock, each of q and k takes six ops a layer step -- two slices, a concat (the rotate-half), two broadcast
multiplies and an add -- ~35 us a layer for 2 x 32 tiny tile rows. Here each core takes a tile row of q or k.
The cos / signed-sin rows come in as FULL tiles, every row the same bits (`full_rows`: 1.0 * row, built once a
decode step for all layers -- a per-core broadcast fill in the reader cost ~80 us); the rotate-half is a
whole-tile swap (half the head is a whole number of tiles); the arithmetic is binary_ng's float32 SFPU ops in its
order -- mul_binary_tile (x * cos), mul_binary_tile (rot * sin), add_binary_tile NearestEven -- every operand
unpacked straight to DEST. Bit for bit the stock RoPE.

On unless VOXTRAL_CPP_ROPE_DEC=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_rope_dec_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_ROPE_DEC", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_rope_dec: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _tile_rows(x):
    rows = 1
    for d in list(x.padded_shape)[:-1]:
        rows *= int(d)
    return rows // _TILE


def _plain(x):
    return x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT and not x.is_sharded()


_ONES: dict = {}


def full_rows(row):
    """A float32 `[..., 1, head]` row as a `[1, 1, 32, head]` tile row in L1 whose every row is the same bits
    (1.0 * row, binary_ng's row broadcast). The ones tile is made on the first (eager) call and kept."""
    device = row.device()
    head = int(row.shape[-1])
    ones = _ONES.get((id(device), head))
    if ones is None:
        ones = _ONES[(id(device), head)] = ttnn.full(
            ttnn.Shape([1, 1, _TILE, head]),
            1.0,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return ttnn.multiply(ones, row, memory_config=ttnn.L1_MEMORY_CONFIG)


def supports(xs, cos, sin) -> bool:
    """xs: float32 TILE interleaved tensors with the head as the last dim (an even number of tiles); cos / sin: the
    `full_rows` of the float32 rows they are multiplied by (broadcast over every other dim)."""
    try:
        if not enabled() or not xs:
            return False
        head = int(cos.shape[-1])
        acc = _accessor_args(xs[0])
        rows = 0
        for x in xs:
            if not _plain(x) or int(x.shape[-1]) != head or _accessor_args(x) != acc:
                return False
            rows += _tile_rows(x)
        for t in (cos, sin):
            if not _plain(t) or [int(d) for d in t.shape] != [1, 1, _TILE, head]:
                return False
        grid = xs[0].device().compute_with_storage_grid_size()
        return head % (2 * _TILE) == 0 and rows <= int(grid.x) * int(grid.y) * 8
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


def apply(xs, cos, sin):
    """`[x * cos + cat(x2, x1) * sin for x in xs]`, each float32 in its input's memory."""
    device = xs[0].device()
    dt = int(cos.shape[-1]) // _TILE
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    outs = [ttnn.allocate_tensor_on_device(x.shape, ttnn.float32, ttnn.TILE_LAYOUT, device, x.memory_config()) for x in xs]
    # (tensor index, first tile row, rows) per core: each tensor's rows split over its share of the cores
    total = sum(_tile_rows(x) for x in xs)
    per_core = -(-total // (gx * gy))
    work = []
    for i, x in enumerate(xs):
        rows = _tile_rows(x)
        for r0 in range(0, rows, per_core):
            work.append((i, r0, min(per_core, rows - r0)))
    ncores = len(work)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    ca, sa = cos.buffer_address(), sin.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (i, r0, n) in enumerate(work):
        cy, cx = divmod(c, gx)
        rr[cx][cy] = [xs[i].buffer_address(), ca, sa, r0, n]
        rc[cx][cy] = [n]
        rw[cx][cy] = [outs[i].buffer_address(), r0, n]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt] + _accessor_args(xs[0]) + _accessor_args(cos) + _accessor_args(sin),
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
            compile_time_args=[dt] + _accessor_args(outs[0]),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # binary_ng's float32 SFPU path: fp32 DEST, operands unpacked straight to DEST.
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
    ttnn.generic_op(list(xs) + [cos, sin] + outs, desc)
    return outs
