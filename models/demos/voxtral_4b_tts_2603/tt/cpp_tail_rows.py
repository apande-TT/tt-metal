# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The compact prefill's k / v tails regrouped one sample a tile row, in ONE ttnn.generic_op for k and v
(kernels in tt/cpp_tail_rows_kernels).

`_tail_rows` in the prefill attention stubs turns the compact `[1, H, B * R, D]` bf16 k (and v) into
`[H * B, 1, R, D]` bf8_b TILE -- each (head, sample)'s R rows in their own tile row, zero-padded -- for the
paged cache fill. Stock, that is an untilize, a ROW_MAJOR view and a tilize-with-padding for each of the
two (4.7 + 10.4 us, launch-bound: 3 tile rows a core on 86 cores). Here each unit (pair, head, sample)
reads its R rows straight out of the TILE input (two 32-byte face-row segments a row and column tile)
into a zeroed tile row, and the packer writes it as bf8_b -- the same conversion the tilize ends with,
so the cache holds the same bits.

On unless VOXTRAL_CPP_TAIL_ROWS=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_tail_rows_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_TAIL_ROWS", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_tail_rows: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(ts, batch, real, dtype) -> bool:
    """Every `t` in `ts` a compact `[1, H, batch * real, D]` bf16 TILE tensor (same shape), real < 32."""
    try:
        shapes = {tuple(int(x) for x in t.shape) for t in ts}
        if len(shapes) != 1 or len(ts) not in (1, 2):
            return False
        one, h, rows, d = next(iter(shapes))
        return (
            enabled()
            and one == 1
            and rows == int(batch) * int(real)
            and 0 < int(real) <= _TILE
            and d % _TILE == 0
            and dtype in _BYTES
            and all(t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded() for t in ts)
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, fmt, tiles):
    page = _BYTES[fmt]
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def apply(ts, batch, real, dtype, memory_config=None):
    """`[ _tail_rows(t) for t in ts ]`: each `[H * batch, 1, real, D]` TILE in `dtype` (L1 unless `memory_config`)."""
    device = ts[0].device()
    _, h, rows, d = (int(x) for x in ts[0].shape)
    batch, real = int(batch), int(real)
    dt, hb, np_ = d // _TILE, h * batch, len(ts)
    rti = -(-rows // _TILE)
    units = np_ * hb
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    ys = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([hb, 1, real, d]), dtype, ttnn.TILE_LAYOUT, device, memory_config or ttnn.L1_MEMORY_CONFIG
        )
        for _ in ts
    ]
    ins = list(ts) + ([ts[0]] if np_ == 1 else [])
    outs = list(ys) + ([ys[0]] if np_ == 1 else [])
    ia = [t.buffer_address() for t in ins]
    oa = [y.buffer_address() for y in outs]
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [ia[0], ia[1], u0, nu]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [oa[0], oa[1], u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, real, batch, h, rti] + _accessor_args(ins[0]) + _accessor_args(ins[1]),
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
            compile_time_args=[dt, hb] + _accessor_args(outs[0]) + _accessor_args(outs[1]),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    # The stock bf8_b conversion: PRECISE bfp8 packing (what ttnn.typecast and the tilize use for a
    # bf8_b output). The descriptor's default approximate packing rounds the shared-exponent mantissas
    # differently, and the cache bits change (measured: PCC 0.985115 with it, bit-identical with this).
    # fp32 DEST with the input unpacked straight to DEST, as the tilize runs.
    cfg = kernels[1].config
    cfg.bfp8_pack_precise = True
    cfg.fp32_dest_acc_en = True
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, ttnn.bfloat16, 2 * dt),  # gathered rows (pad rows stay zero)
        _cb(cores, 1, ttnn.bfloat16, 2 * dt),  # reader scratch: the (at most two) input tile rows a sample spans
        _cb(cores, 16, dtype, 2 * dt),  # the tile row in the cache dtype
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = (
        hash(("voxtral_cpp_tail_rows", h, batch, real, dt, np_, str(dtype), tuple(ia), tuple(oa), str(memory_config)))
        & 0xFFFFFFFFFFFFFFFF
    )
    ttnn.generic_op(list(ts) + list(ys), desc)
    return ys
