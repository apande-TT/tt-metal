# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A stride-1 convolution over row-major padded rows as ONE wide matmul and ONE shift-add (ttnn.generic_op,
kernels in tt/cpp_shift_add_kernels).

The codec convolutions run as shifted matmuls, `out[t] = sum_k x[t + k] @ W_k`: per tap, a row-offset cut of
the padded rows (a full copy of the [B, L, C_in] activation), its tilize, a [B*L, C_in] x [C_in, C_out]
matmul, and an add into the running sum -- K copies and K tilizes of the WIDE side (C_in = 1024 for the
k = 7 output projection, whose C_out is 240).

Here the padded rows are tilized ONCE (each sample's rows padded to a tile multiple first, so the batch
folds into M as a view), multiplied ONCE against the K taps laid side by side (each tap's columns padded to
a tile multiple: Y = X @ [W_0 | W_1 | ...]), and the row shift moves to the NARROW side: unit (sample,
output tile row, output column tile) reads, for each tap k, the two tile rows of Y its rows t + k span,
gathers the shifted tile locally and sums the K of them in tap order with the stock float32 SFPU add.

On unless VOXTRAL_CPP_SHIFT_ADD=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_shift_add_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64
# shard: the wide product Y lands in L1 (interleaved over the grid) while it fits this many bytes (the
# output_proj caller runs the product at HiFi2).
_Y_L1_BYTES = 16 * 1024 * 1024


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SHIFT_ADD", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_shift_add: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _cb(cores, index, tiles, fmt=ttnn.float32):
    tile_bytes = _FP32_TILE if fmt == ttnn.float32 else _FP32_TILE // 2
    return ttnn.CBDescriptor(
        total_size=tiles * tile_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=tile_bytes)],
    )


_MODES = {None: 0, "reflect": 1, "replicate": 2}


def shift_add(
    y, batch, rows_per_sample, out_rows, taps, tap_cols, out_cols, pad_mode=None, pad_front=0, real_rows=0, memory_config=None
):
    """`out[b, t, c] = sum_k Y[b * rows_per_sample + src(t + k), k * tap_cols + c]`, float32 `[B, 1, out_rows, out_cols]`.

    `y` is `[1, 1, batch * rows_per_sample, taps * tap_cols]` float32 or bfloat16 TILE (rows_per_sample and
    tap_cols tile multiples; a bf16 Y is gathered as bf16 tiles and summed in fp32 DEST). With no `pad_mode`, Y holds the padded rows (src(p) = p); with `pad_mode` 'reflect' or
    'replicate', Y holds the `real_rows` unpadded rows and padded row p resolves to the row its padding copies
    (`pad_front` rows in front, the rest behind)."""
    device = y.device()
    rpt, ct = rows_per_sample // _TILE, tap_cols // _TILE
    rt, ot = -(-out_rows // _TILE), -(-out_cols // _TILE)
    y_fmt = ttnn.bfloat16 if y.dtype == ttnn.bfloat16 else ttnn.float32
    yrt, yct = int(y.shape[-2]) // _TILE, int(y.shape[-1]) // _TILE
    units = batch * rt * ot
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, 1, out_rows, out_cols]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
    ya, oa = y.buffer_address(), out.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [ya, u0, nu]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [oa, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[taps, rpt, rt, ot, ct, yct, _MODES[pad_mode], int(pad_front), int(real_rows)]
            + [2 if y_fmt == ttnn.bfloat16 else 4]
            + _accessor_args(y),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[taps],
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
    # The stock float32 binary_ng add: fp32 DEST, operands unpacked straight to DEST.
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    if y_fmt == ttnn.float32:
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, 2 * taps, y_fmt),  # shifted tap tiles
        _cb(cores, 1, 2 * taps, y_fmt),  # reader scratch: the two Y tile rows each tap's shift spans
        _cb(cores, 16, 2),  # summed output tile
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = (
        hash(
            (
                "voxtral_cpp_shift_add",
                batch,
                rpt,
                rt,
                ot,
                ct,
                taps,
                yct,
                yrt,
                pad_mode,
                pad_front,
                real_rows,
                str(y_fmt),
                ya,
                oa,
                str(memory_config),
            )
        )
        & 0xFFFFFFFFFFFFFFFF
    )
    ttnn.generic_op([y, out], desc)
    return out


_ZEROS_RM = {}


def _zeros_rm(device, shape, dtype):
    """A persistent ROW_MAJOR zero block (created once per shape: a trace cannot replay the host write)."""
    key = (id(device), tuple(shape), str(dtype))
    z = _ZEROS_RM.get(key)
    if z is None:
        z = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        _ZEROS_RM[key] = z
    return z


def supports(x_rm, taps, stride=1, dilation=1) -> bool:
    try:
        return (
            enabled()
            and int(stride) == 1
            and int(dilation) == 1
            and len(taps) > 1
            and x_rm.layout == ttnn.ROW_MAJOR_LAYOUT
            and x_rm.dtype == ttnn.float32
            and len(x_rm.shape) == 4
            and int(x_rm.shape[1]) == 1
            and all(t.layout == ttnn.TILE_LAYOUT and t.dtype == taps[0].dtype and len(t.shape) == 2 for t in taps)
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def wide_weight(taps):
    """The K `[C_in, C_out]` taps side by side, each padded to a tile multiple of columns:
    `([C_in, K * tap_cols] TILE, K, C_out)`."""
    cout = int(taps[0].shape[-1])
    tap_cols = -(-cout // _TILE) * _TILE
    blocks = taps if tap_cols == cout else [ttnn.pad(t, [(0, 0), (0, tap_cols - cout)], 0.0) for t in taps]
    return ttnn.concat(blocks, dim=-1), len(taps), cout


def _y_mem(x, w, y_dtype):
    """shard: Y in L1 while it fits `_Y_L1_BYTES` (the product writes it and the shift-add gathers it there)."""
    es = 4 if y_dtype == ttnn.float32 else 2
    nbytes = int(x.shape[-2]) * int(w.shape[-1]) * es
    return {"memory_config": ttnn.L1_MEMORY_CONFIG} if nbytes <= _Y_L1_BYTES else {}


def conv_wide_unpadded(
    x_cl, wide, pad_mode, pad_front, out_len, compute_kernel_config, linear=None, y_dtype=ttnn.bfloat16
):
    """The same conv straight from the UNPADDED channels-last TILE rows `x_cl` `[B, 1, L, C_in]` (L a tile
    multiple): Y = X @ W_wide on the L real rows, and the reflect / replicate padding resolved in the shift-add
    (a padded row's product is the product of the row it copies) -- no untilize, no edge-row concat, no tilize.
    dtype: the wide product Y is written `y_dtype` (bf16 by default: half the bytes of its write and of the
    shift-add's reads; the K taps still sum in fp32 DEST)."""
    w, k, cout = wide
    batch, _, length, cin = (int(v) for v in x_cl.shape)
    tap_cols = -(-cout // _TILE) * _TILE
    x = ttnn.reshape(x_cl, [1, 1, batch * length, cin])
    y = (linear or ttnn.linear)(x, w, compute_kernel_config=compute_kernel_config, dtype=y_dtype, **_y_mem(x, w, y_dtype))
    out = shift_add(y, batch, length, out_len, k, tap_cols, cout, pad_mode=pad_mode, pad_front=pad_front, real_rows=length)
    ttnn.deallocate(y)
    return out


def supports_unpadded(x_cl, pad_mode) -> bool:
    try:
        return (
            enabled()
            and pad_mode in ("reflect", "replicate")
            and x_cl.layout == ttnn.TILE_LAYOUT
            and x_cl.dtype == ttnn.float32
            and len(x_cl.shape) == 4
            and int(x_cl.shape[1]) == 1
            and int(x_cl.shape[-2]) % _TILE == 0
            and int(x_cl.shape[-2]) >= _TILE
            and not x_cl.is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def conv_wide(x_rm, wide, out_len, compute_kernel_config, linear=None, y_dtype=ttnn.bfloat16):
    """`out[b, t] = sum_k x_rm[b, 0, t + k] @ W_k` for t < out_len, float32 `[B, 1, out_len, C_out]` TILE.

    `x_rm` is `[B, 1, Lp, C_in]` float32 ROW_MAJOR (Lp >= out_len + K - 1); `wide` is `wide_weight(taps)`.
    `linear(x, w, **kw)` (default ttnn.linear) runs the one wide product, written `y_dtype`."""
    w, k, cout = wide
    batch, _, lp, cin = (int(v) for v in x_rm.shape)
    tap_cols = -(-cout // _TILE) * _TILE
    rp = -(-lp // _TILE) * _TILE
    device = x_rm.device()
    if rp > lp:
        x_rm = ttnn.concat([x_rm, _zeros_rm(device, [batch, 1, rp - lp, cin], x_rm.dtype)], dim=2)
    x = ttnn.to_layout(ttnn.reshape(x_rm, [1, 1, batch * rp, cin]), ttnn.TILE_LAYOUT)
    y = (linear or ttnn.linear)(x, w, compute_kernel_config=compute_kernel_config, dtype=y_dtype, **_y_mem(x, w, y_dtype))
    ttnn.deallocate(x)
    out = shift_add(y, batch, rp, out_len, k, tap_cols, cout)
    ttnn.deallocate(y)
    return out
