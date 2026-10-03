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


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SHIFT_ADD", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_shift_add: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _cb(cores, index, tiles):
    return ttnn.CBDescriptor(
        total_size=tiles * _FP32_TILE,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=_FP32_TILE)
        ],
    )


def shift_add(y, batch, rows_per_sample, out_rows, taps, tap_cols, out_cols, memory_config=None):
    """`out[b, t, c] = sum_k Y[b * rows_per_sample + t + k, k * tap_cols + c]`, float32 `[B, 1, out_rows, out_cols]`.

    `y` is `[1, 1, batch * rows_per_sample, taps * tap_cols]` float32 TILE (rows_per_sample and tap_cols
    tile multiples)."""
    device = y.device()
    rpt, ct = rows_per_sample // _TILE, tap_cols // _TILE
    rt, ot = -(-out_rows // _TILE), -(-out_cols // _TILE)
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
            compile_time_args=[taps, rpt, rt, ot, ct, yct, yrt] + _accessor_args(y),
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
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, 2 * taps),  # shifted tap tiles
        _cb(cores, 1, 2),  # reader scratch: the two Y tile rows a shift spans
        _cb(cores, 16, 2),  # summed output tile
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = (
        hash(("voxtral_cpp_shift_add", batch, rpt, rt, ot, ct, taps, yct, yrt, ya, oa, str(memory_config)))
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


def conv_wide(x_rm, wide, out_len, compute_kernel_config, linear=None):
    """`out[b, t] = sum_k x_rm[b, 0, t + k] @ W_k` for t < out_len, float32 `[B, 1, out_len, C_out]` TILE.

    `x_rm` is `[B, 1, Lp, C_in]` float32 ROW_MAJOR (Lp >= out_len + K - 1); `wide` is `wide_weight(taps)`.
    `linear(x, w, **kw)` (default ttnn.linear) runs the one wide product."""
    w, k, cout = wide
    batch, _, lp, cin = (int(v) for v in x_rm.shape)
    tap_cols = -(-cout // _TILE) * _TILE
    rp = -(-lp // _TILE) * _TILE
    device = x_rm.device()
    if rp > lp:
        x_rm = ttnn.concat([x_rm, _zeros_rm(device, [batch, 1, rp - lp, cin], x_rm.dtype)], dim=2)
    x = ttnn.to_layout(ttnn.reshape(x_rm, [1, 1, batch * rp, cin]), ttnn.TILE_LAYOUT)
    y = (linear or ttnn.linear)(x, w, compute_kernel_config=compute_kernel_config, dtype=ttnn.float32)
    ttnn.deallocate(x)
    out = shift_add(y, batch, rp, out_len, k, tap_cols, cout)
    ttnn.deallocate(y)
    return out
