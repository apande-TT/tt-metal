# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The codec's kernel-4 / stride-2 transposed convolution as ONE wide matmul and ONE interleaving shift-add
(ttnn.generic_op, kernels in tt/cpp_upsample2_kernels).

With the sequence channels-last, `out[2m + j] = x[m] @ W_j + x[m - 1] @ W_{2+j}` (j = 0, 1; x[-1] and x[L]
read as zero). The stubs ran that as four tap products, a zero-row concat per delayed tap (a TILE concat of
a non-tile-aligned piece: untilize both pieces, row-major concat, re-tilize), two adds, a channel concat of
even | odd and a `[B, L+1, 2C] -> [B, 2L+2, C]` reshape that is a full TILE relayout -- ~2.1 ms for the
128-frame upsample of 16 samples, 460 us of it the products.

Here `Y = X @ [W_0 | W_1 | W_2 | W_3]` is one product, and one generic_op writes the interleaved, shifted
sum straight into the `[B, 1, out_len, C]` TILE output: per output tile, the reader gathers the "now" rows
(Y_j[m]) and the writer the "delayed" rows (Y_{2+j}[m - 1]) in output order, and the compute kernel adds
them with the stock float32 SFPU add -- now + delayed, the same add the stub's `ttnn.add` runs.

On unless VOXTRAL_CPP_UPSAMPLE2=0.
"""

from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_upsample2_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_UPSAMPLE2", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_upsample2: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _cb(cores, index, tiles, fmt=ttnn.float32):
    page = _FP32_TILE if fmt == ttnn.float32 else _FP32_TILE // 2
    return ttnn.CBDescriptor(
        total_size=tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def wide_weight(taps):
    """The four `[C_in, C_out]` taps side by side: `[C_in, 4 * C_out]` (C_out a tile multiple)."""
    return ttnn.concat(list(taps), dim=-1)


def supports(x_cl, taps) -> bool:
    """`x_cl` `[B, 1, L, C_in]` float32 TILE (any L: each sample's rows stay tile-padded), four taps whose C_out
    is a tile multiple."""
    try:
        return (
            enabled()
            and len(taps) == 4
            and x_cl.layout == ttnn.TILE_LAYOUT
            and x_cl.dtype == ttnn.float32
            and len(x_cl.shape) == 4
            and int(x_cl.shape[1]) == 1
            and int(x_cl.shape[-2]) >= 1
            and int(taps[0].shape[-1]) % _TILE == 0
            and not x_cl.is_sharded()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def interleave_add(y, batch, length, cout, out_len, memory_config=None):
    """`out[b, 2m + j] = Y[b, m, j-block] + Y[b, m - 1, (2 + j)-block]`, float32 `[B, 1, out_len, C_out]`.

    `y` is `[B, 1, L, 4 * C_out]` float32 or bfloat16 TILE (each sample's L rows tile-padded); rows m outside
    [0, L) read as zero; out_len <= 2L + 2, and the output's own padding rows are written as zeros. A bfloat16 `y`
    is gathered as bfloat16 and widened exactly into the float32 DEST the add runs in."""
    device = y.device()
    ydt = ttnn.float32 if y.dtype == ttnn.float32 else ttnn.bfloat16
    es = 4 if ydt == ttnn.float32 else 2
    rty, ct = -(-length // _TILE), cout // _TILE
    orows = -(-out_len // _TILE)
    units = batch * orows * ct
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, 1, out_len, cout]),
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
        rw[cx][cy] = [ya, oa, u0, nu]
        u0 += nu
    dims = [length, rty, out_len, orows, ct, es]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=dims + _accessor_args(y),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=dims + _accessor_args(y) + _accessor_args(out),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # The stock float32 binary_ng add: fp32 DEST, operands unpacked straight to DEST.
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    if ydt == ttnn.float32:
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, 2, ydt),  # gathered "now" tiles
        _cb(cores, 1, 2, ydt),  # gathered "delayed" tiles
        _cb(cores, 2, 2),  # reader scratch: two half-tiles + a zero row
        _cb(cores, 3, 2),  # writer scratch: two half-tiles, the previous rows, a zero row
        _cb(cores, 16, 2),  # summed output tiles
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = (
        hash(("voxtral_cpp_upsample2", batch, length, rty, orows, ct, out_len, es, ya, oa, str(memory_config)))
        & 0xFFFFFFFFFFFFFFFF
    )
    ttnn.generic_op([y, out], desc)
    return out


def apply(x_cl, wide, cout, out_len, compute_kernel_config, linear=None, y_dtype=ttnn.bfloat16):
    """The transposed conv of `x_cl` `[B, 1, L, C_in]` (float32 TILE) as `[B, 1, out_len, C_out]` float32 TILE:
    ONE product against `wide` (`wide_weight(taps)`) and ONE interleaving shift-add. `out_len` 2L + 2 is the
    bare conv, 2L its causal trim. The product runs on the 4-D rows (a 2-D multicast config folds the batch
    into M over each sample's tile-padded rows, so an unaligned L needs no relayout); `linear(x, w, **kw)`
    (default ttnn.linear) runs it."""
    batch, _, length, _ = (int(v) for v in x_cl.shape)
    # dtype: the wide product Y written bfloat16 by default (`y_dtype`) -- half its write and the shift-add's
    # reads; the two terms are still added in float32 (the gathered tiles widen exactly into the fp32 DEST).
    # shard: a small wide product (<= 16 MB) lands in L1 for the shift-add that reads it.
    kw = {}
    if tile_rows(x_cl) * int(wide.shape[-1]) * (4 if y_dtype == ttnn.float32 else 2) <= (16 << 20):
        kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
    y = (linear or ttnn.linear)(x_cl, wide, compute_kernel_config=compute_kernel_config, dtype=y_dtype, **kw)
    out = interleave_add(y, batch, length, cout, out_len)
    ttnn.deallocate(y)
    return out


def tile_rows(x):
    """The rows a batch-folded product of `x` runs over: the leading dims times the tile-padded row count."""
    shape = [int(v) for v in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return lead * (-(-shape[-2] // _TILE) * _TILE)
