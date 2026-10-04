# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The codec attention's sliding window as a BANDED attention: one ttnn.generic_op (kernels in
tt/cpp_band_attn_kernels) in place of the full `[B, H, T, T]` chain -- scores bmm, scale multiply,
tt/cpp_softmax and the P@V bmm.

The codec's ALiBi mask blocks every key outside `[i - window, i]`, and the decoder's windows are 2 to
16 -- under a tile -- so each query tile row r only ever sees key tile rows r - 1 and r. The full
chain still computes all T / 32 key tiles, writes and re-reads the whole `[T, T]` float32 score
matrix three times, and at T = 256 more than 90% of it is exactly zero after the softmax.

Per unit (a (batch, head)'s query tile row) a core reads q, the band's k and v tile rows and the two
mask tiles (in place from the prebuilt mask), and replays the stock primitives in the stock order
(see the compute kernel). The skipped tiles' weights are exact zeros in the stock chain (their mask
is -1e9), so they change neither the row max, nor the sums (0 + x is exact), nor the context: the
output is bit-identical.

On unless VOXTRAL_CPP_BAND_ATTN=0.
"""

from __future__ import annotations

import os
import pathlib
import struct

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_band_attn_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64
# The CBs the compute kernel copy_tile()s from (unpacked straight to DEST, as the stock binary_ng and
# reduce SFPU paths do): raw scores, scale, mask, s, filled max, exp, filled sum. The matmul operands
# (q, k, v, P) go through SrcA / SrcB like the stock bmm's.
_UNPACK_TO_DEST = (3, 4, 5, 6, 8, 9, 11)


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_BAND_ATTN", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_band_attn: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(q, k, v, mask, window) -> bool:
    """q / k / v `[B, H, T, D]` float32 TILE (q / k may both be bf16; T tile-aligned, D <= 128); `mask` `[1, H, MM, MS]` float32 TILE
    with MM >= T and at least two column tiles; `window` < 32, so a row's keys never leave its band."""
    try:
        b, h, t, d = (int(x) for x in q.shape)
        mb, mh, mm, ms = (int(x) for x in mask.shape)
        return (
            enabled()
            and window is not None
            and 0 <= int(window) < _TILE
            and tuple(int(x) for x in k.shape) == (b, h, t, d)
            and tuple(int(x) for x in v.shape) == (b, h, t, d)
            and mb == 1
            and mh == h
            and mm >= t
            and ms >= 2 * _TILE
            and mm % _TILE == 0
            and ms % _TILE == 0
            and t % _TILE == 0
            and d % _TILE == 0
            and d // _TILE <= 4
            and q.dtype in (ttnn.float32, ttnn.bfloat16)
            and k.dtype == q.dtype
            and all(x.dtype == ttnn.float32 for x in (v, mask))
            and all(x.layout == ttnn.TILE_LAYOUT for x in (q, k, v, mask))
            and not any(x.is_sharded() for x in (q, k, v, mask))
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def supports_merged(q, k, v, mask, window, n_heads) -> bool:
    """The merged-head form: q / k / v `[B, 1, T, n_heads * D]` float32 TILE (head h = column tiles h * D/32 ..),
    the layout the q / k / v linears produce -- read in place, the context written back the same way. Any T:
    the padded rows of a partial last tile row hold keys past T, which the causal mask always blocks."""
    try:
        b, one, t, hd = (int(x) for x in q.shape)
        h = int(n_heads)
        mb, mh, mm, ms = (int(x) for x in mask.shape)
        d = hd // h if h else 0
        return (
            enabled()
            and one == 1
            and h > 0
            and hd % h == 0
            and window is not None
            and 0 <= int(window) < _TILE
            and tuple(int(x) for x in k.shape) == (b, 1, t, hd)
            and tuple(int(x) for x in v.shape) == (b, 1, t, hd)
            and mb == 1
            and mh == h
            and mm >= t
            and ms >= 2 * _TILE
            and mm % _TILE == 0
            and ms % _TILE == 0
            and t > 0
            and d % _TILE == 0
            and d // _TILE <= 4
            and q.dtype in (ttnn.float32, ttnn.bfloat16)
            and k.dtype == q.dtype
            and all(x.dtype == ttnn.float32 for x in (v, mask))
            and all(x.layout == ttnn.TILE_LAYOUT for x in (q, k, v, mask))
            and not any(x.is_sharded() for x in (q, k, v, mask))
        )
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


def _bf16_cb(cores, index, tiles):
    return ttnn.CBDescriptor(
        total_size=tiles * 2048,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)],
    )


def apply(q, k, v, mask, scale, memory_config=None, merged_heads=None, dtype=ttnn.float32, real_rows=None):
    """`softmax(q @ k^T * scale + mask) @ v`, float32 `[B, H, T, D]` (DRAM unless `memory_config`). With
    `merged_heads`, q / k / v and the result are the merged `[B, 1, T, H * D]` (see `supports_merged`). `dtype`:
    the context's dtype (float32 or bfloat16; the softmax and both products stay float32 in DEST either way)."""
    device = q.device()
    merged = bool(merged_heads)
    if merged:
        b, _, t, hd = (int(x) for x in q.shape)
        h = int(merged_heads)
        d = hd // h
    else:
        b, h, t, d = (int(x) for x in q.shape)
    mmt, mst = int(mask.shape[2]) // _TILE, int(mask.shape[3]) // _TILE
    # A partial last tile row (T not tile-aligned) is one more row of units: its padding rows compute
    # garbage nobody reads, and every key past T is above the diagonal, so the causal mask blocks it.
    rt, dt = -(-t // _TILE), d // _TILE
    # a short sequence (every real query row in a tile's top 16) runs the SFPU steps half-tile
    half = int(real_rows is not None and rt == 1 and int(real_rows) <= 16)
    units = b * h * rt
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([b, 1, t, h * d] if merged else [b, h, t, d]),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    # dtype: bf16 q / k (the fused codec qk-norm's output) are read as the score product's operands as they are --
    # the kernel switches the unpacker to their format around that product only.
    qk_narrow = q.dtype == ttnn.bfloat16
    qk_bytes = 2048 if qk_narrow else _FP32_TILE
    qa, ka, va, ma, ya = (x.buffer_address() for x in (q, k, v, mask, y))
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [qa, ka, va, ma, u0, nu, scale_bits]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [ya, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, rt, h, mmt, mst, int(merged), qk_bytes]
            + _accessor_args(q)
            + _accessor_args(k)
            + _accessor_args(v)
            + _accessor_args(mask),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, int(dtype != ttnn.float32), half, int(qk_narrow)],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[dt, h, rt, int(merged)] + _accessor_args(y),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # HiFi2 for the two products (the stock codec bmm's fidelity); the SFPU softmax steps are exact
    # (non-approx) float32 functions with every operand unpacked straight to DEST, as in tt/cpp_softmax.
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    for i in _UNPACK_TO_DEST:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _bf16_cb(cores, 0, 2 * dt) if qk_narrow else _cb(cores, 0, 2 * dt),  # q tile row
        _bf16_cb(cores, 1, 4 * dt) if qk_narrow else _cb(cores, 1, 4 * dt),  # k band (2 tile rows)
        _cb(cores, 2, 4 * dt),  # v band (2 tile rows)
        _cb(cores, 3, 2),  # raw scores
        _cb(cores, 4, 1),  # scale tile
        _cb(cores, 5, 4),  # mask band
        _cb(cores, 6, 2),  # s = scores * scale + mask
        _cb(cores, 7, 1),  # row max (column 0)
        _cb(cores, 8, 1),  # row max, column-filled
        _cb(cores, 9, 2),  # e = exp(s - max)
        _cb(cores, 10, 1),  # row sum (column 0)
        _cb(cores, 11, 1),  # row sum, column-filled
        _cb(cores, 12, 2),  # weights P
        _cb(cores, 16, 2 * dt) if dtype == ttnn.float32 else _bf16_cb(cores, 16, 2 * dt),  # context tile row
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    key = ("voxtral_cpp_band_attn", b, h, rt, dt, mmt, mst, merged, half, qk_narrow, scale_bits, qa, ka, va, ma, ya)
    desc.custom_program_hash = hash(key + (str(memory_config), str(dtype))) & 0xFFFFFFFFFFFFFFFF
    ttnn.generic_op([q, k, v, mask, y], desc)
    return y
