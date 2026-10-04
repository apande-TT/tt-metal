# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the codec LayerNorm: a block's residual add and the FFN RMS norm after it as ONE
ttnn.generic_op (kernels in tt/cpp_addnorm_kernels).

Stock, a codec block runs `h = h + r` (binary_ng: reads h and r, writes the float32 residual stream), then the
FFN norm (`ttnn.rms_norm`, which reads the stream back) and a typecast of its float32 output to bf16. Here each
unit -- one 32-row tile row -- reads h and r once: the compute adds them with binary_ng's own float32 SFPU add
(so the new residual stream is the stock one), keeps the row in L1, folds the row sums of its squares in
float32, takes rsqrt(sum / dim + eps), and writes the normalised row straight in the consumer's dtype. No gamma
(the codec FFN norm's gamma is folded into w1 / w3).

Taken where the stream is long enough for the saved pass to matter (>= 2048 rows); on unless
VOXTRAL_CPP_ADDNORM=0.
"""
from __future__ import annotations

import os
import pathlib
import struct

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_addnorm_kernels"
_READER = str(_DIR / "reader.cpp")
_COMPUTE = str(_DIR / "compute.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_FP32_TILE = 4096
_NUM_CBS = 64
_MIN_ROWS = 2048
# The CBs the compute kernel copy_tile()s from: h, r, the kept row s, the column-filled scale.
_UNPACK_TO_DEST = (0, 1, 2, 4)
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_ADDNORM", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_addnorm: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _rows(x):
    rows = 1
    for d in list(x.shape)[:-1]:
        rows *= int(d)
    return rows


def supports(h, r, dtype=ttnn.bfloat16) -> bool:
    """h / r float32 TILE interleaved, same unpadded shape `[..., rows, dim]`, rows >= 2048."""
    try:
        shape = [int(d) for d in h.shape]
        return (
            enabled()
            and [int(d) for d in r.shape] == shape
            and [int(d) for d in h.padded_shape] == shape
            and [int(d) for d in r.padded_shape] == shape
            and h.dtype == ttnn.float32
            and r.dtype == ttnn.float32
            and dtype in _TILE_BYTES
            and h.layout == ttnn.TILE_LAYOUT
            and r.layout == ttnn.TILE_LAYOUT
            and not h.is_sharded()
            and not r.is_sharded()
            and shape[-1] % _TILE == 0
            and _rows(h) % _TILE == 0
            and _rows(h) >= _MIN_ROWS
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, tiles, dtype=ttnn.float32):
    size = _TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=tiles * size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
    )


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def apply(h, r, eps, dtype=ttnn.bfloat16, memory_config=None):
    """`(h + r, rms_norm(h + r) as dtype)`: the new float32 residual stream (in h's memory) and its normed rows
    (in `memory_config`, else h's)."""
    device = h.device()
    shape = [int(d) for d in h.shape]
    rows, dim = _rows(h), shape[-1]
    wt = dim // _TILE
    units = rows // _TILE
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    hn = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device, h.memory_config())
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, memory_config or h.memory_config()
    )
    ha, ra, hna, ya = h.buffer_address(), r.buffer_address(), hn.buffer_address(), y.buffer_address()
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = [ha, ra, u0, nu]
        rc[cx][cy] = [nu]
        rw[cx][cy] = [hna, ya, u0, nu]
        u0 += nu
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt] + _accessor_args(h) + _accessor_args(r),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, _bits(1.0 / dim), _bits(eps), int(dtype != ttnn.float32)],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt] + _accessor_args(hn) + _accessor_args(y),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cfg = kernels[1].config
    # float32 SFPU throughout (binary_ng's float32 path): HiFi4, fp32 DEST, exact functions, half-sync DEST,
    # every operand the compute copies unpacked straight to DEST.
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CBS
    for i in _UNPACK_TO_DEST:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    cbs = [
        _cb(cores, 0, 2),  # h, a tile at a time
        _cb(cores, 1, 2),  # r, a tile at a time
        _cb(cores, 2, wt),  # s = h + r, the whole row (kept for the stats and the scale passes)
        _cb(cores, 3, 1),  # inv = rsqrt(mean + eps) (column 0)
        _cb(cores, 4, 1),  # inv, column-filled by the writer
        _cb(cores, 16, 2),  # the new residual stream h + r, to the writer
        _cb(cores, 17, 2, dtype),  # y, to the writer
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    key = ("voxtral_cpp_addnorm", units, wt, _bits(eps), str(dtype), ha, ra, hna, ya, str(memory_config))
    desc.custom_program_hash = hash(key) & 0xFFFFFFFFFFFFFFFF
    ttnn.generic_op([h, r, hn, y], desc)
    return hn, y
