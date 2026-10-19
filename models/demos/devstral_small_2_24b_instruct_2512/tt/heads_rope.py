# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""C++ Metalium head split fused with the q / k RoPE (generic_op, kernels/heads_rope_*.cpp) for the vision attention.

From the fused QKV output qkv (B, 1, S, 3 * H * hd) bf16 tiles -- each row [q heads | k heads | v heads] -- and the
rotary tables cos / signed_sin (1, 1, S, hd) bf16 tiles in HF (rotate-half) order it writes q, k, v as (B, H, S, hd)
bf16 tiles with q = x * cos + rotate_half(x) * sin (k likewise; v copied): nlp_create_qkv_heads followed by the
rotate-half chain (x @ R, x * cos, rot * sin, add) in one launch, bit for bit. On whole tiles rotate_half swaps the
two head halves and negates the first, so no matmul is needed: the negation rides on signed_sin (sin with its first
half negated, exact), whose product with the partner tile has the operand magnitudes and sign of the chain's
rot * sin. The products and sum replay the ops the chain ran for bf16 operands: ttnn.multiply's accurate SFPU path
(mul_binary_tile in a 16-bit DEST, the product rounded to bf16 nearest-even) and ttnn.add's default FPU path
(add_tiles of the two bf16 product tiles in a 16-bit DEST).

Work units are (tile row, head) pairs, head-fastest, split evenly over the full compute grid; a core keeps the
cos / sin tiles of its current sequence tile, so the few rows it spans cost a handful of table reads."""
from __future__ import annotations

from pathlib import Path

import ttnn

_DIR = Path(__file__).resolve().parent / "kernels"
_READER = str(_DIR / "heads_rope_reader.cpp")
_WRITER = str(_DIR / "heads_rope_writer.cpp")
_COMPUTE = str(_DIR / "heads_rope_compute.cpp")

_TILE = 32
_TILE_BYTES = 2048  # bf16
CB_X, CB_COS, CB_SIN, CB_V, CB_OUT, CB_AB = 0, 1, 2, 3, 16, 24


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("heads_rope: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _cb(cores, index, tiles):
    return ttnn.CBDescriptor(
        total_size=tiles * _TILE_BYTES,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=_TILE_BYTES)
        ],
    )


def supports(qkv, cos, sin, heads) -> bool:
    shape = tuple(int(s) for s in qkv.shape)
    if len(shape) != 4 or shape[1] != 1 or shape[-1] % (3 * heads):
        return False
    hd = shape[-1] // (3 * heads)
    return (
        all(t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16 and not t.is_sharded() for t in (qkv, cos, sin))
        and hd % (2 * _TILE) == 0
        and shape[2] % _TILE == 0
        and tuple(int(s) for s in cos.shape)[-2:] == (shape[2], hd)
        and tuple(int(s) for s in sin.shape)[-2:] == (shape[2], hd)
    )


def apply(qkv, cos, signed_sin, heads, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """qkv (B, 1, S, 3 * heads * hd), cos / signed_sin (1, 1, S, hd): bf16 tiles, interleaved. Returns (q, k, v), each
    (B, heads, S, hd) bf16 tiles in memory_config, q and k rotated."""
    device = qkv.device()
    b, _, s, w = (int(d) for d in qkv.shape)
    hd = w // (3 * heads)
    ht, st = hd // _TILE, s // _TILE
    units = b * st * heads
    grid = device.compute_with_storage_grid_size()
    gx, ncores = int(grid.x), min(int(grid.x) * int(grid.y), units)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    out = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([b, heads, s, hd]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)
        for _ in range(3)
    ]
    q, k, v = out
    base, extra = divmod(units, ncores)
    rr, rw, rc = [], [], []
    u0 = 0
    for c in range(ncores):
        nu = base + (1 if c < extra else 0)
        cy, cx = divmod(c, gx)
        rr.append(((cx, cy), [qkv.buffer_address(), cos.buffer_address(), signed_sin.buffer_address(), u0, nu]))
        rw.append(((cx, cy), [q.buffer_address(), k.buffer_address(), v.buffer_address(), u0, nu]))
        rc.append(((cx, cy), [u0, nu]))
        u0 += nu
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False  # BinaryNg's 16-bit DEST for bf16 operands
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    ct = [heads, ht, st]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct + _accessor_args(qkv) + _accessor_args(cos) + _accessor_args(signed_sin),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct + _accessor_args(q) + _accessor_args(k) + _accessor_args(v),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rc,
            config=cfg,
        ),
    ]
    cbs = [
        _cb(cores, CB_X, 4 * ht),
        _cb(cores, CB_COS, 2 * ht),
        _cb(cores, CB_SIN, 2 * ht),
        _cb(cores, CB_V, 2 * ht),
        _cb(cores, CB_OUT, 4 * ht),
        _cb(cores, CB_AB, 4 * ht),
    ]
    ttnn.generic_op([qkv, cos, signed_sin, q, k, v], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return q, k, v
