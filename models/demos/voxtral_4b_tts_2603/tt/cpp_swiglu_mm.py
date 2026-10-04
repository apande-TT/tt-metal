# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the short prefill SwiGLU: `minimal_matmul(x, w_gu, fuse_swiglu=True)` re-laid over
the whole grid, bit for bit (kernels in tt/cpp_swiglu_mm_kernels).

minimal_matmul spreads its weight over grid.x = 11 column readers (M on the grid rows, N on the columns), so at
the text prefix's 5 tile rows it runs on 55 cores and streams the 32 MB bf4_b gate / up weight through 11 of
them (~126 GB/s, 252 us). Here each of ncores cores owns ALL rows of W fused (gate, up) column tiles and
streams just those columns of every K row, so every core reads weight.

The arithmetic is minimal_matmul's, tile for tile (see compute.cpp): the same K blocks (its K_block_size)
accumulate in a fresh 16-bit DEST at LoFi, are packed into a Float16_b accumulator (packer L1 accumulation from
the second block), and the same copy / silu (approximate) / SFPU multiply / bf16 pack produces each output tile.
The ORDER of the K blocks is minimal_matmul's too: it walks each core's N blocks in a snake (K forward, then
backward to reuse the resident in0 block, then forward ...), so an output tile in an odd N block of its
minimal_matmul core accumulates its blocks last to first -- and bf16 accumulation is order-sensitive. Each core
here owns tiles of ONE such N block and streams its K blocks in that block's order.

Taken for <= 8 tile rows (the text prefix); on unless VOXTRAL_CPP_SWIGLU_MM=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_swiglu_mm_kernels"
_READER_W = str(_DIR / "reader_w.cpp")
_READER_X = str(_DIR / "reader_x_writer.cpp")
_COMPUTE = str(_DIR / "compute.cpp")

_TILE = 32
_MAX_MT = 8
_DEPTH = 4
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SWIGLU_MM", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_swiglu_mm: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _rows(x):
    rows = 1
    for d in list(x.shape)[:-1]:
        rows *= int(d)
    return rows


def _reversed(n, pairs, n_block, grid_x):
    """Whether minimal_matmul (fuse_swiglu, N_block_size `n_block`, `grid_x` column cores) accumulates output
    tile n's K blocks last to first: pairs are padded to a multiple of grid_x and split evenly over the columns,
    each column walks its N blocks (n_block // 2 output tiles each) alternating K forward / backward."""
    per = -(-pairs // grid_x)
    return ((n % per) // (n_block // 2)) % 2 == 1


def _split(device, pairs, n_block, grid_x):
    """The most cores (<= the grid) that split the output column tiles evenly, with <= 8 fused tiles a core and
    every core's tiles in one K order."""
    grid = device.compute_with_storage_grid_size()
    cap = int(grid.x) * int(grid.y)
    for c in range(min(cap, pairs), 0, -1):
        pw = pairs // c
        if pairs % c or 2 * pw > 8:
            continue
        if all(
            len({_reversed(n, pairs, n_block, grid_x) for n in range(i * pw, (i + 1) * pw)}) == 1 for i in range(c)
        ):
            return c
    return None


def supports(x, w, n_block, grid_x) -> bool:
    """x: bf8_b / bf16 TILE interleaved `[..., rows, K]`, rows a multiple of 32 up to 256; w: the bf4_b / bf8_b
    tile-pair-interleaved `[K, 2N]` minimal_matmul fuse_swiglu weight, interleaved."""
    try:
        if not enabled():
            return False
        k = int(x.shape[-1])
        rows = _rows(x)
        wk, wn = int(w.shape[-2]), int(w.shape[-1])
        return (
            x.layout == ttnn.TILE_LAYOUT
            and w.layout == ttnn.TILE_LAYOUT
            and not x.is_sharded()
            and not w.is_sharded()
            and x.dtype in (ttnn.bfloat8_b, ttnn.bfloat16)
            and w.dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
            and [int(d) for d in x.padded_shape][-1] == k
            and rows % _TILE == 0
            and 1 <= rows // _TILE <= _MAX_MT
            and wk == k
            and k % _TILE == 0
            and wn % (2 * _TILE) == 0
            and n_block % 2 == 0
            and _split(x.device(), wn // (2 * _TILE), n_block, grid_x) is not None
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, dtype, tiles):
    size = _TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=tiles * size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
    )


def apply(x, w, kb, n_block, grid_x, memory_config=None):
    """`silu(x @ gate) * (x @ up)` bf16 `[..., rows, N]` (L1 interleaved unless `memory_config`), exactly as
    minimal_matmul(fuse_swiglu=True, K_block_size=kb, N_block_size=n_block, grid_x column cores, LoFi, 16-bit
    DEST, packer L1 acc) computes it."""
    device = x.device()
    shape = [int(d) for d in x.shape]
    rows, k = _rows(x), shape[-1]
    mt, kt = rows // _TILE, k // _TILE
    ntf = int(w.shape[-1]) // _TILE
    nt = ntf // 2
    if kt % kb:
        raise RuntimeError(f"cpp_swiglu_mm: K tiles {kt} not a multiple of the K block {kb}")
    nb = kt // kb
    ncores = _split(device, nt, n_block, grid_x)
    pw = nt // ncores
    wf = 2 * pw
    sw = wf
    grid = device.compute_with_storage_grid_size()
    gx = int(grid.x)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape[:-1] + [nt * _TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        memory_config or ttnn.L1_MEMORY_CONFIG,
    )
    xa, wa, ya = x.buffer_address(), w.buffer_address(), y.buffer_address()
    rw, rx, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        rev = int(_reversed(c * pw, nt, n_block, grid_x))
        rw[cx][cy] = [wa, c * wf, rev]
        rx[cx][cy] = [xa, ya, c * pw, rev]
        rc[cx][cy] = []
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER_W,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[kb, nb, wf, ntf] + _accessor_args(w),
            runtime_args=rw,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_READER_X,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[mt, kb, nb, pw, kt, nt] + _accessor_args(x) + _accessor_args(y),
            runtime_args=rx,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[mt, kb, nb, wf, sw],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]
    cfg = kernels[2].config
    # minimal_matmul's prefix config: LoFi, 16-bit half-sync DEST, approximate SFPU (the default), packer L1 acc
    # (switched per block in the kernel).
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = True
    cfg.dst_full_sync_en = False
    cbs = [
        _cb(cores, 0, x.dtype, 2 * mt * kb),
        _cb(cores, 1, w.dtype, _DEPTH * kb * wf),
        _cb(cores, 16, ttnn.bfloat16, mt * pw),
        _cb(cores, 24, ttnn.bfloat16, mt * wf),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op([x, w, y], desc)
    return y
