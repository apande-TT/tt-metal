# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The guarded exact-lane tail of the precise linears as ONE Metalium kernel (ttnn.generic_op).

The text-encoder ports' EXACT_MODE "guarded" (models/demos/qwen_image_edit_text_encoder/_stubs/attention.py,
_guarded) finishes every precise linear / matmul with

    y = where(|ex - dn| <= tol, ex, median(ex, dn, -nn)) + lo

spelled as ~9 float32 ttnn passes over the [rows, N] output (subtract, compare, 4x min/max, where, add),
each a full DRAM round trip. This kernel reads ex, dn, nn, tol, lo once and writes y once, with the same
SFPU calls those ttnn ops make on float32 operands unpacked straight to DST, so y is bit-identical.

Kernels: kernels/guard_tail_reader.cpp (five operands -> c_0..c_4), kernels/guard_tail_compute.cpp, and
ttnn's stock interleaved unary writer (c_16 -> y).
"""

from __future__ import annotations

import ttnn

TILE = 32
_DIR = "models/demos/qwen_image_edit/tt/kernels"
_READER = f"{_DIR}/guard_tail_reader.cpp"
_COMPUTE = f"{_DIR}/guard_tail_compute.cpp"
_VOTE_COMPUTE = f"{_DIR}/vote_tail_compute.cpp"
_WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
_IN_CBS = (0, 1, 2, 3, 4)
_OUT_CB = 16
_FP32_TILE_BYTES = 4096


def _cores(grid, n_tiles):
    """Row-major split of n_tiles over the compute grid: [(core, start, count)], each core a contiguous run."""
    n_cores = min(n_tiles, grid.x * grid.y)
    per, extra = divmod(n_tiles, n_cores)
    out, start = [], 0
    for i in range(n_cores):
        count = per + (1 if i < extra else 0)
        out.append((ttnn.CoreCoord(i % grid.x, i // grid.x), start, count))
        start += count
    return out


def _program(ins, y, n_tiles, compute=_COMPUTE, mods=(0, 0, 0, 0, 0)):
    """Five float32 tile operands -> y, tile by tile; operand k read at page i % mods[k] when nonzero."""
    grid = y.device().compute_with_storage_grid_size()
    work = _cores(grid, n_tiles)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
            ],
        )
        for cb in _IN_CBS + (_OUT_CB,)
    ]

    reader_cta = []
    for t in ins:
        reader_cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in ins]
    for c, start, count in work:
        reader_rt[c.x][c.y] = addrs + [count, start] + list(mods)
        compute_rt[c.x][c.y] = [count]
        writer_rt[c.x][c.y] = [y.buffer_address(), count, start]

    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64  # covers Wormhole (32) and Blackhole (64) CB counts
    for cb in _IN_CBS:
        unpack[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32  # float32 operands exact in DST, not TF32
    compute_cfg.unpack_to_dest_mode = unpack

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_cta,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=compute,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[_OUT_CB] + ttnn.TensorAccessorArgs(y).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _plain(t, s):
    return list(t.shape) == s and t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded()


def guard_tail(ex, dn, nn, tol, lo_fn):
    """where(|ex - dn| <= tol, ex, median(ex, dn, -nn)) + lo_fn() for same-shape interleaved float32 tile
    tensors [..., M, N] (M and N tile-aligned). Consumes (deallocates) ex, dn, nn, tol and lo_fn()'s tensor.
    Returns None for a shape it does not take, before calling lo_fn."""
    s = list(ex.shape)
    if len(s) < 2 or s[-2] % TILE or s[-1] % TILE or not all(_plain(t, s) for t in (ex, dn, nn, tol)):
        return None
    lo = lo_fn()
    if not _plain(lo, s):  # not expected: the trailing product has the leading one's shape and dtype
        ttnn.deallocate(lo)
        return None
    n_tiles = 1
    for d in s[:-2]:
        n_tiles *= d
    n_tiles *= (s[-2] // TILE) * (s[-1] // TILE)
    ins = [ex, dn, nn, tol, lo]
    y = ttnn.empty_like(ex)
    ttnn.generic_op(ins + [y], _program(ins, y, n_tiles))
    # the inputs are dead after the tail: free them now, not at the caller's return (the text encoder and
    # the denoiser run at the edge of DRAM, where the later free fragments it)
    for t in ins:
        ttnn.deallocate(t)
    return y


def vote_tail(acc, bn, f0, e_neg, tol):
    """where(|e_x - e_neg| <= tol, e_x, median(e_x, f0, e_neg)), e_x = acc - bn: precise_affine's exact-mode
    vote tail (models/tt_dit/pipelines/qwen_image_edit_vae/_stubs/_resident.py, VOTE_TAIL), ~8 float32 ttnn
    passes over the conv output, here one, bit-identical. Same-shape interleaved float32 tile tensors [..., M, C]
    (M and C tile-aligned); bn that shape or a (1, 1, C) row broadcast over the rows. Consumes (deallocates)
    every operand. Returns None for a shape it does not take."""
    s = list(acc.shape)
    if len(s) < 2 or s[-2] % TILE or s[-1] % TILE or not all(_plain(t, s) for t in (acc, f0, e_neg, tol)):
        return None
    c = s[-1]
    if _plain(bn, s):
        row, mods = None, (0, 0, 0, 0, 0)
    elif list(bn.shape) == [1, 1, c] and bn.dtype == ttnn.float32 and bn.layout == ttnn.TILE_LAYOUT:
        # the bias row as a full (32, C) block: tile column j of every output row reads its page j
        row = ttnn.repeat(bn, (1, TILE, 1))
        mods = (0, c // TILE, 0, 0, 0)
    else:
        return None
    n_tiles = 1
    for d in s[:-2]:
        n_tiles *= d
    n_tiles *= (s[-2] // TILE) * (c // TILE)
    ins = [acc, bn if row is None else row, f0, e_neg, tol]
    y = ttnn.empty_like(acc)
    ttnn.generic_op(ins + [y], _program(ins, y, n_tiles, compute=_VOTE_COMPUTE, mods=mods))
    for t in ins + ([] if row is None else [bn]):
        ttnn.deallocate(t)
    return y
