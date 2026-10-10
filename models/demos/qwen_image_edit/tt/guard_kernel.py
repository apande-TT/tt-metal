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
_SUM_COMPUTE = f"{_DIR}/sum_parts_compute.cpp"
_SPLIT_COMPUTE = f"{_DIR}/split_limbs_compute.cpp"
_TWO_OUT_WRITER = f"{_DIR}/two_out_writer.cpp"
_WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
_IN_CBS = (0, 1, 2, 3, 4, 5)
_OUT_CB = 16  # the outputs' CBs: c_16, then c_17, c_18
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


def _program(ins, y, n_tiles, compute=_COMPUTE, mods=(), offs=(), compute_args=()):
    """Up to six float32 tile operands -> y (one tensor, or two / three written through c_16..c_18), tile by tile;
    operand k read at page (i % mods[k] when nonzero, else i) + offs[k] (missing mods / offs are 0)."""
    mods = list(mods) + [0] * (len(_IN_CBS) - len(mods))
    offs = list(offs) + [0] * (len(_IN_CBS) - len(offs))
    n_in = len(ins)
    ys = list(y) if isinstance(y, (list, tuple)) else [y]
    grid = ys[0].device().compute_with_storage_grid_size()
    work = _cores(grid, n_tiles)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb_desc(cb, dtype):
        page = _FP32_TILE_BYTES if dtype == ttnn.float32 else _FP32_TILE_BYTES // 2
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb, data_format=dtype, page_size=page)],
        )

    cbs = [cb_desc(cb, ttnn.float32) for cb in _IN_CBS[:n_in]]
    cbs += [cb_desc(_OUT_CB + k, t.dtype) for k, t in enumerate(ys)]

    pad = len(_IN_CBS) - n_in  # the reader's unused operand slots repeat operand 0 (never read)
    reader_cta = []
    for t in list(ins) + [ins[0]] * pad:
        reader_cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader_cta.append(n_in)
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in ins] + [0] * pad
    for c, start, count in work:
        # runtime layout: addr0..4, count, start, mod0..4, off0..4, then addr5, mod5, off5
        reader_rt[c.x][c.y] = addrs[:5] + [count, start] + mods[:5] + offs[:5] + [addrs[5], mods[5], offs[5]]
        compute_rt[c.x][c.y] = [count]
        if len(ys) == 1:
            writer_rt[c.x][c.y] = [ys[0].buffer_address(), count, start]
        else:  # the multi-output writer takes three address slots
            writer_rt[c.x][c.y] = [t.buffer_address() for t in ys] + [0] * (3 - len(ys)) + [count, start]

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
            compile_time_args=list(compute_args),
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER if len(ys) == 1 else _TWO_OUT_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=(
                [_OUT_CB] + ttnn.TensorAccessorArgs(ys[0]).get_compile_time_args()
                if len(ys) == 1
                else [len(ys)]
                + [a for t in ys + ys[:1] * (3 - len(ys)) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
            ),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _plain(t, s):
    return list(t.shape) == s and t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded()


def guard_tail(ex, dn, nn, tol, lo_fn, lo2_fn=None):
    """where(|ex - dn| <= tol, ex, median(ex, dn, -nn)) + lo_fn() [+ lo2_fn()] for same-shape interleaved float32
    tile tensors [..., M, N] (M and N tile-aligned). Consumes (deallocates) ex, dn, nn, tol and the lo_fn() /
    lo2_fn() tensors. Returns None for a shape it does not take, before calling lo_fn."""
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
    if lo2_fn is not None:
        lo2 = lo2_fn()
        if not _plain(lo2, s):  # not expected, as for lo
            ttnn.deallocate(lo2)
            ttnn.deallocate(lo)
            return None
        ins.append(lo2)
    y = ttnn.empty_like(ex)
    ttnn.generic_op(ins + [y], _program(ins, y, n_tiles, compute_args=[int(lo2_fn is not None)]))
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


def sum_parts(g, n):
    """((g[0] + g[1]) + g[2]) + ... for an interleaved float32 tile tensor g = [n, ...] (the TP all-reduce's
    gathered partials), in one pass: each slab is read at its page offset, with no slice copies, and added
    in the order of the ttnn.add chain (bit-identical). Returns g[0]'s shape; None for an n or a g it does not
    take. g is left as it is."""
    s = list(g.shape)
    if not 2 <= n <= len(_IN_CBS) or s[0] != n or len(s) < 3:
        return None
    if g.dtype != ttnn.float32 or g.layout != ttnn.TILE_LAYOUT or g.is_sharded():
        return None
    ps = list(g.padded_shape)
    pages = 1
    for d in ps[1:-2]:
        pages *= d
    pages *= (ps[-2] // TILE) * (ps[-1] // TILE)
    y = ttnn.empty(
        s[1:], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=g.device(), memory_config=g.memory_config()
    )
    ins = [g] * n
    program = _program(
        ins, y, pages, compute=_SUM_COMPUTE, offs=[k * pages for k in range(n)] + [0] * (5 - n), compute_args=[n]
    )
    return ttnn.generic_op([g, y], program)


def sum_list(ts):
    """((ts[0] + ts[1]) + ts[2]) + ... for 2..6 same-shape interleaved float32 tile tensors (an exact-lane sum's
    products), in one pass, bit-identical to the ttnn.add chain. Consumes (deallocates) ts. None for a list it
    does not take (ts left as they are)."""
    if not 2 <= len(ts) <= len(_IN_CBS):
        return None
    s = list(ts[0].shape)
    if len(s) < 2 or not all(_plain(t, s) for t in ts):
        return None
    ps = list(ts[0].padded_shape)
    pages = 1
    for d in ps[:-2]:
        pages *= d
    pages *= (ps[-2] // TILE) * (ps[-1] // TILE)
    y = ttnn.empty_like(ts[0])
    ttnn.generic_op(list(ts) + [y], _program(ts, y, pages, compute=_SUM_COMPUTE, compute_args=[len(ts)]))
    for t in ts:
        ttnn.deallocate(t)
    return y


def split_limbs(x, limbs=2):
    """split_bf16's `limbs` (2 or 3) bf16 limbs of an interleaved float32 tile tensor in one pass -- hi = bf16(x),
    then the bf16 of each remainder -- with the same SFPU typecast / subtract; or None for a tensor it does
    not take."""
    if limbs not in (2, 3):
        return None
    if x.dtype != ttnn.float32 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or len(x.shape) < 2:
        return None
    ps = list(x.padded_shape)
    pages = 1
    for d in ps[:-2]:
        pages *= d
    pages *= (ps[-2] // TILE) * (ps[-1] // TILE)
    outs = [
        ttnn.empty(
            list(x.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=x.device(),
            memory_config=x.memory_config(),
        )
        for _ in range(limbs)
    ]
    ttnn.generic_op([x] + outs, _program([x], outs, pages, compute=_SPLIT_COMPUTE, compute_args=[limbs]))
    return tuple(outs)
