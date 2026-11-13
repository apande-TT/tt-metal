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
    tile tensors [..., M, N] (a ragged M or N -- the denoiser's 622 keys -- runs over the padded tiles: the same
    shape pads alike, and the padding's values are never read). Consumes (deallocates) ex, dn, nn, tol and the
    lo_fn() / lo2_fn() tensors. Returns None for a shape it does not take, before calling lo_fn."""
    s = list(ex.shape)
    if len(s) < 2 or not all(_plain(t, s) for t in (ex, dn, nn, tol)):
        return None
    lo = lo_fn()
    if not _plain(lo, s):  # not expected: the trailing product has the leading one's shape and dtype
        ttnn.deallocate(lo)
        return None
    ps = list(ex.padded_shape)
    n_tiles = 1
    for d in ps[:-2]:
        n_tiles *= d
    n_tiles *= (ps[-2] // TILE) * (ps[-1] // TILE)
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


_NORMS_READER = f"{_DIR}/guard_norms_reader.cpp"
_NORMS_COMPUTE = f"{_DIR}/guard_norms_compute.cpp"


def _batch(t):
    v = 1
    for d in list(t.padded_shape)[:-2]:
        v *= d
    return v


# guard_tail_norms' output in L1 (interleaved) when it is at most this many bytes: the softmax reads the guarded
# scores straight back. 0 = always DRAM.
GUARD_OUT_L1_MAX = 16 * 2**20


def guard_tail_norms(ex, dn, nn, a_s, b_norm, lo_fn):
    """guard_tail(ex, dn, nn, a_s * b_norm, lo_fn) with the tolerance's outer product formed tile by tile inside the
    kernel (a_s: the slack-scaled [.., M, 1] row norms, b_norm the [.., 1, N] column norms; each batch-matched to ex
    or a single batch broadcast over it): no output-sized tol tensor written and read back. Consumes ex, dn, nn and
    lo_fn()'s product, not a_s / b_norm. Returns None (lo_fn not called) for a shape it does not take."""
    s = list(ex.shape)
    # (nn None: no negated-input product -- where the guard trips the tail takes dn)
    if len(s) < 2 or not all(_plain(t, s) for t in ((ex, dn) if nn is None else (ex, dn, nn))):
        return None
    for t in (a_s, b_norm):
        if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded():
            return None
    ps = list(ex.padded_shape)
    mt, nt = ps[-2] // TILE, ps[-1] // TILE
    pa, pb = list(a_s.padded_shape), list(b_norm.padded_shape)
    if list(a_s.shape)[-2:] != [s[-2], 1] or list(b_norm.shape)[-2:] != [1, s[-1]]:
        return None
    if pa[-2] // TILE != mt or pa[-1] != TILE or pb[-1] // TILE != nt or pb[-2] != TILE:
        return None
    nb = _batch(ex)
    if _batch(a_s) not in (1, nb) or _batch(b_norm) not in (1, nb):
        return None
    a_bstride = mt if _batch(a_s) == nb and nb > 1 else 0
    b_bstride = nt if _batch(b_norm) == nb and nb > 1 else 0
    lo = lo_fn()
    if not _plain(lo, s):
        ttnn.deallocate(lo)
        return None
    n_tiles = nb * mt * nt
    y = None
    if GUARD_OUT_L1_MAX and n_tiles * _FP32_TILE_BYTES <= GUARD_OUT_L1_MAX:
        # (the guarded scores stay in L1 for the softmax that reads them next; DRAM when L1 has no room)
        try:
            y = ttnn.empty(
                s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=ex.device(), memory_config=ttnn.L1_MEMORY_CONFIG
            )
        except Exception:  # noqa: BLE001
            y = None
    y = ttnn.empty_like(ex) if y is None else y
    grid = y.device().compute_with_storage_grid_size()
    work = _cores(grid, n_tiles)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb_desc(cb):
        return ttnn.CBDescriptor(
            total_size=2 * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
            ],
        )

    cbs = [cb_desc(cb) for cb in (0, 1, 2, 3, 4, 5, _OUT_CB)]
    ins = [ex, dn, dn if nn is None else nn, lo, a_s, b_norm]
    reader_cta = []
    for t in ins:
        reader_cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader_cta.append(int(nn is not None))
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in ins]
    for c, start, count in work:
        reader_rt[c.x][c.y] = addrs + [count, start, mt, nt, a_bstride, b_bstride]
        compute_rt[c.x][c.y] = [count, start, nt]
        writer_rt[c.x][c.y] = [y.buffer_address(), count, start]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for cb in _IN_CBS:
        unpack[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_NORMS_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_cta,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_NORMS_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[int(nn is not None)],
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
    ttnn.generic_op(ins + [y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    for t in (ex, dn, lo) if nn is None else (ex, dn, nn, lo):
        ttnn.deallocate(t)
    return y


_ES_READER = f"{_DIR}/exp_split_reader.cpp"
_ES_COMPUTE = f"{_DIR}/exp_split_compute.cpp"


def exp_split(s, m, approx=True):
    """(e, hi, lo) = (exp(s - m), bf16(e), bf16(e - hi)) in one pass for a float32 interleaved tile tensor s [.., M, N]
    and its row maxima m [.., M, 1] (batch-matched): the softmax's fused subtract + EXP post-activation and the P @ V
    limb split, without writing e and reading it back for the split. None for a shape it does not take."""
    sh = list(s.shape)
    if not _plain(s, sh) or len(sh) < 2:
        return None
    if m.dtype != ttnn.float32 or m.layout != ttnn.TILE_LAYOUT or m.is_sharded() or list(m.shape) != sh[:-1] + [1]:
        return None
    ps = list(s.padded_shape)
    nt = ps[-1] // TILE
    rows = 1
    for d in ps[:-1]:
        rows *= d
    n_tiles = (rows // TILE) * nt
    dev = s.device()
    e = ttnn.empty_like(s)
    hi, lo = (ttnn.empty(sh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(2))
    work = _cores(dev.compute_with_storage_grid_size(), n_tiles)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb_desc(cb, dtype):
        page = _FP32_TILE_BYTES if dtype == ttnn.float32 else _FP32_TILE_BYTES // 2
        return ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb, data_format=dtype, page_size=page)],
        )

    cbs = [cb_desc(0, ttnn.float32), cb_desc(1, ttnn.float32), cb_desc(16, ttnn.float32)]
    cbs += [cb_desc(17, ttnn.bfloat16), cb_desc(18, ttnn.bfloat16)]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [s.buffer_address(), m.buffer_address(), count, start, nt]
        compute_rt[c.x][c.y] = [count, start, nt]
        writer_rt[c.x][c.y] = [e.buffer_address(), hi.buffer_address(), lo.buffer_address(), count, start]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for cb in (0, 1):
        unpack[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_ES_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[a for t in (s, m) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_ES_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[int(bool(approx))],
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_TWO_OUT_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[3]
            + [a for t in (e, hi, lo) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([s, m, e, hi, lo], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return e, hi, lo


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
    return ttnn.generic_op([g, y], _sum_blk_program(g, y, n, pages))


_SUM_BLK_READER = f"{_DIR}/sum_blk_reader.cpp"
_SUM_BLK_COMPUTE = f"{_DIR}/sum_blk_compute.cpp"
_SUM_BLK = 4  # tiles per slab per barrier / per DST acquire (2 x 4 float32 tiles fill the 8-tile DST)


def _sum_blk_program(g, y, n, pages):
    """SUM_PARTS' program, blocked: the reader moves _SUM_BLK tiles of every slab per barrier and the compute sums
    them under one DST acquire (each init once per block), the same adds in the same order per tile."""
    grid = y.device().compute_with_storage_grid_size()
    work = _cores(grid, pages)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb_desc(cb):
        return ttnn.CBDescriptor(
            total_size=2 * _SUM_BLK * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
            ],
        )

    cbs = [cb_desc(k) for k in range(n)] + [cb_desc(_OUT_CB)]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [g.buffer_address(), count, start, pages]
        compute_rt[c.x][c.y] = [count]
        writer_rt[c.x][c.y] = [y.buffer_address(), count, start]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for k in range(n):
        unpack[k] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_SUM_BLK_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(g).get_compile_time_args() + [n, _SUM_BLK],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_SUM_BLK_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[n, _SUM_BLK],
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


_LC_READER = f"{_DIR}/lane_cols_reader.cpp"
_LC_COMPUTE = f"{_DIR}/lane_cols_compute.cpp"


def lane_cols_sum(g, lanes):
    """sum_r g[..., r * n : (r + 1) * n] (left to right, the ttnn.add chain's float32 values) for an interleaved
    float32 tile tensor g [..., M, lanes * n] -- the N-stacked exact-lane product x @ [w_0 | ... | w_{L-1}] -- in one
    pass with no slice copies. Consumes (deallocates) g. None for a g it does not take (g left as it is)."""
    s, ps = list(g.shape), list(g.padded_shape)
    if len(s) < 2 or g.dtype != ttnn.float32 or g.layout != ttnn.TILE_LAYOUT or g.is_sharded():
        return None
    if not 2 <= lanes <= 8 or s[-1] % (lanes * TILE) or ps[-1] != s[-1]:
        return None
    n = s[-1] // lanes
    nt = n // TILE
    n_tiles = nt * (ps[-2] // TILE)
    for d in ps[:-2]:
        n_tiles *= d
    y = ttnn.empty(
        s[:-1] + [n], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=g.device(), memory_config=g.memory_config()
    )
    work = _cores(g.device().compute_with_storage_grid_size(), n_tiles)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb(index, pages):
        return ttnn.CBDescriptor(
            total_size=pages * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
            ],
        )

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [g.buffer_address(), count, start]
        compute_rt[c.x][c.y] = [count]
        writer_rt[c.x][c.y] = [y.buffer_address(), count, start]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_LC_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(g).get_compile_time_args() + [lanes, nt],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_LC_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[lanes],
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
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(0, 2 * lanes), cb(_OUT_CB, 2)])
    ttnn.generic_op([g, y], program)
    ttnn.deallocate(g)
    return y


_MM_READER = f"{_DIR}/masked_mm_reader.cpp"
_MM_COMPUTE = f"{_DIR}/masked_mm_compute.cpp"
_MM_WRITER = f"{_DIR}/masked_mm_writer.cpp"
_COL_MASKS = {}
_MM_L1_BYTES = 1000 * 1024


def _col_masks(device, lanes=8):
    """The column-lane mask tiles ([lanes x 32, 32] bf16; tile r keeps the columns j % lanes == r: lane r of an
    input's K)."""
    key = (id(device), lanes)
    if key not in _COL_MASKS:
        vals = [1.0 if j % lanes == r else 0.0 for r in range(lanes) for _ in range(TILE) for j in range(TILE)]
        m = ttnn.Tensor(vals, [lanes * TILE, TILE], ttnn.float32, ttnn.TILE_LAYOUT, device)
        _COL_MASKS[key] = ttnn.typecast(m, ttnn.bfloat16)
    return _COL_MASKS[key]


def _col_masks_ones(device, lanes):
    """_col_masks plus one all-ones tile (index lanes): the lane slot of masked_lane_linear's folded trailing limb."""
    key = (id(device), lanes, "ones")
    if key not in _COL_MASKS:
        vals = [1.0 if j % lanes == r else 0.0 for r in range(lanes) for _ in range(TILE) for j in range(TILE)]
        vals += [1.0] * (TILE * TILE)
        m = ttnn.Tensor(vals, [(lanes + 1) * TILE, TILE], ttnn.float32, ttnn.TILE_LAYOUT, device)
        _COL_MASKS[key] = ttnn.typecast(m, ttnn.bfloat16)
    return _COL_MASKS[key]


def masked_lane_linear(x, w, fidelity=ttnn.MathFidelity.HiFi3, lanes=8, x_lo=None):
    """sum_r lane_r(x) @ w in float32 (lane r of x keeping its K columns k % 8 == r: the exact-lane sum of the
    weight-side lanes) for a bf16 interleaved tile input [..., M, K] and bf16 weight [K, N] (M, K, N tile-aligned),
    with no 8x copy of x: each core masks its own x tiles into their lanes in L1 and reads w once. None for a shape
    it does not take."""
    s, ws = list(x.shape), list(w.shape)
    if len(s) < 2 or len(ws) != 2 or s[-1] != ws[0] or s[-2] % TILE or s[-1] % TILE or ws[1] % TILE:
        return None
    if x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded():
        return None
    if w.layout != ttnn.TILE_LAYOUT or w.is_sharded() or list(x.padded_shape) != s:
        return None
    if x_lo is not None and (
        list(x_lo.padded_shape) != s
        or x_lo.dtype != ttnn.bfloat16
        or x_lo.layout != ttnn.TILE_LAYOUT
        or x_lo.is_sharded()
    ):
        return None
    extra = 0 if x_lo is None else 1
    rows = 1
    for d in s[:-1]:
        rows *= d
    mt, kt, nt = rows // TILE, s[-1] // TILE, ws[1] // TILE
    dev = x.device()
    work = _cores(dev.compute_with_storage_grid_size(), mt)
    rmax = max(c for _, _, c in work)
    page = _FP32_TILE_BYTES // 2
    ln = lanes + extra
    fixed = (2 * rmax * (1 + extra) + ln + ln * rmax) * page  # x (+ lo) (double), masks, lanes
    nc = next(
        (
            d
            for d in range(nt, 0, -1)
            if nt % d == 0 and fixed + d * (rmax * _FP32_TILE_BYTES + 2 * page) <= _MM_L1_BYTES
        ),
        0,
    )
    w_bufs = 2
    # a single-buffered weight row when it cuts the column chunks: each chunk re-masks every x tile into its lanes
    nc1 = next(
        (d for d in range(nt, 0, -1) if nt % d == 0 and fixed + d * (rmax * _FP32_TILE_BYTES + page) <= _MM_L1_BYTES),
        0,
    )
    if nc and nc1 and nt // nc1 < nt // nc:
        nc, w_bufs = nc1, 1
    if not nc:
        return None
    sbw = 2 if nc % 2 == 0 else 1  # (2-row subblocks: 2 x sbw <= the 4 float32 DST tiles)
    masks = _col_masks(dev, lanes) if not extra else _col_masks_ones(dev, lanes)
    y = ttnn.empty(s[:-1] + [ws[1]], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb(index, pages, dtype):
        size = _FP32_TILE_BYTES if dtype == ttnn.float32 else page
        return ttnn.CBDescriptor(
            total_size=pages * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )

    cbs = [
        cb(0, 2 * rmax * (1 + extra), ttnn.bfloat16),
        cb(1, w_bufs * nc, ttnn.bfloat16),
        cb(2, ln, ttnn.bfloat16),
        cb(3, ln * rmax, ttnn.bfloat16),
        cb(_OUT_CB, rmax * nc, ttnn.float32),
    ]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [x.buffer_address(), w.buffer_address(), masks.buffer_address(), start, count] + (
            [x_lo.buffer_address()] if extra else []
        )
        compute_rt[c.x][c.y] = [count]
        writer_rt[c.x][c.y] = [y.buffer_address(), start, count]
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_MM_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args()
            + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(masks).get_compile_time_args()
            + [kt, nt, nc, lanes, extra]
            + ttnn.TensorAccessorArgs(x_lo if extra else x).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_MM_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[kt, nt, nc, sbw, rmax, lanes, extra],
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_MM_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args() + [nt, nc],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op(
        [x, w, masks, y] + ([x_lo] if extra else []), ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    )
    return y


_MM2_READER = f"{_DIR}/masked_mm2_reader.cpp"
_MM2_WRITER = f"{_DIR}/masked_mm2_writer.cpp"


def masked_lane_linear2(x, w, w2, fidelity=ttnn.MathFidelity.HiFi3, lanes=8):
    """(masked_lane_linear(x, w), masked_lane_linear(x, w2)) in ONE pass for two bf16 weights [K, N], [K, N2] sharing
    the input: x is read and masked into its lanes once per column chunk of both, not of each. None for shapes the
    single-weight kernel does not take."""
    s, ws, ws2 = list(x.shape), list(w.shape), list(w2.shape)
    if len(ws2) != 2 or ws2[0] != ws[0] or ws2[1] % TILE or w2.dtype != ttnn.bfloat16 or w2.is_sharded():
        return None
    if len(s) < 2 or len(ws) != 2 or s[-1] != ws[0] or s[-2] % TILE or s[-1] % TILE or ws[1] % TILE:
        return None
    if x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded():
        return None
    if w.layout != ttnn.TILE_LAYOUT or w.is_sharded() or list(x.padded_shape) != s:
        return None
    rows = 1
    for d in s[:-1]:
        rows *= d
    mt, kt, nt1 = rows // TILE, s[-1] // TILE, ws[1] // TILE
    nt = nt1 + ws2[1] // TILE
    dev = x.device()
    work = _cores(dev.compute_with_storage_grid_size(), mt)
    rmax = max(c for _, _, c in work)
    page = _FP32_TILE_BYTES // 2
    fixed = (2 * rmax + lanes + lanes * rmax) * page  # x (double), masks, lanes
    nc = next(
        (
            d
            for d in range(nt, 0, -1)
            if nt % d == 0 and fixed + d * (rmax * _FP32_TILE_BYTES + 2 * page) <= _MM_L1_BYTES
        ),
        0,
    )
    w_bufs = 2
    # a single-buffered weight row when it cuts the column chunks: each chunk re-masks every x tile into its lanes
    nc1 = next(
        (d for d in range(nt, 0, -1) if nt % d == 0 and fixed + d * (rmax * _FP32_TILE_BYTES + page) <= _MM_L1_BYTES),
        0,
    )
    if nc and nc1 and nt // nc1 < nt // nc:
        nc, w_bufs = nc1, 1
    if not nc:
        return None
    sbw = 2 if nc % 2 == 0 else 1  # (2-row subblocks: 2 x sbw <= the 4 float32 DST tiles)
    masks = _col_masks(dev, lanes)
    y = ttnn.empty(s[:-1] + [ws[1]], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    y2 = ttnn.empty(s[:-1] + [ws2[1]], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb(index, pages, dtype):
        size = _FP32_TILE_BYTES if dtype == ttnn.float32 else page
        return ttnn.CBDescriptor(
            total_size=pages * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )

    cbs = [
        cb(0, 2 * rmax, ttnn.bfloat16),
        cb(1, w_bufs * nc, ttnn.bfloat16),
        cb(2, lanes, ttnn.bfloat16),
        cb(3, lanes * rmax, ttnn.bfloat16),
        cb(_OUT_CB, rmax * nc, ttnn.float32),
    ]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [
            x.buffer_address(),
            w.buffer_address(),
            masks.buffer_address(),
            start,
            count,
            w2.buffer_address(),
        ]
        compute_rt[c.x][c.y] = [count]
        writer_rt[c.x][c.y] = [y.buffer_address(), start, count, y2.buffer_address()]
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_MM2_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args()
            + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(masks).get_compile_time_args()
            + ttnn.TensorAccessorArgs(w2).get_compile_time_args()
            + [kt, nt, nc, nt1, lanes],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_MM_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[kt, nt, nc, sbw, rmax, lanes, 0],
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_MM2_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args()
            + [nt, nc, nt1]
            + ttnn.TensorAccessorArgs(y2).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([x, w, masks, w2, y, y2], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y, y2


_BMM_M_READER = f"{_DIR}/masked_bmm_reader.cpp"
_BMM_M_COMPUTE = f"{_DIR}/masked_bmm_compute.cpp"
_BMM_M_WRITER = f"{_DIR}/masked_bmm_writer.cpp"


def masked_lane_bmm(a, b, b_rep=1, fidelity=ttnn.MathFidelity.HiFi4):
    """sum_r lane_r(a) @ b in float32 for a batched bf16 a [E, 1, M, K] and b [E / b_rep, 1, K, N] (entry e of a
    against b's entry e // b_rep; M, K, N tile-aligned): the 8 exact a-side lane products and their sum in ONE pass,
    each core masking its a tiles into their lanes in L1 and reading each b entry once per a row. None for a shape it
    does not take."""
    sa, sb = list(a.shape), list(b.shape)
    if len(sa) != 4 or len(sb) != 4 or sa[1] != 1 or sb[1] != 1 or sa[0] != sb[0] * b_rep or sa[3] != sb[2]:
        return None
    if sa[2] % TILE or sa[3] % TILE or sb[3] % TILE or list(a.padded_shape) != sa or list(b.padded_shape) != sb:
        return None
    for t in (a, b):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded():
            return None
    mte, kt, nt = sa[2] // TILE, sa[3] // TILE, sb[3] // TILE
    rows = sa[0] * mte
    dev = a.device()
    work = _cores(dev.compute_with_storage_grid_size(), rows)
    sbw = 4 if nt % 4 == 0 else (2 if nt % 2 == 0 else 1)
    masks = _col_masks(dev)
    y = ttnn.empty([sa[0], 1, sa[2], sb[3]], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    page = _FP32_TILE_BYTES // 2

    def cb(index, pages, dtype):
        size = _FP32_TILE_BYTES if dtype == ttnn.float32 else page
        return ttnn.CBDescriptor(
            total_size=pages * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )

    cbs = [
        cb(0, 2, ttnn.bfloat16),
        cb(1, 2 * nt, ttnn.bfloat16),
        cb(2, 8, ttnn.bfloat16),
        cb(3, 8, ttnn.bfloat16),
        cb(_OUT_CB, nt, ttnn.float32),
    ]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [a.buffer_address(), b.buffer_address(), masks.buffer_address(), start, count]
        compute_rt[c.x][c.y] = [count]
        writer_rt[c.x][c.y] = [y.buffer_address(), start, count]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_BMM_M_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(a).get_compile_time_args()
            + ttnn.TensorAccessorArgs(b).get_compile_time_args()
            + ttnn.TensorAccessorArgs(masks).get_compile_time_args()
            + [kt, nt, mte, b_rep],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_BMM_M_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[kt, nt, sbw],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_BMM_M_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args() + [nt],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([a, b, masks, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y


_WL_READER = f"{_DIR}/weight_lanes_reader.cpp"
_WL_COMPUTE = f"{_DIR}/weight_lanes_compute.cpp"
_EIGHT_OUT_WRITER = f"{_DIR}/eight_out_writer.cpp"
_ROW_MASKS = {}


def _row_masks(device):
    """The 8 row-lane mask tiles ([8 x 32, 32] bf16; tile r keeps the rows i % 8 == r), built once per device
    with ttnn.Tensor (no host op inside an observed forward)."""
    if id(device) not in _ROW_MASKS:
        vals = [1.0 if i % 8 == r else 0.0 for r in range(8) for i in range(TILE) for _ in range(TILE)]
        m = ttnn.Tensor(vals, [8 * TILE, TILE], ttnn.float32, ttnn.TILE_LAYOUT, device)
        _ROW_MASKS[id(device)] = ttnn.typecast(m, ttnn.bfloat16)
    return _ROW_MASKS[id(device)]


def weight_lanes(w):
    """The 8 exact-lane copies of a bf16 weight [K, N] (copy r keeps the K rows k % 8 == r, the strided lanes of
    models/demos/qwen_image_edit_text_encoder/_stubs/attention.py) in one pass that reads w once; or None for a
    weight it does not take."""
    s = list(w.shape)
    if len(s) != 2 or s[0] % TILE or s[1] % TILE:
        return None
    if w.dtype != ttnn.bfloat16 or w.layout != ttnn.TILE_LAYOUT or w.is_sharded():
        return None
    masks = _row_masks(w.device())
    n_tiles = (s[0] // TILE) * (s[1] // TILE)
    outs = [ttnn.empty_like(w) for _ in range(8)]
    grid = w.device().compute_with_storage_grid_size()
    work = _cores(grid, n_tiles)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    page = _FP32_TILE_BYTES // 2

    def cb(index, pages):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page)],
        )

    cbs = [cb(0, 2), cb(1, 8)] + [cb(_OUT_CB + r, 2) for r in range(8)]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [w.buffer_address(), masks.buffer_address(), count, start]
        compute_rt[c.x][c.y] = [count]
        writer_rt[c.x][c.y] = [t.buffer_address() for t in outs] + [count, start]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_WL_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(masks).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WL_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_EIGHT_OUT_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[a for t in outs for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([w, masks] + outs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return outs


def split_limbs(x, limbs=2, memory_config=None):
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
            memory_config=x.memory_config() if memory_config is None else memory_config,
        )
        for _ in range(limbs)
    ]
    ttnn.generic_op([x] + outs, _program([x], outs, pages, compute=_SPLIT_COMPUTE, compute_args=[limbs]))
    return tuple(outs)


_SW_READER = f"{_DIR}/small_w_mm_reader.cpp"
_SW_COMPUTE = f"{_DIR}/small_w_mm_compute.cpp"
_SW_MAX_W_TILES = 96  # the whole weight in each core's L1: 96 bf16 tiles (192 KB) beside the row and output blocks


def _rows_mm(a, w, rows, kt, nt, rows_per_entry, w_entry_pages, transpose_b, fidelity, w_rep=1):
    """y [rows * 32, nt * 32] float32 = each tile row of a (kt tiles, bf16) @ its entry's weight block (bf16) in w:
    the tile rows split over the whole grid, each entry's block read into L1 once per core."""
    chunk = max(d for d in range(1, 9) if nt % d == 0)  # output tiles per DST pass (8 float32 tiles, full sync)
    y = ttnn.empty((rows * TILE, nt * TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=a.device())
    grid = a.device().compute_with_storage_grid_size()
    work = _cores(grid, rows)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    bf_page = _FP32_TILE_BYTES // 2
    w_blocks = 2 if 2 * kt * nt <= _SW_MAX_W_TILES else 1  # the next entry's block loads during this one's rows

    def cb(index, pages, dtype, page):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    cbs = [
        cb(0, 2 * kt, ttnn.bfloat16, bf_page),
        cb(1, w_blocks * kt * nt, ttnn.bfloat16, bf_page),
        cb(_OUT_CB, 2 * chunk, ttnn.float32, _FP32_TILE_BYTES),
    ]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        reader_rt[c.x][c.y] = [a.buffer_address(), w.buffer_address(), start, count]
        compute_rt[c.x][c.y] = [start, count]
        writer_rt[c.x][c.y] = [y.buffer_address(), count * nt, start * nt]
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, dst_full_sync_en=True)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_SW_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(a).get_compile_time_args()
            + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + [kt, nt, rows_per_entry, w_entry_pages, w_rep],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_SW_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[kt, nt, chunk, rows_per_entry, int(transpose_b)],
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
    ttnn.generic_op([a, w, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y


def _plain_bf16(*ts):
    return all(t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded() for t in ts)


def small_weight_linear(p, w, fidelity=ttnn.MathFidelity.HiFi4):
    """p [..., M, K] bf16 @ w [K, N] bf16 -> float32 [..., M, N] for a SMALL weight (the VAE's 1x1 conv shortcuts,
    192 -> 384): every core reads the whole weight into L1 once and streams its own tile rows of p, each row's K
    reduction in the float32 DST (k ascending) -- the M tile rows split over the whole grid, where the stock 2D
    config spreads the narrow N over 6 of the 8 grid columns. None for a product it does not take."""
    s, ws = list(p.shape), list(w.shape)
    if len(ws) != 2 or len(s) < 2 or s[-1] != ws[0] or s[-1] % TILE or ws[1] % TILE or s[-2] % TILE:
        return None
    kt, nt = ws[0] // TILE, ws[1] // TILE
    if kt * nt > _SW_MAX_W_TILES or not _plain_bf16(p, w):
        return None
    rows = 1
    for d in s[:-1]:
        rows *= d
    rt = rows // TILE
    y = _rows_mm(p, w, rt, kt, nt, rt, 0, False, fidelity)
    return ttnn.view(y, s[:-1] + [ws[1]])


def batched_matmul(a, b, transpose_b=False, b_rep=1):
    """a [..., M, K] @ b [..., K, N] (b [..., N, K] when transpose_b), bf16 -> float32, one weight block per batch
    entry (the LM's per-head attention products: 112 entries of 96 x 128 @ 128 x 192 / 96 x 192 @ 192 x 128). The
    tile rows of all entries split over the whole grid; each core reads an entry's block into L1 once. None for a
    product it does not take."""
    sa, sb = list(a.shape), list(b.shape)
    kb, n = (sb[-1], sb[-2]) if transpose_b else (sb[-2], sb[-1])
    if len(sa) < 3 or len(sb) != len(sa) or sa[-1] != kb or any(d % TILE for d in (sa[-2], sa[-1], n)):
        return None
    ea, eb = 1, 1
    for d in sa[:-2]:
        ea *= d
    for d in sb[:-2]:
        eb *= d
    if ea != eb * b_rep or (b_rep == 1 and sa[:-2] != sb[:-2]):  # b_rep: each b entry serves b_rep a entries
        return None
    if list(a.padded_shape)[-2] != sa[-2] or list(b.padded_shape)[-2] != sb[-2] or not _plain_bf16(a, b):
        return None
    kt, nt, mt = sa[-1] // TILE, n // TILE, sa[-2] // TILE
    if kt * nt > _SW_MAX_W_TILES:
        return None
    entries = 1
    for d in sa[:-2]:
        entries *= d
    y = _rows_mm(a, b, entries * mt, kt, nt, mt, kt * nt, transpose_b, ttnn.MathFidelity.HiFi4, w_rep=b_rep)
    return ttnn.view(y, sa[:-1] + [n])


_HS_READER = f"{_DIR}/head_split_reader.cpp"


def head_split(qkv, heads, memory_config=None, parts=3, merge=False):
    """nlp_create_qkv_heads(qkv, num_heads=heads, num_kv_heads=heads, transpose_k_heads=False) for a float32 tile
    tensor qkv [N, 1, S, 3 * heads * D] -> (q, k, v) [N, heads, S, D], as tile copies (each output tile read from
    its qkv page, the three written by the three-output writer). parts=1: one projection x [N, S, heads * D] (or
    [N, 1, S, heads * D]) -> [N, heads, S, D], the reshape + permute of a head split (a ragged S runs over the padded
    tiles: the padding rows' values are never read). merge=True (parts=1): the inverse, x [N, heads, S, D] ->
    [N, S, heads * D]. None for a tensor it does not take."""
    s = list(qkv.padded_shape)
    logical = list(qkv.shape)
    if parts == 1 and not merge and len(s) == 3:
        s, logical = [s[0], 1, s[1], s[2]], [logical[0], 1, logical[1], logical[2]]
    if qkv.dtype != ttnn.float32 or qkv.layout != ttnn.TILE_LAYOUT or qkv.is_sharded() or len(s) != 4:
        return None
    if merge:
        if parts != 1 or logical[-1] != s[-1] or s[-1] % TILE:
            return None
        n, heads_in, sq, d = s
        if heads_in != heads:
            return None
        out_shape = [n, logical[2], heads * d]
    else:
        if s[1] != 1 or logical[-1] != s[-1] or s[3] % (parts * heads * TILE):
            return None
        n, sq, d = s[0], s[2], s[3] // (parts * heads)
        out_shape = [n, heads, logical[2], d]
    dt, st = d // TILE, sq // TILE
    mc = memory_config or qkv.memory_config()
    ys = [
        ttnn.empty(out_shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=qkv.device(), memory_config=mc)
        for _ in range(parts)
    ]
    pages = n * heads * st * dt
    grid = qkv.device().compute_with_storage_grid_size()
    work = _cores(grid, pages)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=_OUT_CB + k, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
            ],
        )
        for k in range(parts)
    ]
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        rd[c.x][c.y] = [qkv.buffer_address(), start, count]
        wr[c.x][c.y] = (
            [ys[0].buffer_address(), count, start] if parts == 1 else [t.buffer_address() for t in ys] + [count, start]
        )
    writer_cta = (
        [_OUT_CB] + ttnn.TensorAccessorArgs(ys[0]).get_compile_time_args()
        if parts == 1
        else [3] + [a for t in ys for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    )
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_HS_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(qkv).get_compile_time_args() + [heads, dt, st, parts, int(merge)],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER if parts == 1 else _TWO_OUT_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=writer_cta,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([qkv] + ys, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return tuple(ys) if parts == 3 else ys[0]


_RG_READER = f"{_DIR}/row_gather_reader.cpp"
_RG_WRITER = f"{_DIR}/row_gather_writer.cpp"


def row_gather(x, perm):
    """y [..., j, :] = x [..., perm[j], :] for j < len(perm) within every leading entry of a float32 tile tensor
    (the rows past len(perm) are left unset: the caller drops them) -- an exact row permutation, each row moved as
    its two 64-byte face rows straight between tiles. None for a tensor it does not take."""
    s = list(x.padded_shape)
    if x.dtype != ttnn.float32 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or len(s) < 2:
        return None
    if s[-2] % TILE or s[-1] % TILE or len(perm) > s[-2]:
        return None
    entries = 1
    for d in s[:-2]:
        entries *= d
    mt, nt = s[-2] // TILE, s[-1] // TILE
    y = ttnn.empty_like(x)
    grid = x.device().compute_with_storage_grid_size()
    work = _cores(grid, entries * nt)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * mt * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
            ],
        )
    ]
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    perm = [int(p) for p in perm]
    for c, start, count in work:
        rd[c.x][c.y] = [x.buffer_address(), start, count] + perm
        wr[c.x][c.y] = [y.buffer_address(), start, count]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_RG_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args() + [mt, nt, len(perm)],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_RG_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args() + [mt, nt],
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([x, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y


_KR_READER = f"{_DIR}/chunk_copy_reader.cpp"
_KR_WRITER = f"{_DIR}/k_repeat_writer.cpp"
_KR_CHUNK = 8


def repeat_k(x, reps):
    """y [..., M, reps * K] = [x | x | ... | x] (reps copies side by side) for an interleaved bf16 tile tensor
    [..., M, K] (K tile-aligned), from one read of x; None for a tensor it does not take."""
    s = list(x.shape)
    ps = list(x.padded_shape)
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or len(s) < 2 or s[-1] % TILE:
        return None
    kt = ps[-1] // TILE
    pages = 1
    for d in ps[:-2]:
        pages *= d
    pages *= (ps[-2] // TILE) * kt
    y = ttnn.empty(
        s[:-1] + [reps * s[-1]],
        dtype=x.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=x.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = x.device().compute_with_storage_grid_size()
    work = _cores(grid, pages)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
    page = _FP32_TILE_BYTES // 2
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * _KR_CHUNK * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=x.dtype, page_size=page)],
        )
    ]
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        rd[c.x][c.y] = [x.buffer_address(), start, count]
        wr[c.x][c.y] = [y.buffer_address(), start, count]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_KR_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args() + [_KR_CHUNK],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_KR_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args() + [_KR_CHUNK, kt, reps],
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([x, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y


_RS_READER = f"{_DIR}/row_sumsq_reader.cpp"
_RS_COMPUTE = f"{_DIR}/row_sumsq_compute.cpp"


def row_sumsq(t):
    """sum(t * t, dim=-1, keepdim=True) in float32 for an interleaved bf16 or float32 tile tensor [..., M, K] (K
    tile-aligned), from one read of t (each square in t's dtype, row-reduced into a float32 running sum); None for a
    tensor it does not take."""
    s = list(t.shape)
    ps = list(t.padded_shape)
    if t.dtype not in (ttnn.bfloat16, ttnn.float32) or t.layout != ttnn.TILE_LAYOUT or t.is_sharded():
        return None
    if len(s) < 2 or s[-1] % TILE:
        return None
    kt = ps[-1] // TILE
    rows = 1
    for d in ps[:-2]:
        rows *= d
    rows *= ps[-2] // TILE
    y = ttnn.empty(
        s[:-1] + [1],
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=t.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = t.device().compute_with_storage_grid_size()
    work = _cores(grid, rows)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb(index, dtype, pages):
        page = _FP32_TILE_BYTES if dtype == ttnn.float32 else _FP32_TILE_BYTES // 2
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    cbs = [
        cb(0, t.dtype, 2),
        cb(1, ttnn.bfloat16, 1),
        cb(2, t.dtype, 2),
        cb(3, ttnn.float32, 2),
        cb(_OUT_CB, ttnn.float32, 2),
    ]
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        rd[c.x][c.y] = [t.buffer_address(), start, count]
        cp[c.x][c.y] = [count]
        wr[c.x][c.y] = [y.buffer_address(), count, start]
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[3] = ttnn.UnpackToDestMode.UnpackToDestFp32  # the running sum reloads exactly
    if t.dtype == ttnn.float32:
        unpack[0] = ttnn.UnpackToDestMode.UnpackToDestFp32  # a float32 input squared from its exact value
    compute_cfg.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_RS_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(t).get_compile_time_args() + [kt],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_RS_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[kt],
            runtime_args=cp,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[_OUT_CB] + ttnn.TensorAccessorArgs(y).get_compile_time_args(),
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([t, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return y


_MOD_READER = f"{_DIR}/modulate_reader.cpp"
_MOD_COMPUTE = f"{_DIR}/modulate_compute.cpp"


def modulate_tail(xc, r, s, h):
    """((xc * r) * s) + h for a centred float32 tile tensor xc [B, S, D], with r [B, S, 32] (each row's 1/std across a
    tile), s and h [B, 32, D] (each sample's (1 + scale) / shift down a tile): the modulated LayerNorm's three
    broadcast passes in one, bit-identical. Consumes nothing (the caller frees). None for shapes it does not take."""
    if len(xc.shape) != 3 or not _plain(xc, list(xc.shape)):
        return None
    ps = list(xc.padded_shape)
    b, st, ct = ps[0], ps[1] // TILE, ps[2] // TILE
    if list(r.padded_shape) != [b, ps[1], TILE] or list(s.padded_shape) != [b, TILE, ps[2]]:
        return None
    if list(h.padded_shape) != [b, TILE, ps[2]] or any(t.dtype != ttnn.float32 for t in (r, s, h)):
        return None
    n_tiles = b * st * ct
    # r, s and h are re-read for every output tile (r once per tile column, s and h once per tile row): staged in L1
    # (< 1 MB together), three of the four reads per tile leave DRAM
    staged = [
        ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
        for t in (r, s, h)
        if t.memory_config().buffer_type == ttnn.BufferType.DRAM
    ]
    if len(staged) == 3:
        r, s, h = staged
    else:  # not expected (the caller's operands are DRAM): leave them as they are
        for t in staged:
            ttnn.deallocate(t)
        staged = []
    y = ttnn.empty_like(xc)
    grid = xc.device().compute_with_storage_grid_size()
    work = _cores(grid, n_tiles)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])

    def cb(index):
        return ttnn.CBDescriptor(
            total_size=2 * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
            ],
        )

    cbs = [cb(0), cb(1), cb(2), cb(3), cb(_OUT_CB)]
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in (xc, r, s, h)]
    for c, start, count in work:
        rd[c.x][c.y] = addrs + [count, start]
        cp[c.x][c.y] = [count]
        wr[c.x][c.y] = [y.buffer_address(), count, start]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for k in range(4):
        unpack[k] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_cfg.unpack_to_dest_mode = unpack
    reader_cta = []
    for t in (xc, r, s, h):
        reader_cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_MOD_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_cta + [ct, st],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_MOD_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=cp,
            config=compute_cfg,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[_OUT_CB] + ttnn.TensorAccessorArgs(y).get_compile_time_args(),
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([xc, r, s, h, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    for t in staged:
        ttnn.deallocate(t)
    return y
