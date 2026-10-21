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


def head_split(qkv, heads, memory_config=None):
    """nlp_create_qkv_heads(qkv, num_heads=heads, num_kv_heads=heads, transpose_k_heads=False) for a float32 tile
    tensor qkv [N, 1, S, 3 * heads * D] -> (q, k, v) [N, heads, S, D], as tile copies (each output tile read from
    its qkv page, the three written by the three-output writer). None for a tensor it does not take."""
    s = list(qkv.padded_shape)
    if qkv.dtype != ttnn.float32 or qkv.layout != ttnn.TILE_LAYOUT or qkv.is_sharded() or len(s) != 4 or s[1] != 1:
        return None
    if list(qkv.shape) != s or s[2] % TILE or s[3] % (3 * heads * TILE):
        return None
    n, sq, d = s[0], s[2], s[3] // (3 * heads)
    dt, st = d // TILE, sq // TILE
    mc = memory_config or qkv.memory_config()
    ys = [
        ttnn.empty([n, heads, sq, d], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=qkv.device(), memory_config=mc)
        for _ in range(3)
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
        for k in range(3)
    ]
    rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, start, count in work:
        rd[c.x][c.y] = [qkv.buffer_address(), start, count]
        wr[c.x][c.y] = [t.buffer_address() for t in ys] + [count, start]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_HS_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ttnn.TensorAccessorArgs(qkv).get_compile_time_args() + [heads, dt, st],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_TWO_OUT_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[3] + [a for t in ys for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([qkv] + ys, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return tuple(ys)


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
