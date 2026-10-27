# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One lane pattern of an exact-lane batched product of two-limb operands as one C++ Metalium kernel
(ttnn.generic_op): sum over the limb terms (hi.hi, hi.lo, lo.hi) and the 8 lanes of (a_i * mask_r) @ b_j.

The ports' split_matmul spells each pattern as 8 lanes x 3 terms of multiply (lane copy) + matmul + add,
72 small programs; kernels/lane_bmm.cpp does the lane copies (as exact matmuls with diagonal 0/1 tiles),
the full-K lane products in the float32 DEST and their float32 adds in one program. Plugged into
split_matmul as its EXACT_BMM_FN.
fused_lane_linear does the same for split_linear's exact-lane products against one bf16 weight
(kernels/lane_linear.cpp, EXACT_LINEAR_FN).
"""
from __future__ import annotations

import math
import os

import ttnn

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TT_METAL = os.environ.get("TT_METAL_HOME", "")
_TILE = 32
_LANES = 8
# diagonal lane tiles per (device, lane assignment)
_DIAG = {}


def _diag_tiles(device, lanes, kt):
    """[8 * kt * 32, 32] bf16 TILE: tile r * kt + t is diag(lane(t * 32 + c) == r) (0 beyond len(lanes))."""
    key = (id(device), tuple(lanes), kt)
    if key not in _DIAG:
        vals = []
        for r in range(_LANES):
            for t in range(kt):
                for row in range(_TILE):
                    k = t * _TILE + row
                    on = 1.0 if k < len(lanes) and lanes[k] == r else 0.0
                    vals += [on if col == row else 0.0 for col in range(_TILE)]
        d = ttnn.Tensor(vals, [_LANES * kt * _TILE, _TILE], ttnn.float32, ttnn.TILE_LAYOUT, device)
        _DIAG[key] = ttnn.typecast(d, ttnn.bfloat16)
    return _DIAG[key]


def _tile_of(unit, chunks, csize, n_t):
    """First output tile of work unit `unit` = (row block unit // chunks, column chunk unit % chunks)."""
    return (unit // chunks) * n_t + (unit % chunks) * csize


def fused_lane_bmm(pa, pb, lanes, transpose_b, mem):
    """pa, pb: (hi, lo) bf16 TILE limbs of a [..., M, K] and b [..., K, N] ([..., N, K] if transpose_b),
    same batch dims; lanes: lane id of each of a's K entries. -> float32 [..., M, N] in `mem`.
    None when the operands do not fit the kernel (the caller then runs the stock path)."""
    k_t = int(pa[0].padded_shape[-1]) // _TILE if len(pa) == 2 else 0
    diag = lambda device: _diag_tiles(device, lanes, k_t)  # noqa: E731
    return lane_program(pa, pb, transpose_b, mem, diag, _LANES * k_t, 2 * _LANES * k_t, "lane_bmm.cpp")


def lane_program(pa, pb, transpose_b, mem, diag_fn, n_diag, n_copies, compute_kernel):
    """One exact-lane product program over two-limb operands (reader_lane_bmm.cpp + compute_kernel + the
    stock tile writer): diag_fn(device) -> the n_diag diagonal lane tiles, n_copies lane-copy tiles per row
    block in cb 3. -> float32 [..., M, N] in `mem`, or None when the operands do not fit the kernel."""
    if len(pa) != 2 or len(pb) != 2:
        return None
    a_s, b_s = [int(v) for v in pa[0].padded_shape], [int(v) for v in pb[0].padded_shape]
    if a_s[:-2] != b_s[:-2]:
        return None
    m_t, k_t = a_s[-2] // _TILE, a_s[-1] // _TILE
    n_t = (b_s[-2] if transpose_b else b_s[-1]) // _TILE
    if k_t != (b_s[-1] if transpose_b else b_s[-2]) // _TILE:
        return None
    # cb tiles per core: a and b blocks (2 limbs, double-buffered b), the lane tiles, the lane copies
    if (2 * k_t + 4 * k_t + n_diag + n_copies) * 2048 + 2 * 4096 > 900 * 1024:
        return None
    device = pa[0].device()
    batch = math.prod(a_s[:-2])
    n_logical = int(pb[0].shape[-2] if transpose_b else pb[0].shape[-1])
    out_shape = list(pa[0].shape)[:-1] + [n_logical]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.float32, ttnn.TILE_LAYOUT, device, mem)
    diag = diag_fn(device)

    grid = device.compute_with_storage_grid_size()
    cores_n = grid.x * grid.y
    rows = batch * m_t
    chunks = min(n_t, max(1, -(-cores_n // rows)))
    csize = -(-n_t // chunks)
    chunks = -(-n_t // csize)
    units = rows * chunks
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, units)

    def _cb(index, dtype, page, n):
        return ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048, 2 * k_t),
        _cb(1, ttnn.bfloat16, 2048, 4 * k_t),
        _cb(2, ttnn.bfloat16, 2048, n_diag),
        _cb(3, ttnn.bfloat16, 2048, n_copies),
        _cb(16, ttnn.float32, 4096, 2),
    ]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    addrs = [pa[0].buffer_address(), pa[1].buffer_address(), pb[0].buffer_address(), pb[1].buffer_address()]
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    # this core's units cover output tiles [first, last) contiguously (row-major (b, mt, nt))
                    first, last = _tile_of(start, chunks, csize, n_t), _tile_of(start + per, chunks, csize, n_t)
                    rd_rt[cx][cy] = addrs + [diag.buffer_address(), start, per]
                    wr_rt[cx][cy] = [out.buffer_address(), last - first, first]
                    cp_rt[cx][cy] = [start, per]
                    start += per
    ct = [m_t, k_t, n_t, chunks, csize, 1 if transpose_b else 0]
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.dst_full_sync_en = True  # 8 float32 DEST tiles: lane_bmm.cpp batches 7 lane products per group
    cfg.math_approx_mode = False
    acc = []
    for t in (pa[0], pa[1], pb[0], pb[1], diag):
        acc += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_lane_bmm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct + [n_diag] + acc,
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(
                _TT_METAL,
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            ),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, compute_kernel),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    ttnn.generic_op([pa[0], pa[1], pb[0], pb[1], diag, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out


# the 8 diagonal lane tiles diag(col % 8 == q) per device
_DIAG8 = {}


def _diag8_tiles(device):
    """[8 * 32, 32] bf16 TILE: tile q is diag(col % 8 == q) (the lane copy a @ E_q keeps a's columns c % 8 == q)."""
    if id(device) not in _DIAG8:
        vals = [
            1.0 if col == row and col % _LANES == q else 0.0
            for q in range(_LANES)
            for row in range(_TILE)
            for col in range(_TILE)
        ]
        d = ttnn.Tensor(vals, [_LANES * _TILE, _TILE], ttnn.float32, ttnn.TILE_LAYOUT, device)
        _DIAG8[id(device)] = ttnn.typecast(d, ttnn.bfloat16)
    return _DIAG8[id(device)]


def _lane_shift(lanes):
    """s with lanes[k] == (k % 8 + s * (k // 32)) % 8 for every k (the 8-spaced rotated patterns), else None."""
    for s in range(_LANES):
        if all(v == (k % _LANES + s * (k // _TILE)) % _LANES for k, v in enumerate(lanes)):
            return s
    return None


def fused_lane_linear(parts, w, lanes, mem):
    """One lane pattern of split_linear's exact-lane product: sum over the bf16 limbs (parts, [..., M, K], 1-3)
    and the 8 lanes of (part * mask_r) @ w (w: bf16 [K, N] TILE, any interleaved buffer); lanes: lane id of each
    K entry. -> float32 [..., M, N] in `mem` (kernels/lane_linear.cpp), or None when the operands do not fit
    the kernel (the caller then runs the stock ops)."""
    return fused_lane_linear_patterns(parts, w, [lanes], mem)


def fused_lane_linear_patterns(parts, w, lane_lists, mem, bias=None):
    """fused_lane_linear over 1 or 3 lane patterns in one program; with 3, the elementwise median of the three
    patterns' products (split_linear's vote), the weight block and limbs loaded once for all of them. bias (3
    patterns only): float32 [1, 1, 32, N] TILE, the bias in every row, added to the median by one float32 add
    (round to nearest even), as a ttnn.add of the [1, 1, 1, N] bias after the vote would."""
    if len(lane_lists) not in (1, 3):
        return None
    if bias is not None and (
        len(lane_lists) != 3
        or bias.dtype != ttnn.float32
        or bias.layout != ttnn.TILE_LAYOUT
        or bias.is_sharded()
        or [int(v) for v in bias.padded_shape][-2:] != [_TILE, int(w.padded_shape[-1])]
    ):
        return None
    if not 1 <= len(parts) <= 3 or len(w.shape) != 2 or w.dtype != ttnn.bfloat16:
        return None
    if any(p.dtype != ttnn.bfloat16 or p.layout != ttnn.TILE_LAYOUT or p.is_sharded() for p in parts):
        return None
    if w.layout != ttnn.TILE_LAYOUT or w.is_sharded():
        return None
    a_s, w_s = [int(v) for v in parts[0].padded_shape], [int(v) for v in w.padded_shape]
    k = int(parts[0].shape[-1])
    shifts = [_lane_shift(lanes) for lanes in lane_lists]
    # whole K tiles only (no padded K entries for the lane masks to cover), K matching the weight
    if any(s is None for s in shifts) or any(len(lanes) != k for lanes in lane_lists):
        return None
    if k % _TILE or a_s[-1] != k or w_s[0] != k or int(w.shape[0]) != k:
        return None
    n_pat = len(lane_lists)
    if any([int(v) for v in p.padded_shape] != a_s for p in parts[1:]):
        return None
    k_t, n_t = k // _TILE, w_s[1] // _TILE
    rows = math.prod(a_s[:-2]) * (a_s[-2] // _TILE)
    device = parts[0].device()
    grid = device.compute_with_storage_grid_size()
    cores_n = grid.x * grid.y
    # work unit = (row block, csize output columns) with the weight's column block resident in L1; cb bytes per
    # core: a limb (double-buffered), the weight block, the 8 lane tiles, one lane copy, the running sums and
    # the output (float32). Pick the column split with the least work on the busiest core (lane copies cost
    # about two columns' products)
    a_tiles = len(parts) * k_t  # every limb's row block, held for all the unit's patterns
    best = None
    for c in range(1, n_t + 1):
        csize = -(-n_t // c)
        chunks = -(-n_t // csize)
        est = 6 * csize if n_pat == 3 else 0  # the three patterns' sums awaiting their median (double-buffered)
        est += csize if bias is not None else 0  # and the unit's bias tiles
        if (a_tiles + _LANES + k_t + csize * k_t) * 2048 + (csize + est + 4) * 4096 > 900 * 1024:
            continue
        cost = -(-rows * chunks // cores_n) * (csize + 2)
        if best is None or cost < best[0]:
            best = (cost, chunks, csize)
    if best is None:
        return None
    _, chunks, csize = best
    units = rows * chunks
    out_shape = list(parts[0].shape)[:-1] + [int(w.shape[-1])]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.float32, ttnn.TILE_LAYOUT, device, mem)
    diag = _diag8_tiles(device)

    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, units)

    def _cb(index, dtype, page, n):
        return ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048, a_tiles),
        _cb(1, ttnn.bfloat16, 2048, csize * k_t),
        _cb(2, ttnn.bfloat16, 2048, _LANES),
        _cb(3, ttnn.bfloat16, 2048, k_t),
        _cb(4, ttnn.float32, 4096, csize),
        _cb(16, ttnn.float32, 4096, 4),
    ]
    if n_pat == 3:  # each pattern's sums in its own cb (5, 6, 7), read front tile first
        cbs += [_cb(5 + q, ttnn.float32, 4096, 2 * csize) for q in range(3)]
    if bias is not None:
        cbs.append(_cb(8, ttnn.float32, 4096, csize))
    a_list = list(parts) + [parts[0]] * (3 - len(parts))  # unused limb slots read nothing
    addrs = [p.buffer_address() for p in a_list] + [w.buffer_address(), diag.buffer_address()]
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    # this core's units cover output tiles [first, last) contiguously (row-major (row, nt))
                    first, last = _tile_of(start, chunks, csize, n_t), _tile_of(start + per, chunks, csize, n_t)
                    rd_rt[cx][cy] = addrs + [start, per, bias.buffer_address() if bias is not None else 0]
                    wr_rt[cx][cy] = [out.buffer_address(), last - first, first]
                    cp_rt[cx][cy] = [start, per]
                    start += per
    # the reader's (its tensor accessors follow at index 6; index 5: whether it streams a bias)
    ct = [k_t, n_t, chunks, csize, len(parts), int(bias is not None)]
    shift3 = (shifts + [0, 0])[:3]
    ct_compute = [k_t, n_t, chunks, csize, len(parts)] + shift3 + [n_pat, int(bias is not None)]
    # as the stock linear's precise_config: HiFi4, float32 DEST; the running sums reload exactly (UnpackToDestFp32)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 32
    modes[4] = ttnn.UnpackToDestMode.UnpackToDestFp32
    for q in range(4):  # the patterns' sums and the bias
        modes[5 + q] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    acc = []
    for t in a_list + [w, diag, bias if bias is not None else diag]:  # the bias slot reads nothing without one
        acc += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_lane_linear.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct + acc,
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(
                _TT_METAL,
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            ),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "lane_linear.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct_compute,
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    ttnn.generic_op(list(parts) + [w, diag] + ([bias] if bias is not None else []) + [out], prog)
    return out
