# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One lane pattern of an exact-lane batched product of two-limb operands as one C++ Metalium kernel
(ttnn.generic_op): sum over the limb terms (hi.hi, hi.lo, lo.hi) and the 8 lanes of (a_i * mask_r) @ b_j.

The ports' split_matmul spells each pattern as 8 lanes x 3 terms of multiply (lane copy) + matmul + add,
72 small programs; kernels/lane_bmm.cpp does the lane copies (as exact matmuls with diagonal 0/1 tiles),
the full-K lane products in the float32 DEST and their float32 adds in one program. Plugged into
split_matmul as its EXACT_BMM_FN.
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
