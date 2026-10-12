# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The precise batched matmul of bf16 limbs, y = (ah @ bh + ah @ bl) + al @ bh in float32, as one C++ Metalium
kernel (ttnn.generic_op) on the full core grid (_precise.MATMUL_FN; the attention's P @ V).

_precise.matmul spells it as three ttnn.matmul + two ttnn.add. With A in L1, the products' auto config is
create_simple_matmul_program_config: for these shapes (Mt ~19 above the per-core factor, one N block) the 1-D
in1-multicast config (in0_block_w = 1 for odd Kt), or MatmulMultiCoreProgramConfig (bmm.cpp) when the blocks do
not map onto the grid. Both form each output tile over the full K in the float32 DEST: bmm.cpp directly, the 1-D
factory by reloading its K-block partials through an UnpackToDestFp32 view (exact). kernels/limb_bmm.cpp does
the same full-K matmul_tiles accumulation (HiFi4, float32 DEST) of the three products side by side in DEST and
the two float32 adds with the SFPU add (round to nearest even) binary_ng uses, so y is bit-identical.
RELOAD_TF32 instead mirrors the reuse factory (MatmulMultiCoreReuseProgramConfig, in0_block_w = 1), whose float32
partials reload through SrcA (TF32) after every K tile, for the case the products run on that config.

Work unit = (batch, block of R output tile rows): a's R rows (both limbs) are read once per unit, b's columns
(both limbs) stream past them, double-buffered when L1 allows.
"""
from __future__ import annotations

import math
import os

import ttnn

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TILE = 32
# True: the three partials are packed (float32) and reloaded through SrcA after every K tile, as the
# MatmulMultiCoreReuseOptimized factory does with in0_block_w = 1 (False: full-K float32 DEST accumulation)
RELOAD_TF32 = False
# output tile rows per work unit (None: the fewest rows per core that fit the CB budget)
ROWS = None
# cb bytes per core
_CB_BUDGET = 300 * 1024


def _cb_bytes(k_t, rows, b_cols):
    """a's rows (2 limbs), b_cols buffered b columns (2 limbs), [the 3 + 3 reload partials], 2 output tiles."""
    return (2 * k_t * rows + 2 * k_t * b_cols) * 2048 + ((6 if RELOAD_TF32 else 0) + 2) * 4096


def _pick_rows(total_rows, m_t, k_t, cores):
    """(rows per unit, buffered b columns): rows up to the balanced share (the busiest core's row count
    unchanged), as many as fit with double-buffered b columns, else single-buffered; None if nothing fits."""
    share = -(-total_rows // cores)
    cands = [ROWS] if ROWS is not None else range(min(share, m_t), 0, -1)
    for b_cols in (2, 1):
        for rows in cands:
            if _cb_bytes(k_t, rows, b_cols) <= _CB_BUDGET:
                return rows, b_cols
    return None


def fused_limb_bmm(ah, al, bh, bl, mem):
    """ah, al [..., M, K], bh, bl [..., K, N]: bf16 TILE interleaved limbs, same batch dims -> float32
    (ah @ bh + ah @ bl) + al @ bh [..., M, N] in `mem`, or None when the operands do not fit the kernel."""
    ts = (ah, al, bh, bl)
    if any(t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.is_sharded() for t in ts):
        return None
    a_s, b_s = [int(v) for v in ah.padded_shape], [int(v) for v in bh.padded_shape]
    if [int(v) for v in al.padded_shape] != a_s or [int(v) for v in bl.padded_shape] != b_s:
        return None
    if len(a_s) < 3 or a_s[:-2] != b_s[:-2] or a_s[-1] != b_s[-2]:
        return None
    m_t, k_t, n_t = a_s[-2] // _TILE, a_s[-1] // _TILE, b_s[-1] // _TILE
    batch = math.prod(a_s[:-2])
    device = ah.device()
    grid = device.compute_with_storage_grid_size()
    pick = _pick_rows(batch * m_t, m_t, k_t, grid.x * grid.y)
    if pick is None:
        return None
    rows, b_cols = pick
    blocks = -(-m_t // rows)
    out_shape = list(ah.shape)[:-1] + [int(bh.shape[-1])]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.float32, ttnn.TILE_LAYOUT, device, mem)

    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    _, cores, group1, group2, per1, per2 = ttnn.split_work_to_cores(all_cores, batch * blocks)

    def _cb(index, dtype, page, n):
        return ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048, 2 * k_t * rows),
        _cb(1, ttnn.bfloat16, 2048, 2 * k_t * b_cols),
        _cb(16, ttnn.float32, 4096, 2),
    ]
    if RELOAD_TF32:
        cbs.append(_cb(2, ttnn.float32, 4096, 6))
    rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addrs = [t.buffer_address() for t in ts]
    start = 0
    for group, per in ((group1, per1), (group2, per2)):
        for r in group.ranges():
            for cx in range(r.start.x, r.end.x + 1):
                for cy in range(r.start.y, r.end.y + 1):
                    rd_rt[cx][cy] = addrs + [start, per]
                    wr_rt[cx][cy] = [out.buffer_address(), start, per]
                    cp_rt[cx][cy] = [start, per]
                    start += per
    ct = [m_t, k_t, n_t, blocks, rows, int(RELOAD_TF32)]
    # as precise_config: HiFi4, float32 DEST, no packer L1 accumulation; every buffer at its default unpack (the
    # bf16 limbs; the RELOAD_TF32 partials through SrcA like the reuse factory's intermed buffer)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    cfg.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * 32
    acc = []
    for t in ts:
        acc += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_limb_bmm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct + acc,
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "writer_limb_bmm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[m_t, n_t, blocks, rows] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "limb_bmm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=cp_rt,
            config=cfg,
        ),
    ]
    ttnn.generic_op(list(ts) + [out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
