# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A precise linear's two limb products hi @ w + lo @ w as one C++ Metalium program (ttnn.generic_op).

The transformer's limb linears run one stock matmul per limb (each streaming the weight, each writing a float32
product) and then add the products. kernels/limb_linear.cpp keeps each work unit's rows of BOTH limbs (full K)
resident in L1 and multiplies every K chunk of the weight by both limbs, each into its own float32 DEST tiles, then
adds the two with the SFPU float32 add: the weight is streamed once for both limbs, the products' K reductions are
the stock matmul's fp32 sums, and one float32 result is written. The weight reaches the compute cores by multicast from a sender core outside the
compute rectangle (kernels/sender_limb_linear.cpp), read from DRAM once per round of work units; a wide output
(K < N) splits its columns over the grid columns, one sender per grid column streaming that column's slice.
Plugged into the transformer's _precise.linear as its LIMB_LINEAR_FN.
"""
from __future__ import annotations

import math
import os

import ttnn

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TILE = 32
# float32 DEST tiles per output block (fp32 accumulation, half-sync DEST)
_DEST_TILES = 4
# L1 bytes per core for the resident rows of both limbs
_ROWS_L1_BYTES = 400 * 1024


def _largest_divisor(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def fused_limb_linear(hi, lo, w, mem):
    """hi, lo: bf16 TILE [..., M, K] limbs; w: bf16 TILE [K, N]. -> float32 [..., M, N] = hi @ w + lo @ w in `mem`,
    or None when the shapes do not fit the kernel (the caller then runs the stock matmuls)."""
    if w.dtype != ttnn.bfloat16 or hi.dtype != ttnn.bfloat16 or lo.dtype != ttnn.bfloat16:
        return None
    hs, ws = [int(d) for d in hi.padded_shape], [int(d) for d in w.padded_shape]
    if math.prod(ws[:-2]) != 1 or hs[-1] != ws[-2] or [int(d) for d in lo.padded_shape] != hs:
        return None
    rows, k_t, n_t = math.prod(hs[:-1]) // _TILE, hs[-1] // _TILE, ws[-1] // _TILE
    mb = 1
    if 2 * mb * k_t * 2048 > _ROWS_L1_BYTES:
        return None
    device = hi.device()
    grid = device.compute_with_storage_grid_size()
    cols = grid.x
    # column groups: a wide output (K < N) splits its columns over the grid columns, each grid column taking its
    # own slice of the weight from its own sender, so a core's work (and the weight it is sent) shrinks with N;
    # a long-K one (K >= N) is one group, every core taking whole output rows
    groups = cols if k_t < n_t and n_t % cols == 0 else 1
    gw, n_g = cols // groups, n_t // groups  # grid columns and output column tiles per group
    nb = _largest_divisor(n_g, _DEST_TILES // (2 * mb))  # hi and lo accumulate in their own DEST tiles
    kc = _largest_divisor(k_t, max(1, 32 // nb))  # 64 KB weight chunks (2 slots in cb 1)
    # compute rectangle: whole grid rows (leaving one row for the senders), each group's units dealt round-robin
    # over its gw x rect_rows cores
    rect_rows = max((r for r in range(1, grid.y) if (rows // mb) % (r * gw) == 0), default=0)
    if rect_rows == 0:
        return None
    n_cores = rect_rows * gw  # per group
    units = rows // mb // n_cores

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(hi.shape)[:-1] + [int(w.shape[-1])]), ttnn.float32, ttnn.TILE_LAYOUT, device, mem
    )
    rect = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rect_rows - 1))
    senders = [ttnn.CoreCoord(g * gw, rect_rows) for g in range(groups)]
    compute_cores = ttnn.CoreRangeSet([rect])
    sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(senders[0], senders[-1])])
    all_cores = ttnn.CoreRangeSet([rect, ttnn.CoreRange(senders[0], senders[-1])])

    def _cb(index, dtype, page, n):
        return ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=all_cores,  # one layout on every core: the multicast lands at the sender's cb 1 address
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048, 2 * mb * k_t),
        _cb(1, ttnn.bfloat16, 2048, 2 * kc * nb),
        _cb(16, ttnn.float32, 4096, 2 * mb * nb),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_cores, initial_value=0) for i in (0, 1)]
    rd_rt, wr_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for g, sender in enumerate(senders):
        x0 = g * gw
        sender_phys = device.worker_core_from_logical_core(sender)
        for y in range(rect_rows):
            for x in range(x0, x0 + gw):
                first = y * gw + x - x0
                rd_rt[x][y] = [hi.buffer_address(), lo.buffer_address(), first, sender_phys.x, sender_phys.y]
                wr_rt[x][y] = [out.buffer_address(), first, g * n_g]
        start = device.worker_core_from_logical_core(ttnn.CoreCoord(x0, 0))
        end = device.worker_core_from_logical_core(ttnn.CoreCoord(x0 + gw - 1, rect_rows - 1))
        sd_rt[sender.x][sender.y] = [w.buffer_address(), start.x, start.y, end.x, end.y, g * n_g]

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    rd_ct = [k_t, n_g, mb, nb, kc, units, n_cores]
    rd_ct += list(ttnn.TensorAccessorArgs(hi).get_compile_time_args())
    rd_ct += list(ttnn.TensorAccessorArgs(lo).get_compile_time_args())
    sd_ct = [k_t, n_t, n_g, nb, kc, units, n_cores] + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "reader_limb_linear.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=compute_cores,
            compile_time_args=rd_ct + [0, 1],
            runtime_args=rd_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "writer_limb_linear.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=compute_cores,
            compile_time_args=[n_t, n_g, mb, nb, units, n_cores]
            + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "sender_limb_linear.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=sender_cores,
            compile_time_args=sd_ct + [0, 1],
            runtime_args=sd_rt,
            # NOC 0: the multicast rectangle runs from its (start) to its (end) corner
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNELS, "limb_linear.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=compute_cores,
            compile_time_args=[k_t, n_g, mb, nb, kc, units],
            runtime_args=ttnn.RuntimeArgs(),
            config=cfg,
        ),
    ]
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    return ttnn.generic_op([hi, lo, w, out], prog)
