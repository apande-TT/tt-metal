# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode input RMSNorm (unit gamma) of the width-sharded fp32 residual row as ONE program on the cores that hold its
shards (ttnn.generic_op, kernels/norm_rep_*.cpp), replicating ttnn.rms_norm's sharded kernel step for step and
emitting the normed row's bf16 TRUNCATION in the same shard layout for the HiFi2 decode QKV.

The arithmetic is layernorm_sharded.cpp's RMS path with its two-stage reduce: per core x^2 (FPU, fp32 operands as
TF32) row-reduced with the scaler bf16(40 / D); per row the 8 partials reduced with 1.0 on the row's core x = 0; on
the root (0, 0) its row's 8 partials and the 4 row results reduced with bf16(1 / 40), + bf16-truncated eps, rsqrt;
x times that factor (column broadcast). The output is then truncated to bf16 (TRUNC_BF16): a HiFi2 matmul multiplies
only the high bits of its in0, so the QKV computes exactly what it did from the fp32 row, from half the bytes."""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_F32_TILE, _BF16_TILE = 4096, 2048
CB_X, CB_X2, CB_SCALER, CB_PART, CB_EXT, CB_SCG, CB_EX2, CB_EPS, CB_EX2PE, CB_EXG, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16
SEM_GATHER, SEM_FACTOR = 0, 1
# cpp try 2 (run 8): the root multicasts the norm factor to all cores (else a root -> row worker -> row unicast tree)
MCAST_FACTOR = True


def _bf16_pair(v: float) -> int:
    """layernorm's pack_two_bfloat16_into_uint32({bf16(v), bf16(v)}): bf16(v) = the fp32 bits rounded to the top 16."""
    b = struct.unpack("<I", struct.pack("<f", v))[0]
    hi = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return (hi << 16) | hi


def supports(h) -> bool:
    mc = h.memory_config()
    spec = mc.shard_spec
    if h.dtype != ttnn.float32 or h.layout != ttnn.TILE_LAYOUT or not h.is_sharded() or spec is None:
        return False
    if mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED or mc.buffer_type != ttnn.BufferType.L1:
        return False
    if int(spec.shape[0]) != ttnn.TILE_SIZE or tuple(int(h.shape[i]) for i in range(len(h.shape) - 1)) != (1, 1, 32):
        return False
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR or len(spec.grid.ranges()) != 1:
        return False
    r = spec.grid.ranges()[0]
    nx, ny = r.end.x - r.start.x + 1, r.end.y - r.start.y + 1
    wt = int(spec.shape[1]) // ttnn.TILE_SIZE
    return (r.start.x, r.start.y, nx, ny) == (0, 0, 8, 5) and nx * ny * wt * ttnn.TILE_SIZE == int(h.shape[-1])


def trunc_norm(h, eps: float):
    """h: (1, 1, 32, D) fp32 width-sharded in L1 over the 8 x 5 grid at (0, 0); returns the bf16 truncation of
    ttnn.rms_norm(h) (unit weight), same shard spec."""
    device = h.device()
    mc = h.memory_config()
    spec = mc.shard_spec
    nx, ny = 8, 5
    np_, wt, d = nx * ny, int(spec.shape[1]) // ttnn.TILE_SIZE, int(h.shape[-1])
    out = ttnn.allocate_tensor_on_device(h.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    cores = spec.grid
    phys = lambda x, y: device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    xy_phys = lambda x, y: [phys(x, y).x, phys(x, y).y]
    scalar_w = _bf16_pair(float(np_) / d)  # winv = num_blocks / K
    cinv_root, cinv_one = _bf16_pair(1.0 / np_), _bf16_pair(1.0)
    eps_u = struct.unpack("<I", struct.pack("<f", float(eps)))[0]

    def cb(index, tiles, page, dtype):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
        return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=cores, format_descriptors=[fmt])

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, h),
        cb(CB_X2, wt, _F32_TILE, ttnn.float32),
        cb(CB_SCALER, 1, _BF16_TILE, ttnn.bfloat16),
        cb(CB_PART, 1, _F32_TILE, ttnn.float32),
        cb(CB_EXT, nx + ny - 1, _F32_TILE, ttnn.float32),
        cb(CB_SCG, 1, _BF16_TILE, ttnn.bfloat16),
        cb(CB_EX2, 1, _F32_TILE, ttnn.float32),
        cb(CB_EPS, 1, _BF16_TILE, ttnn.bfloat16),
        cb(CB_EX2PE, 1, _F32_TILE, ttnn.float32),
        cb(CB_EXG, 1, _F32_TILE, ttnn.float32),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in (SEM_GATHER, SEM_FACTOR)]
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = ttnn.MathFidelity.HiFi4
    ccfg.fp32_dest_acc_en = True
    ccfg.math_approx_mode = False
    root = xy_phys(0, 0)
    roles = {0: [], 1: [], 2: []}  # role -> [(core, dm runtime args)]
    for y in range(ny):
        for x in range(nx):
            role = 2 if (x, y) == (0, 0) else (1 if x == 0 else 0)
            if MCAST_FACTOR:
                # the root multicasts the factor to (1..7, 0) and (0..7, 1..4)
                rects = [(1, 0, nx - 1, 0), (0, 1, nx - 1, ny - 1)] if role == 2 else []
                tail = [len(rects)]
                for a_, b_, c_, d_ in rects:
                    tail += xy_phys(a_, b_) + xy_phys(c_, d_) + [(c_ - a_ + 1) * (d_ - b_ + 1)]
            else:
                if role == 2:
                    fwd = [(0, yy) for yy in range(1, ny)] + [(xx, 0) for xx in range(1, nx)]
                elif role == 1:
                    fwd = [(xx, y) for xx in range(1, nx)]
                else:
                    fwd = []
                tail = [len(fwd)]
                for fx, fy in fwd:
                    tail += xy_phys(fx, fy)
            rt = [x] + xy_phys(0, y) + root + [scalar_w, cinv_root if role == 2 else cinv_one, eps_u, y] + tail
            roles[role].append(((x, y), rt))
    kernels = []
    for role, items in roles.items():
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c, _ in items])
        nblocks = nx + ny - 1 if role == 2 else nx
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(_KDIR / "norm_rep_dm.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=[wt, role, SEM_GATHER, SEM_FACTOR, nx, ny],
                runtime_args=items,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
                defines=[("MCAST_FACTOR", "1")] if MCAST_FACTOR else [],
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(_KDIR / "norm_rep_compute.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=[wt, role, nblocks],
                runtime_args=[(c, []) for c, _ in items],
                config=ccfg,
                defines=[("TRUNC_BF16", "1")],
            )
        )
    ttnn.generic_op([h, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
    return out
