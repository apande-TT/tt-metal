# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill h = silu(x @ Wg) * (x @ Wu) as one generic_op (kernels/mm_swiglu), bit-identical to
    gate = minimal_matmul(x, Wg); up = minimal_matmul(x, Wu); h = ttnn.multiply(gate, up, input_tensor_a_activations=[SILU])
with the minimal_matmul config used by the prefill MLP (M_block 2, K_block = full K, N_block 2, 2x2 subblocks, full
compute grid, HiFi2, fp32 DEST).

Host side: a Python mirror of the legacy CB-based minimal_matmul program factory
(ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp,
minimal_matmul_factory_helper_common) for: no bias, no fused activation, no fused ternary, no AG / SRS / fused
concat, N_chunks = 1, fuse_swiglu = True. Same partition (gate/up PAIRS over the in1 axis), same CBs (indices, sizes,
formats), same semaphores, same compile-time args / defines / runtime args per core, same NoC / RISC choice and the
same core forwarding orders. The only additions: a bf16 scratch CB (c_7, one M_block x N_block block) used by the
bit-exact SwiGLU epilogue in kernels/mm_swiglu/compute.cpp.

The weight is tile-pair interleaved [gate | up] along N (column tile 2p = gate tile p, 2p+1 = up tile p): see
interleave_gate_up.
"""
from __future__ import annotations

import os

import torch

import ttnn

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "mm_swiglu")
_TILE = 32

# Block config of the reference prefill minimal_matmul (mlp.py _mm): K block = all of K.
_M_BLOCK = 2
_N_BLOCK = 2
_SUBBLOCK_H = 2
_SUBBLOCK_W = 2

# CB indices (as the legacy factory / kernels), plus the epilogue scratch.
_CB_IN0, _CB_IN1, _CB_OUT, _CB_INTERM, _CB_SCRATCH = 0, 1, 2, 3, 7
# Semaphore ids in the factory's CreateSemaphore order.
_SEM_IN0_SENDER, _SEM_IN0_RECEIVER, _SEM_IN0_VALID = 0, 1, 2
_SEM_IN1_SENDER, _SEM_IN1_RECEIVER, _SEM_IN1_VALID = 3, 4, 5
_INVALID, _VALID = 0, 1  # hostdevcommon/common_values.hpp


def interleave_gate_up(wg_t: torch.Tensor, wu_t: torch.Tensor) -> torch.Tensor:
    """[K, F] gate / up weights (already transposed to x @ W form) -> [K, 2F] tile-pair interleaved:
    column tile 2p = gate tile p, column tile 2p + 1 = up tile p."""
    k, f = wg_t.shape
    assert wu_t.shape == (k, f) and f % _TILE == 0
    wg = wg_t.reshape(k, f // _TILE, 1, _TILE)
    wu = wu_t.reshape(k, f // _TILE, 1, _TILE)
    return torch.cat([wg, wu], dim=2).reshape(k, 2 * f)


def _round_up(a, b):
    return -(-a // b) * b


def _div_up(a, b):
    return -(-a // b)


def _dims(x, w_il):
    """(M, K, N) in elements as the factory derives them: M folds every activation dim but the last."""
    k = int(x.padded_shape[-1])
    vol = 1
    for d in x.padded_shape:
        vol *= int(d)
    return vol // k, k, int(w_il.padded_shape[-1])


def mm_swiglu_supported(x, w_il, grid=None) -> bool:
    """Shapes / layouts this op handles (else use the gate / up / multiply path)."""
    try:
        if x.layout != ttnn.TILE_LAYOUT or w_il.layout != ttnn.TILE_LAYOUT:
            return False
        if x.dtype != ttnn.bfloat16 or w_il.dtype != ttnn.bfloat16:
            return False
        if x.memory_config().is_sharded() or w_il.memory_config().is_sharded():
            return False
        if len(w_il.padded_shape) < 2 or any(int(d) != 1 for d in list(w_il.padded_shape)[:-2]):
            return False
        m, k, n = _dims(x, w_il)
        if int(w_il.padded_shape[-2]) != k or list(x.shape) != list(x.padded_shape):
            return False
        if m % _TILE or k % _TILE or n % (2 * _TILE):
            return False
        # K block = all of K: the double-buffered in0 / in1 CBs take 2 * 2 * 2 * K_tiles bf16 tiles (1 MB at
        # K = 2048, the reference prefill config's footprint); do not go beyond it.
        if k // _TILE > 64:
            return False
        if grid is not None and (grid.x < 1 or grid.y < 1):
            return False
        return True
    except Exception:
        return False


def _core_order_for_axis(core, transpose, axis_length, noc_is_0, axis_is_x_when_not_transposed, initial):
    """build_core_order_for_axis: [initial endpoint] + the other cores of the axis, increasing for NOC_0 and
    decreasing for NOC_1; returns (order, index of `core`)."""
    modify_x = axis_is_x_when_not_transposed != transpose
    current = core[0] if modify_x else core[1]
    order = [initial]
    index_of_current = 0
    for worker_idx in range(1, axis_length):
        v = worker_idx if noc_is_0 else axis_length - worker_idx
        if v == current:
            index_of_current = worker_idx
        order.append((v, core[1]) if modify_x else (core[0], v))
    return order, index_of_current


def mm_swiglu(device, x, w_il, compute_fidelity=ttnn.MathFidelity.HiFi2, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """h[.., M, F] = silu(x @ W_il[:, gate tiles]) * (x @ W_il[:, up tiles]) for a [1, 1, K, 2F] tile-pair interleaved
    W_il (bf16 TILE). Output bf16 TILE in `memory_config`."""
    grid = device.compute_with_storage_grid_size()
    assert mm_swiglu_supported(x, w_il, grid), "mm_swiglu: unsupported shapes / layouts (check mm_swiglu_supported)"
    gx, gy = int(grid.x), int(grid.y)
    num_cores = gx * gy

    M, K, N = _dims(x, w_il)
    M_tiles, K_tiles, N_tiles = M // _TILE, K // _TILE, N // _TILE
    M_block, K_block, N_block = _M_BLOCK, K_tiles, _N_BLOCK
    subblock_h, subblock_w = _SUBBLOCK_H, _SUBBLOCK_W
    N_chunks = 1
    N_tiles_per_chunk = N_tiles // N_chunks

    # Data formats / tile sizes (bf16 in / out, fp32 intermediate = the fp32 DEST's datatype).
    in0_tile_size = in1_tile_size = out_tile_size = 2 * _TILE * _TILE
    in2_tile_size = in1_tile_size  # no bias: in2 format = in1 format
    in3_tile_size = in1_tile_size  # no second in0 source: in3 format = in1 format
    interm_tile_size = 4 * _TILE * _TILE

    # Orientation: small input + output DM on RISCV_1 / NOC_1 (preferred_noc_for_dram_write), large input on
    # RISCV_0 / NOC_0 (preferred_noc_for_dram_read); transpose the core grid when M > N.
    small_noc, small_risc = ttnn.NOC.NOC_1, ttnn.DataMovementProcessor.RISCV_1
    large_noc, large_risc = ttnn.NOC.NOC_0, ttnn.DataMovementProcessor.RISCV_0
    transpose = M > N
    in0_noc, in0_risc = (large_noc, large_risc) if transpose else (small_noc, small_risc)
    in1_noc, in1_risc = (small_noc, small_risc) if transpose else (large_noc, large_risc)
    in0_axis = gx if transpose else gy
    in1_axis = gy if transpose else gx

    padded_M_tiles = _round_up(M_tiles, in0_axis)
    padded_K_tiles = _round_up(K_tiles, K_block)
    # fuse_swiglu: partition on gate/up PAIRS (= output tiles).
    out_N_tiles = N_tiles // 2
    padded_out_N_tiles = _round_up(out_N_tiles, in1_axis)
    padded_N_tiles = 2 * padded_out_N_tiles
    N_tiles_per_core = 2 * (padded_out_N_tiles // in1_axis)
    M_tiles_per_core = padded_M_tiles // in0_axis
    K_blocks = padded_K_tiles // K_block
    M_blocks_per_core = _div_up(M_tiles_per_core, M_block)
    N_blocks_per_core = _div_up(N_tiles_per_core, N_block)
    assert N_tiles % 2 == 0 and N_tiles_per_core % 2 == 0 and N_block % 2 == 0
    assert M_block % subblock_h == 0 and N_block % subblock_w == 0 and subblock_h * subblock_w <= 4

    in0_block_num_tiles = M_block * K_block
    in1_block_num_tiles = K_block * N_block
    out_block_num_tiles = M_block * N_block
    in0_cb_num_tiles = 2 * in0_block_num_tiles
    in1_cb_num_tiles = 2 * in1_block_num_tiles
    out_cb_num_tiles = 2 * (out_block_num_tiles // 2)  # SwiGLU: half a block, double buffered
    interm_cb_num_tiles = out_block_num_tiles

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)[:-1] + [N // 2]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )

    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    def rect(x0, y0, x1, y1):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])

    in0_sender_cores = rect(0, 0, gx - 1, 0) if transpose else rect(0, 0, 0, gy - 1)
    in0_receiver_cores = rect(0, 1, gx - 1, gy - 1) if transpose else rect(1, 0, gx - 1, gy - 1)
    in1_sender_cores = rect(0, 0, 0, gy - 1) if transpose else rect(0, 0, gx - 1, 0)
    in1_receiver_cores = rect(1, 0, gx - 1, gy - 1) if transpose else rect(0, 1, gx - 1, gy - 1)

    semaphores = [
        ttnn.SemaphoreDescriptor(id=_SEM_IN0_SENDER, core_ranges=all_cores, initial_value=_INVALID),
        ttnn.SemaphoreDescriptor(id=_SEM_IN0_RECEIVER, core_ranges=all_cores, initial_value=_INVALID),
        ttnn.SemaphoreDescriptor(id=_SEM_IN0_VALID, core_ranges=all_cores, initial_value=_VALID),
        ttnn.SemaphoreDescriptor(id=_SEM_IN1_SENDER, core_ranges=all_cores, initial_value=_INVALID),
        ttnn.SemaphoreDescriptor(id=_SEM_IN1_RECEIVER, core_ranges=all_cores, initial_value=_INVALID),
        ttnn.SemaphoreDescriptor(id=_SEM_IN1_VALID, core_ranges=all_cores, initial_value=_VALID),
    ]

    def cb(idx, n_tiles, dtype, tile_size):
        return ttnn.CBDescriptor(
            total_size=n_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(idx, dtype, tile_size)],
        )

    cbs = [
        cb(_CB_IN0, in0_cb_num_tiles, ttnn.bfloat16, in0_tile_size),
        cb(_CB_IN1, in1_cb_num_tiles, ttnn.bfloat16, in1_tile_size),
        cb(_CB_OUT, out_cb_num_tiles, ttnn.bfloat16, out_tile_size),
        cb(_CB_INTERM, interm_cb_num_tiles, ttnn.float32, interm_tile_size),
        cb(_CB_SCRATCH, out_block_num_tiles, ttnn.bfloat16, out_tile_size),  # mm_swiglu epilogue: G / U tiles
    ]

    defines = [("FUSE_SWIGLU", "1")]
    in0_is_output_writer = 0 if transpose else 1
    in1_is_output_writer = 1 if transpose else 0
    shape_ct = [
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block,
        K_block,
        N_block,
        M_blocks_per_core,
        N_blocks_per_core,
    ]
    x_ta = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    w_ta = list(ttnn.TensorAccessorArgs(w_il).get_compile_time_args())
    out_ta = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    def in0_ct(is_injector):
        return (
            shape_ct
            + [in0_tile_size, out_tile_size, in2_tile_size]
            + [_SEM_IN0_SENDER, _SEM_IN0_RECEIVER, _SEM_IN0_VALID]
            + [in0_is_output_writer, is_injector, N_chunks, N_tiles_per_chunk, in3_tile_size]
            + x_ta
            + out_ta
        )

    def in1_ct(is_injector):
        return (
            shape_ct
            + [in1_tile_size, out_tile_size, in2_tile_size]
            + [_SEM_IN1_SENDER, _SEM_IN1_RECEIVER, _SEM_IN1_VALID]
            + [in1_is_output_writer, is_injector, N_chunks, N_tiles_per_chunk]
            + w_ta
            + out_ta
        )

    compute_ct = [K_blocks, M_block, K_block, N_block, M_blocks_per_core, N_blocks_per_core, subblock_h, subblock_w]

    # Deferred output writes: stride the K block each writer defers to by its core y.
    k_blocks_per_core = _div_up(K_blocks, in1_axis if transpose else in0_axis)
    max_defer_write_k_block = max(min(y * k_blocks_per_core, K_blocks - 1) for y in range(gy))

    x_addr, w_addr, out_addr = x.buffer_address(), w_il.buffer_address(), out.buffer_address()
    rt_in0_sender, rt_in0_receiver = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rt_in1_sender, rt_in1_receiver = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rt_compute = ttnn.RuntimeArgs()

    phys = {}

    def noc_xy(c):
        if c not in phys:
            p = device.worker_core_from_logical_core(ttnn.CoreCoord(c[0], c[1]))
            phys[c] = (int(p.x), int(p.y))
        return phys[c]

    for cy in range(gy):  # corerange_to_cores(row_wise=true) order; the order itself does not matter here
        for cx in range(gx):
            core = (cx, cy)
            in0_idx = cx if transpose else cy
            in1_idx = cy if transpose else cx
            left_core, top_core = (0, cy), (cx, 0)
            in0_order, in0_i = _core_order_for_axis(
                core, transpose, in1_axis, in0_noc == ttnn.NOC.NOC_0, True, top_core if transpose else left_core
            )
            in1_order, in1_i = _core_order_for_axis(
                core, transpose, in0_axis, in1_noc == ttnn.NOC.NOC_0, False, left_core if transpose else top_core
            )
            in0_prev = noc_xy(in0_order[max(in0_i - 1, 0)])
            in0_next = noc_xy(in0_order[min(in0_i + 1, len(in0_order) - 1)])
            in1_prev = noc_xy(in1_order[max(in1_i - 1, 0)])
            in1_next = noc_xy(in1_order[min(in1_i + 1, len(in1_order) - 1)])

            M_start = M_tiles_per_core * in0_idx
            M_end = M_tiles_per_core * (in0_idx + 1)
            N_start = N_tiles_per_core * in1_idx
            N_end = N_tiles_per_core * (in1_idx + 1)
            defer_write_k_block = min(cy * k_blocks_per_core, K_blocks - 1)
            is_in0_sink = 1 if core == in0_order[-1] else 0
            is_in1_sink = 1 if core == in1_order[-1] else 0

            in0_args = [
                x_addr,
                0,  # in2 (bias) addr
                0,  # in3 (second in0 source) addr
                is_in0_sink,
                in0_next[0],
                in0_next[1],
                in0_prev[0],
                in0_prev[1],
                M_start,
                M_end,
                N_start,
                N_end,
                defer_write_k_block,
                max_defer_write_k_block,
                out_addr,
            ]
            (rt_in0_sender if in1_idx == 0 else rt_in0_receiver)[cx][cy] = in0_args

            in1_args = [
                w_addr,
                0,  # in2 (bias) addr
                is_in1_sink,
                in1_next[0],
                in1_next[1],
                in1_prev[0],
                in1_prev[1],
                M_start,
                M_end,
                N_start,
                N_end,
                defer_write_k_block,
                max_defer_write_k_block,
                out_addr,
            ]
            (rt_in1_sender if in0_idx == 0 else rt_in1_receiver)[cx][cy] = in1_args

            rt_compute[cx][cy] = [M_start, M_end, N_start, N_end]

    def dm_kernel(src, cores, ct, rt, risc, noc):
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, src),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            defines=defines,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc),
        )

    kernels = [
        dm_kernel("dm_in0_sender.cpp", in0_sender_cores, in0_ct(1), rt_in0_sender, in0_risc, in0_noc),
        dm_kernel("dm_in0_sender.cpp", in0_receiver_cores, in0_ct(0), rt_in0_receiver, in0_risc, in0_noc),
        dm_kernel("dm_in1_sender_out.cpp", in1_sender_cores, in1_ct(1), rt_in1_sender, in1_risc, in1_noc),
        dm_kernel("dm_in1_sender_out.cpp", in1_receiver_cores, in1_ct(0), rt_in1_receiver, in1_risc, in1_noc),
        ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "compute.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            defines=defines,
            runtime_args=rt_compute,
            # As the factory's ComputeConfig: fidelity / approx / fp32 DEST from the compute kernel config, half-sync
            # DEST, every CB unpacked to SrcA (unpack_to_dest_mode Default). No MM_THROTTLE define: the
            # reference's throttle level is the default (none).
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=compute_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        ),
    ]
    assert num_cores == gx * gy
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    return ttnn.generic_op([x, w_il, out], program)
