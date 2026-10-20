# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.minimal_matmul as a generic_op program whose compute kernel the model owns (kernels/mm_compute.cpp).

The program is the stock op's (minimal_matmul_program_factory.cpp, plain path: no bias / ternary / fused activation /
fused CCL): the same core grid split, block sizes, semaphores, circular buffers and runtime arguments, and the stock
data-movement kernels themselves (dm_in0_sender.cpp, dm_in1_sender_out.cpp). Only the compute kernel differs, in its
output stage (see mm_compute.cpp): the fused SwiGLU runs silu and the up multiply as one SFPU pass with its inits once
a block (416-row text gate/up 9844 -> 9722 us, vision 245 -> 234 us), bit-identical to the stock op.

Call matmul(x, w, config, compute_kernel_config, ...) exactly as ttnn.experimental.minimal_matmul (config required)."""
from __future__ import annotations

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_COMPUTE = str(_KDIR / "mm_compute.cpp")
_DM_DIR = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/"
_DM_IN0 = _DM_DIR + "dm_in0_sender.cpp"
_DM_IN1 = _DM_DIR + "dm_in1_sender_out.cpp"

_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_INVALID, _VALID = 0, 1

# Output-stage grouping (mm_compute.cpp): tiles a DEST acquire in the plain epilogue (fp32 DEST half: 4 tiles), and
# gate/up pairs a DEST acquire in the SwiGLU epilogue.
EPI_TILES = 4
SWIGLU_PAIRS = 2


def _round_up(a, b):
    return -(-a // b) * b


def _core_order(core, transpose, axis_len, noc, axis_is_x, initial):
    """minimal_matmul_program_factory.cpp build_core_order_for_axis: the forwarding chain along one grid axis."""
    order = [initial]
    if transpose:
        cur = core[1] if axis_is_x else core[0]
    else:
        cur = core[0] if axis_is_x else core[1]
    increasing = noc == ttnn.NOC.NOC_0
    idx = 0
    for w in range(1, axis_len):
        x, y = core
        v = w if increasing else axis_len - w
        if transpose:
            if axis_is_x:
                y = v
            else:
                x = v
        else:
            if axis_is_x:
                x = v
            else:
                y = v
        if v == cur:
            idx = w
        order.append((x, y))
    return order, idx


def supported(x, w, compute_kernel_config=None, fuse_swiglu=False) -> bool:
    """Shapes / layouts the program covers (else call the stock op); the SwiGLU epilogue replays fp32-DEST math only."""
    if fuse_swiglu and not (compute_kernel_config is not None and compute_kernel_config.fp32_dest_acc_en):
        return False
    if x.layout != ttnn.TILE_LAYOUT or w.layout != ttnn.TILE_LAYOUT or x.is_sharded() or w.is_sharded():
        return False
    if x.dtype not in _TILE_BYTES or w.dtype not in _TILE_BYTES:
        return False
    xs, ws = [int(d) for d in x.padded_shape], [int(d) for d in w.padded_shape]
    return xs[-1] == ws[-2] and all(d == 1 for d in ws[:-2])


def matmul(
    x, w, config, compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, fuse_swiglu=False
):
    device = x.device()
    grid = config.compute_with_storage_grid_size
    gx, gy = int(grid.x), int(grid.y)
    fidelity = compute_kernel_config.math_fidelity
    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    approx = bool(compute_kernel_config.math_approx_mode)

    xs = [int(d) for d in x.padded_shape]
    K = xs[-1]
    M = 1
    for d in xs[:-1]:
        M *= d
    N = int(w.padded_shape[-1])
    T = ttnn.TILE_SIZE
    M_tiles, K_tiles, N_tiles = M // T, K // T, N // T

    M_block, K_block, N_block = int(config.M_block_size), int(config.K_block_size), int(config.N_block_size)
    sub_h, sub_w = int(config.subblock_h), int(config.subblock_w)

    transpose = M > N
    dram_read, dram_write = ttnn.NOC.NOC_0, ttnn.NOC.NOC_1  # Blackhole preferred NOCs
    small_noc, small_risc = dram_write, ttnn.DataMovementProcessor.RISCV_1
    large_noc, large_risc = dram_read, ttnn.DataMovementProcessor.RISCV_0
    in0_noc, in0_risc = (large_noc, large_risc) if transpose else (small_noc, small_risc)
    in1_noc, in1_risc = (small_noc, small_risc) if transpose else (large_noc, large_risc)
    in0_par = gx if transpose else gy
    in1_par = gy if transpose else gx

    padded_M = _round_up(M_tiles, in0_par)
    padded_K = _round_up(K_tiles, K_block)
    if fuse_swiglu:
        out_N = N_tiles // 2
        padded_out_N = _round_up(out_N, in1_par)
        padded_N = 2 * padded_out_N
        N_per_core = 2 * (padded_out_N // in1_par)
        assert N_tiles % 2 == 0 and N_per_core % 2 == 0 and N_block % 2 == 0
    else:
        padded_N = _round_up(N_tiles, in1_par)
        N_per_core = padded_N // in1_par
    M_per_core = padded_M // in0_par
    K_blocks = padded_K // K_block
    M_blocks_per_core = -(-M_per_core // M_block)
    N_blocks_per_core = -(-N_per_core // N_block)

    in0_tile, in1_tile = _TILE_BYTES[x.dtype], _TILE_BYTES[w.dtype]
    out_tile = _TILE_BYTES[dtype]
    interm_dtype = ttnn.float32 if fp32_acc else ttnn.bfloat16
    interm_tile = _TILE_BYTES[interm_dtype]

    out_block = M_block * N_block
    out_written = out_block // 2 if fuse_swiglu else out_block

    out_shape = xs[:-1] + [N // 2 if fuse_swiglu else N]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, ttnn.TILE_LAYOUT, device, memory_config)

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rng = lambda a, b, c, d: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d))])
    if transpose:
        in0_senders, in0_receivers = rng(0, 0, gx - 1, 0), rng(0, 1, gx - 1, gy - 1)
        in1_senders, in1_receivers = rng(0, 0, 0, gy - 1), rng(1, 0, gx - 1, gy - 1)
    else:
        in0_senders, in0_receivers = rng(0, 0, 0, gy - 1), rng(1, 0, gx - 1, gy - 1)
        in1_senders, in1_receivers = rng(0, 0, gx - 1, 0), rng(0, 1, gx - 1, gy - 1)

    S_IN0_SEND, S_IN0_RECV, S_IN0_VALID, S_IN1_SEND, S_IN1_RECV, S_IN1_VALID = range(6)
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=core_grid, initial_value=v)
        for i, v in enumerate((_INVALID, _INVALID, _VALID, _INVALID, _INVALID, _VALID))
    ]

    def cb(index, tiles, page, fmt):
        f = ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)
        return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=core_grid, format_descriptors=[f])

    # ONE output block buffer: the writer drains block i at K step `defer` (<= K_blocks - 1) of block i + 1, before it
    # reads that K block, so compute -- which packs block i + 1 only after its last K block -- never waits on it. (The
    # vision MLP's 7 x 12-tile SwiGLU block: 86 KB less static L1, which the decode all_reduce's scratch needs.)
    out_bufs = 1
    cbs = [
        cb(0, M_block * K_block * 2, in0_tile, x.dtype),
        cb(1, K_block * N_block * 2, in1_tile, w.dtype),
        cb(2, out_written * out_bufs, out_tile, dtype),
        cb(3, out_block, interm_tile, interm_dtype),
    ]

    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    head = [M_tiles, padded_M, K_tiles, padded_K, N_tiles, padded_N, M_block, K_block, N_block]
    head += [M_blocks_per_core, N_blocks_per_core]
    in0_ct = lambda injector: (
        head
        + [in0_tile, out_tile, in1_tile, S_IN0_SEND, S_IN0_RECV, S_IN0_VALID, int(not transpose), injector, 1, N_tiles]
        + [in1_tile]
        + acc(x)
        + acc(out)
    )
    in1_ct = lambda injector: (
        head
        + [in1_tile, out_tile, in1_tile, S_IN1_SEND, S_IN1_RECV, S_IN1_VALID, int(transpose), injector, 1, N_tiles]
        + acc(w)
        + acc(out)
    )

    k_blocks_per_core = -(-K_blocks // (in1_par if transpose else in0_par))
    max_defer = max(min(y * k_blocks_per_core, K_blocks - 1) for y in range(gy))
    phys = {}

    def pxy(c):
        if c not in phys:
            p = device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
            phys[c] = (int(p.x), int(p.y))
        return phys[c]

    x_addr, w_addr, o_addr = x.buffer_address(), w.buffer_address(), out.buffer_address()
    rt_in0_s, rt_in0_r, rt_in1_s, rt_in1_r, rt_c = [], [], [], [], []
    for cy in range(gy):
        for cx in range(gx):
            core = (cx, cy)
            in0_idx = cx if transpose else cy
            in1_idx = cy if transpose else cx
            left, top = (0, cy), (cx, 0)
            o0, i0 = _core_order(core, transpose, in1_par, in0_noc, True, top if transpose else left)
            o1, i1 = _core_order(core, transpose, in0_par, in1_noc, False, left if transpose else top)
            prev0, next0 = o0[max(i0 - 1, 0)], o0[min(i0 + 1, len(o0) - 1)]
            prev1, next1 = o1[max(i1 - 1, 0)], o1[min(i1 + 1, len(o1) - 1)]
            m0, m1 = M_per_core * in0_idx, M_per_core * (in0_idx + 1)
            n0, n1 = N_per_core * in1_idx, N_per_core * (in1_idx + 1)
            defer = min(cy * k_blocks_per_core, K_blocks - 1)
            a0 = [x_addr, 0, 0, int(core == o0[-1]), *pxy(next0), *pxy(prev0), m0, m1, n0, n1, defer, max_defer, o_addr]
            a1 = [w_addr, 0, int(core == o1[-1]), *pxy(next1), *pxy(prev1), m0, m1, n0, n1, defer, max_defer, o_addr]
            (rt_in0_s if in1_idx == 0 else rt_in0_r).append((core, a0))
            (rt_in1_s if in0_idx == 0 else rt_in1_r).append((core, a1))
            rt_c.append((core, [m0, m1, n0, n1]))

    defines = [("FUSE_SWIGLU", "1")] if fuse_swiglu else []
    cdefines = defines + [("EPI_TILES", str(EPI_TILES if N_block % EPI_TILES == 0 else 1))]
    cdefines += [("SWIGLU_PAIRS", str(SWIGLU_PAIRS))]

    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)
    kern = lambda src, cores, ct, rt, cfg, d: ttnn.KernelDescriptor(
        kernel_source=src,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=cfg,
        defines=d,
    )
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = fidelity
    ccfg.fp32_dest_acc_en = fp32_acc
    ccfg.math_approx_mode = approx
    ccfg.dst_full_sync_en = False
    c_ct = [K_blocks, M_block, K_block, N_block, M_blocks_per_core, N_blocks_per_core, sub_h, sub_w]
    kernels = [
        kern(_DM_IN0, in0_senders, in0_ct(1), rt_in0_s, dm(in0_risc, in0_noc), defines),
        kern(_DM_IN0, in0_receivers, in0_ct(0), rt_in0_r, dm(in0_risc, in0_noc), defines),
        kern(_DM_IN1, in1_senders, in1_ct(1), rt_in1_s, dm(in1_risc, in1_noc), defines),
        kern(_DM_IN1, in1_receivers, in1_ct(0), rt_in1_r, dm(in1_risc, in1_noc), defines),
        kern(_COMPUTE, core_grid, c_ct, rt_c, ccfg, cdefines),
    ]
    ttnn.generic_op([x, w, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    return out
