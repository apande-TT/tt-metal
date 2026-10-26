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
# the stock in0 reader with its block read remapped to a heads-layout in0 (matmul(..., in0_heads=))
_DM_IN0_HEADS = str(_KDIR / "mm_in0_heads.cpp")
_DM_IN1 = _DM_DIR + "dm_in1_sender_out.cpp"
# the stock data-movement kernels with deferred output writes on the injector cores too (SWIGLU_OVERLAP)
_DM_IN0_DEFER = str(_KDIR / "mm_dm_in0.cpp")
_DM_IN1_DEFER = str(_KDIR / "mm_dm_in1_out.cpp")

_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_INVALID, _VALID = 0, 1

# Output-stage grouping (mm_compute.cpp): tiles a DEST acquire in the plain epilogue (fp32 DEST half: 4 tiles), and
# gate/up pairs a DEST acquire in the SwiGLU epilogue.
EPI_TILES = 4
SWIGLU_PAIRS = 2
# The SwiGLU epilogue's SFPU pass (silu * up) runs on the PACK thread (kernels/mm_compute.cpp pack_silu_mul) while MATH
# copies the next pair group's partials into the other DEST half.
SWIGLU_PACK_SFPU = True
# ... with the exp polynomial's constants held in registers across the rows (mm_compute.cpp
# calculate_silu_mul_hoisted: 42 instead of 52 SFPU instructions a row, the same arithmetic)
SWIGLU_HOIST = True
# Skip the padding of the core ranges (the last core row's M rows past M_tiles, the last core column's N tiles past
# N_tiles): no matmul subblocks or epilogue work for tiles the writer never stores (identical real tiles; less FPU /
# SFPU energy, which is what the power-capped prefill and encode run on).
SKIP_PADDING = True
# The SwiGLU epilogue of each output block runs during the NEXT block's matmul, one gate/up pair between its subblocks
# (mm_compute.cpp SWIGLU_OVERLAP: the PACK thread's SFPU pass beside the FPU), drained by K step EPI_DONE_FRAC * K blocks;
# the intermediate CB holds two blocks and every writer defers its block write to that K step or later. Only where the
# doubled intermediate fits L1 (SWIGLU_OVERLAP_L1_BYTES of circular buffers).
SWIGLU_OVERLAP = True
# (0.5: 416-row gate/up 9205 us; 0.75: 9070; 0.9: 9210 a call)
EPI_DONE_FRAC = 0.75
SWIGLU_OVERLAP_L1_BYTES = 1_240_000
# ... and with an L1 input or product (the vision MLP: its L1 residual, input and product sit beside the CBs) where the
# circular buffers, doubled intermediate included, stay within this many bytes (the non-overlapped 6 x 14-tile vision
# blocks' 870 KB fit; 2 x 14-tile blocks with two intermediates take 567 KB)
SWIGLU_OVERLAP_L1_IO_BYTES = 880_000
# SWIGLU_DEEP_K: under SWIGLU_OVERLAP the PACK thread runs a pair's SFPU pass (~1344 cycles) between two 1 x 4
# subblocks, but an 8-deep K block makes a subblock only ~1024 cycles at HiFi2, so MATH waits for its DEST half every
# pair (~0.7 us a pair exposed). 16-deep K blocks give the pass a whole subblock; M blocks of at most
# SWIGLU_DEEP_M_MAX tiles and SWIGLU_DEEP_N_BLOCK-tile N blocks keep the doubled intermediate within
# SWIGLU_OVERLAP_L1_BYTES. The fp32 partials are then summed in 16-tile K runs instead of 8 (the epilogue reads them as
# TF32 either way): not bit-identical.
SWIGLU_DEEP_K = 16
SWIGLU_DEEP_N_BLOCK = 8
SWIGLU_DEEP_M_MAX = 6
# ... with the epilogue copying the fp32 partials straight to DEST (UnpackToDestFp32) instead of through SrcA as TF32
# (off: 416-row gate/up 8800 vs 8590 us). Callers opt layers in with matmul(..., deep_k_ok=) (the image prompt's
# prefill-hidden PCC drops below its 0.995 floor with every text layer deep-K: 0.994431).
SWIGLU_FP32_EPILOGUE = False


# minimal_matmul splits M along the grid's x axis and N along y when M > N, else the other way round. ORIENT_MIN_WORK
# takes the other orientation where it hands each core at least ORIENT_MIN_GAIN fewer output tiles (ceil splits,
# SwiGLU pairs kept whole; the vision MLP / attention: 64 M tiles against 32-96 N tiles) -- the same tiles, each
# accumulated over K in the same order. Callers that size blocks to the per-core ranges use transposed() too.
ORIENT_MIN_WORK = True
ORIENT_MIN_GAIN = 0.05

# The in0 / in1 blocks travel the grid as a forwarding chain (each core receives a block from the previous core of its
# row / column and writes it on to the next). Stock: store-and-forward of the whole K block, so a chain's last core
# gets its first block only after every hop has moved all of it -- the vision QKV's 144 KB weight blocks down 9 hops.
# CHUNKED_FWD: each block moves in chunks of about FWD_CHUNK_BYTES (whole rows: M rows of in0, K rows of in1), every
# core passing a chunk on as soon as it lands (kernels/mm_dm_in0.cpp, mm_dm_in1_out.cpp CHUNKED_FWD). The CB contents
# the compute reads are the same tiles in the same slots, so the products are unchanged. Only where the activation has
# at most FWD_MAX_ROWS tile rows: the short vision / 128-row matmuls, whose few K blocks the chain's fill time weighs on
# (vision QKV 132 -> 125, o_proj 65 -> 55, down 123 -> 107 us), not the 416-row prompt's long ones (+0.3-3.5%).
CHUNKED_FWD = True
FWD_CHUNK_BYTES = 16 * 1024
FWD_MAX_ROWS = 128


def transposed(m_tiles, n_tiles, gx, gy, swiglu=False):
    """Whether minimal_matmul (this module's program) splits M along the grid's x axis (and N along y)."""
    stock = m_tiles > n_tiles
    if not ORIENT_MIN_WORK:
        return stock

    def work(t):
        mp, np_ = (gx, gy) if t else (gy, gx)
        n = 2 * -(-(n_tiles // 2) // np_) if swiglu else -(-n_tiles // np_)
        return -(-m_tiles // mp) * n

    return (not stock) if work(not stock) <= (1 - ORIENT_MIN_GAIN) * work(stock) else stock


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


def default_config(device, compute_kernel_config=None, x=None, w=None):
    """The blocking ttnn.experimental.minimal_matmul runs when given no config (minimal_matmul_program_factory.cpp
    determine_default_block_sizes): 8 x 8 x 8 blocks on the device's compute grid, 2 x 2 subblocks with fp32 DEST
    (else 2 x 4 when N >= M, 4 x 2 when M > N; x, w needed then)."""
    sub_h, sub_w = 2, 2
    if compute_kernel_config is not None and not compute_kernel_config.fp32_dest_acc_en:
        xs = [int(d) for d in x.padded_shape]
        m = 1
        for d in xs[:-1]:
            m *= d
        sub_h, sub_w = (2, 4) if int(w.padded_shape[-1]) >= m else (4, 2)
    return ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )


def _in0_view(x, in0_heads):
    """x's padded shape as the matmul sees it: (B, 1, S, NH * HD) for a heads-layout (B, NH, S, HD) in0."""
    xs = [int(d) for d in x.padded_shape]
    if in0_heads is None:
        return xs
    b, nh, sq, hd = xs
    return [b, 1, sq, nh * hd]


def supports_heads(x, w) -> bool:
    """Whether matmul(x, w, in0_heads=...) can read x, a tiled interleaved (B, NH, S, HD) attention output, as its
    head concat (B, 1, S, NH * HD)."""
    if x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or len(x.padded_shape) != 4:
        return False
    b, nh, sq, hd = (int(d) for d in x.padded_shape)
    return sq % ttnn.TILE_SIZE == 0 and hd % ttnn.TILE_SIZE == 0 and nh * hd == int(w.padded_shape[-2])


def supports_residual(x, w, residual, dtype, in0_heads=None) -> bool:
    """Whether matmul(..., residual=residual) can fold the fp32 residual add in (a fp32 output of the residual's shape)."""
    if dtype != ttnn.float32 or residual is None or residual.dtype != ttnn.float32:
        return False
    if residual.layout != ttnn.TILE_LAYOUT or residual.is_sharded():
        return False
    out = _in0_view(x, in0_heads)[:-1] + [int(w.padded_shape[-1])]
    return [int(d) for d in residual.padded_shape] == out


def matmul(
    x,
    w,
    config,
    compute_kernel_config,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
    fuse_swiglu=False,
    residual=None,
    in0_heads=None,
    stock_orient=False,
    swapped=False,
    orient=None,
    deep_k_ok=True,
    heads_out=None,
    n_seg=None,
):
    """n_seg (tiles, under each core's N range): N blocks of config.N_block_size within two segments of the core's
    range, [0, n_seg) and the rest, walked K-forward and K-backward -- the K-block summation order every tile gets from
    blocks [n_seg | rest] (the readers reverse K on every other N block to reuse in0), so the smaller blocks are
    bit-identical to that split (kernels/mm_dm_in0.cpp, mm_dm_in1_out.cpp, mm_compute.cpp N_SEG_TILES).
    residual (supports_residual): return residual + x @ w, the add folded into the output stage (mm_compute.cpp
    residual_add_block: the stock data-movement kernels' FUSE_TERNARY read brings the residual block in on the RISC that
    does not write the output), bit-identical to ttnn.add(residual, x @ w) on the fp32 output.
    in0_heads (True, supports_heads): x is an attention output (B, NH, S, HD) multiplied as its head concat
    (B, 1, S, NH * HD) -- the in0 reader (kernels/mm_in0_heads.cpp) fetches each tile where nlp_concat_heads would
    have copied it from, so the concat program is not needed; same tiles, same matmul.
    swapped (w = the (N x K) TRANSPOSED weight): x @ w^T computed as the program's (w @ x^T)^T -- in0 = w, in1 = x's
    tiles read in transposed positions and unpacked transposed, each output tile transposed on its way out
    (kernels/mm_dataflow_common.hpp IN1_TRANSPOSED / OUT_TRANSPOSED, mm_compute.cpp MM_IN1_TRANSPOSE). The weight
    lands in srcB and the activation in srcA: a block-float8 weight's 7-bit significands fit srcB's one 7-bit
    multiplier pass, so HiFi2 (srcA's two 5-bit passes x srcB's first) multiplies the full operands -- the products
    HiFi4 forms with the weight in srcA, at half the multiplier passes.
    orient (True / False): force the grid split (True: M along the grid's x axis).
    heads_out ((nq, nkv, qk_memory_config, v_memory_config)): return (q, k, v) in nlp_create_qkv_heads' layouts
    ((B, nq | nkv, S, HD), HD = N / (nq + 2 nkv)) instead of the (B, 1, S, N) product -- the output writer drops each
    tile where that split would have copied it (kernels/mm_dataflow_common.hpp OUT_HEADS), so neither the split
    program nor a full-width product is needed; same tiles, same matmul."""
    assert residual is None or not fuse_swiglu
    assert not swapped or (residual is None and not fuse_swiglu and in0_heads is None)
    assert heads_out is None or (residual is None and not fuse_swiglu and not swapped)
    device = x.device()
    grid = config.compute_with_storage_grid_size
    gx, gy = int(grid.x), int(grid.y)
    fidelity = compute_kernel_config.math_fidelity
    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    approx = bool(compute_kernel_config.math_approx_mode)

    xs = _in0_view(x, in0_heads)
    K = xs[-1]
    M = 1
    for d in xs[:-1]:
        M *= d
    N = int(w.padded_shape[-1])
    in0, in1 = x, w
    if swapped:
        assert int(w.padded_shape[-1]) == K
        in0, in1 = w, x
        M, N = int(w.padded_shape[-2]), M
    T = ttnn.TILE_SIZE
    M_tiles, K_tiles, N_tiles = M // T, K // T, N // T

    M_block, K_block, N_block = int(config.M_block_size), int(config.K_block_size), int(config.N_block_size)
    sub_h, sub_w = int(config.subblock_h), int(config.subblock_w)

    # stock_orient: the stock M > N split (the vision MLP down measured no better transposed)
    transpose = M_tiles > N_tiles if stock_orient else transposed(M_tiles, N_tiles, gx, gy, fuse_swiglu)
    if orient is not None:
        transpose = bool(orient)
    dram_read, dram_write = ttnn.NOC.NOC_0, ttnn.NOC.NOC_1  # Blackhole preferred NOCs
    small_noc, small_risc = dram_write, ttnn.DataMovementProcessor.RISCV_1
    large_noc, large_risc = dram_read, ttnn.DataMovementProcessor.RISCV_0
    in0_noc, in0_risc = (large_noc, large_risc) if transpose else (small_noc, small_risc)
    in1_noc, in1_risc = (small_noc, small_risc) if transpose else (large_noc, large_risc)
    in0_par = gx if transpose else gy
    in1_par = gy if transpose else gx

    padded_M = _round_up(M_tiles, in0_par)
    deep_k = False
    if (
        fuse_swiglu
        and deep_k_ok
        and SWIGLU_DEEP_K
        and SWIGLU_OVERLAP
        and SWIGLU_PACK_SFPU
        and fp32_acc
        and in0_heads is None
        and (sub_h, sub_w) == (1, 4)
        and K_tiles % SWIGLU_DEEP_K == 0
        and K_tiles // SWIGLU_DEEP_K >= 2
        and (2 * (_round_up(N_tiles // 2, in1_par) // in1_par)) % SWIGLU_DEEP_N_BLOCK == 0
        and x.memory_config().buffer_type == ttnn.BufferType.DRAM
        and memory_config.buffer_type == ttnn.BufferType.DRAM
    ):
        m_core = padded_M // in0_par
        mb = min(SWIGLU_DEEP_M_MAX, m_core)
        # the M block that tiles the core's M range with the least padding (the larger on a tie)
        M_block = min(range(mb, max(1, mb // 2) - 1, -1), key=lambda b: -(-m_core // b) * b - m_core)
        K_block, N_block = SWIGLU_DEEP_K, SWIGLU_DEEP_N_BLOCK
        deep_k = True
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
    if n_seg:
        assert not deep_k and in0_heads is None and not swapped and 0 < n_seg < N_per_core
        assert not fuse_swiglu or n_seg % 2 == 0
        N_blocks_per_core = -(-n_seg // N_block) + -(-(N_per_core - n_seg) // N_block)

    in0_tile, in1_tile = _TILE_BYTES[in0.dtype], _TILE_BYTES[in1.dtype]
    out_tile = _TILE_BYTES[dtype]
    interm_dtype = ttnn.float32 if fp32_acc else ttnn.bfloat16
    interm_tile = _TILE_BYTES[interm_dtype]

    out_block = M_block * N_block
    out_written = out_block // 2 if fuse_swiglu else out_block

    out_shape = xs[:-1] + [M if swapped else N // 2 if fuse_swiglu else N]
    if heads_out is not None:
        nq, nkv, qk_mem, v_mem = heads_out
        S = int(xs[-2])
        B = M // S
        HD = N // (nq + 2 * nkv)
        assert S % T == 0 and HD % T == 0 and HD * (nq + 2 * nkv) == N
        outs = [
            ttnn.allocate_tensor_on_device(ttnn.Shape([B, h, S, HD]), dtype, ttnn.TILE_LAYOUT, device, mem)
            for h, mem in ((nq, qk_mem), (nkv, qk_mem), (nkv, v_mem))
        ]
    else:
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, ttnn.TILE_LAYOUT, device, memory_config)]
    out = outs[0]

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
    cb_bytes = M_block * K_block * 2 * in0_tile + K_block * N_block * 2 * in1_tile + out_written * out_bufs * out_tile
    overlap = (
        SWIGLU_OVERLAP
        and fuse_swiglu
        and SWIGLU_PACK_SFPU
        and fp32_acc
        and in0_heads is None
        and K_blocks >= 2
        and cb_bytes + 2 * out_block * interm_tile
        <= (
            SWIGLU_OVERLAP_L1_BYTES
            if x.memory_config().buffer_type == ttnn.BufferType.DRAM
            and memory_config.buffer_type == ttnn.BufferType.DRAM
            else SWIGLU_OVERLAP_L1_IO_BYTES
        )
    )
    epi_done_k = min(max(1, int(K_blocks * EPI_DONE_FRAC)), K_blocks - 1) if overlap else 0
    cbs = [
        cb(0, M_block * K_block * 2, in0_tile, in0.dtype),
        cb(1, K_block * N_block * 2, in1_tile, in1.dtype),
        cb(2, out_written * out_bufs, out_tile, dtype),
        cb(3, out_block * (2 if overlap else 1), interm_tile, interm_dtype),
    ]
    if residual is not None:
        # the stock readers' FUSE_TERNARY CBs: the residual block ("a", unpacked to DEST unrounded) and the broadcast "b"
        # row they read with it (the residual's first row here; the compute pops it unused)
        cbs += [cb(5, out_block, _TILE_BYTES[residual.dtype], residual.dtype)]
        cbs += [cb(6, N_block, _TILE_BYTES[residual.dtype], residual.dtype)]

    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    out_ct = [a for t in outs for a in acc(t)]
    res_ct = [] if residual is None else acc(residual) + acc(residual)
    head = [M_tiles, padded_M, K_tiles, padded_K, N_tiles, padded_N, M_block, K_block, N_block]
    head += [M_blocks_per_core, N_blocks_per_core]
    in0_ct = lambda injector: (
        head
        + [in0_tile, out_tile, in1_tile, S_IN0_SEND, S_IN0_RECV, S_IN0_VALID, int(not transpose), injector]
        + [len(outs), N_tiles]
        + [in1_tile]
        + acc(in0)
        + out_ct
        + res_ct
    )
    in1_ct = lambda injector: (
        head
        + [in1_tile, out_tile, in1_tile, S_IN1_SEND, S_IN1_RECV, S_IN1_VALID, int(transpose), injector]
        + [len(outs), N_tiles]
        + acc(in1)
        + out_ct
        + res_ct
    )

    skip_padding = SKIP_PADDING
    k_blocks_per_core = -(-K_blocks // (in1_par if transpose else in0_par))
    max_defer = max(max(min(y * k_blocks_per_core, K_blocks - 1), epi_done_k) for y in range(gy))
    phys = {}

    def pxy(c):
        if c not in phys:
            p = device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
            phys[c] = (int(p.x), int(p.y))
        return phys[c]

    x_addr, w_addr = in0.buffer_address(), in1.buffer_address()
    o_addrs = [t.buffer_address() for t in outs]
    # FUSE_TERNARY runtime args (before the output address): a, b (broadcast row), broadcast flag
    tern = [] if residual is None else [residual.buffer_address(), residual.buffer_address(), 1]
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
            defer = max(min(cy * k_blocks_per_core, K_blocks - 1), epi_done_k)
            a0 = [
                x_addr,
                0,
                0,
                int(core == o0[-1]),
                *pxy(next0),
                *pxy(prev0),
                m0,
                m1,
                n0,
                n1,
                defer,
                max_defer,
                *tern,
                *o_addrs,
            ]
            a1 = [w_addr, 0, int(core == o1[-1]), *pxy(next1), *pxy(prev1), m0, m1, n0, n1, defer, max_defer, *tern]
            a1 += o_addrs
            (rt_in0_s if in1_idx == 0 else rt_in0_r).append((core, a0))
            (rt_in1_s if in0_idx == 0 else rt_in1_r).append((core, a1))
            rt_c.append(
                (
                    core,
                    [m0, m1, n0, n1] + ([M_tiles, N_tiles] if skip_padding else []) + ([epi_done_k] if overlap else []),
                )
            )

    defines = [("FUSE_SWIGLU", "1")] if fuse_swiglu else []
    in0_src, in0_defines = _DM_IN0, []
    if in0_heads is not None:
        nh, sq, hd = [int(d) for d in x.padded_shape][1:]
        in0_src = _DM_IN0_HEADS
        in0_defines = [("IN0_NH", str(nh)), ("IN0_ST", str(sq // T)), ("IN0_HDT", str(hd // T))]
    if residual is not None:
        defines += [("FUSE_TERNARY", "1")]
    cdefines = [d for d in defines if d[0] != "FUSE_TERNARY"] + [
        ("EPI_TILES", str(EPI_TILES if N_block % EPI_TILES == 0 else 1))
    ]
    cdefines += [("SWIGLU_PAIRS", str(SWIGLU_PAIRS))]
    if fuse_swiglu and SWIGLU_PACK_SFPU:
        cdefines += [("SWIGLU_PACK_SFPU", "1")]
        if SWIGLU_HOIST:
            cdefines += [("SWIGLU_HOIST", "1")]
    if skip_padding:
        cdefines += [("SKIP_PADDING", "1")]
    in1_src = _DM_IN1
    if overlap:
        cdefines += [("SWIGLU_OVERLAP", "1")]
        in0_src, in1_src = (_DM_IN0_DEFER, _DM_IN1_DEFER)
        defines = defines + [("DEFER_INJECTOR", "1")]
    if residual is not None:
        cdefines += [("FUSE_RESIDUAL", "1")]
    if heads_out is not None:
        defines = defines + [
            ("OUT_HEADS", "1"),
            ("OUT_HEADS_NQ", str(nq)),
            ("OUT_HEADS_NKV", str(nkv)),
            ("OUT_HEADS_ST", str(S // T)),
            ("OUT_HEADS_HDT", str(HD // T)),
            ("OUT_HEADS_NT", str(N_tiles)),
        ]
        in0_src, in1_src = (_DM_IN0_DEFER, _DM_IN1_DEFER)  # (the stock kernels unless DEFER_INJECTOR)
    if swapped:
        cdefines += [("MM_IN1_TRANSPOSE", "1")]
        defines = defines + [("IN1_TRANSPOSED", "1"), ("OUT_TRANSPOSED", "1")]
        in0_src, in1_src = (_DM_IN0_DEFER, _DM_IN1_DEFER)  # (the stock kernels unless DEFER_INJECTOR)
    if n_seg:
        cdefines += [("N_SEG_TILES", str(n_seg))]
        defines = defines + [("N_SEG_TILES", str(n_seg))]
        in0_src, in1_src = (_DM_IN0_DEFER, _DM_IN1_DEFER)  # (the stock kernels unless DEFER_INJECTOR)
    if CHUNKED_FWD and (N_tiles if swapped else M_tiles) <= FWD_MAX_ROWS:
        # (in0_heads: the copy's chunk reads remap the heads layout themselves, IN0_NH / IN0_ST / IN0_HDT)
        # (at most 15 chunks a block: the injector tags chunk c's reads with NOC transaction id c + 1)
        rows0 = max(1, FWD_CHUNK_BYTES // (K_block * in0_tile), -(-M_block // 15))
        rows1 = max(1, FWD_CHUNK_BYTES // (N_block * in1_tile), -(-K_block // 15))
        defines = defines + [("CHUNKED_FWD", "1"), ("FWD_IN0_ROWS", str(rows0)), ("FWD_IN1_ROWS", str(rows1))]
        in0_src, in1_src = (_DM_IN0_DEFER, _DM_IN1_DEFER)

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
    if residual is not None:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[5] = ttnn.UnpackToDestMode.UnpackToDestFp32  # the residual, as binary_ng unpacks a fp32 add's operands
        ccfg.unpack_to_dest_mode = modes
    if deep_k and SWIGLU_FP32_EPILOGUE:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[3] = ttnn.UnpackToDestMode.UnpackToDestFp32  # the SwiGLU reads the fp32 partials whole, not as TF32
        ccfg.unpack_to_dest_mode = modes
    c_ct = [K_blocks, M_block, K_block, N_block, M_blocks_per_core, N_blocks_per_core, sub_h, sub_w]
    kernels = [
        kern(in0_src, in0_senders, in0_ct(1), rt_in0_s, dm(in0_risc, in0_noc), defines + in0_defines),
        kern(in0_src, in0_receivers, in0_ct(0), rt_in0_r, dm(in0_risc, in0_noc), defines + in0_defines),
        kern(in1_src, in1_senders, in1_ct(1), rt_in1_s, dm(in1_risc, in1_noc), defines),
        kern(in1_src, in1_receivers, in1_ct(0), rt_in1_r, dm(in1_risc, in1_noc), defines),
        kern(_COMPUTE, core_grid, c_ct, rt_c, ccfg, cdefines),
    ]
    ios = [in0, in1, *outs] if residual is None else [x, w, residual, out]
    ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    return tuple(outs) if heads_out is not None else out
