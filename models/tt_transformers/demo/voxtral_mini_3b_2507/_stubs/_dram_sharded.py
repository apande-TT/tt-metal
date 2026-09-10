# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared decode-regime projection: weight width-sharded across the DRAM banks.

NOT a graduated component -- `bringup_status.json` drives the inventory, so this file is a helper
the stubs load as a sibling, not a stub of its own.

WHY.  At the decode shape a projection carries one tile-row of activation against a weight of tens
of megabytes, so its cost is entirely "how fast can the weight be read".  With the weight
DRAM-INTERLEAVED every core pulls tiles from every bank over the NoC and the read is spread thin;
width-sharding it across the banks and driving it with
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` pins each worker to the bank slice it
consumes.  That is the variant the decode regime exists for (GUIDELINES/08 section 2).

WHY A MIRROR AND NOT A REPLACEMENT.  Prefill hands the same weight an activation of thousands of
rows; at that height the per-worker output block does not fit L1, and prefill is not read-bound
anyway.  So the interleaved weight stays and the sharded copy is derived FROM IT ON DEVICE at build
time (`to_memory_config`, no second host upload).  Callers use `serves(m_tiles)` to pick.
"""
from __future__ import annotations

import math

import ttnn

TILE = 32
# CEILING ON THE MULTICAST FAN-OUT, not on the shape.  in0 is multicast to every compute core, so
# this bounds how wide that mcast gets; it is NOT the arithmetic limit.  Most projections here are
# pinned to 32 anyway by gcd(k_tiles, n_tiles) -- e.g. gate/up are gcd(96, 256) = 32 -- but the
# FUSED QKV is gcd(96, 192) = 96, so a hard 32 was leaving a legal 6x8 = 48-core rectangle unused
# on the widest attention projection.  Raised to what the grid can actually express.
_MAX_COMPUTE_CORES = 64
# m_tiles x n_tiles of output one DRAM-bank worker may own before its buffers stop fitting L1.
_MAX_OUT_TILES_PER_WORKER = 128
# Reader cores to put on each DRAM bank, widest first.  The op's own bound is [1, 3]
# (matmul_utilities.cpp validate_num_workers_per_dram_bank) and >1 is Blackhole-only, so 3 is a
# legal rung this planner simply never asked for -- it stopped at 2.
_WORKERS_PER_BANK = (3, 2, 1)
# Weight BYTES one bank reader may own before a THIRD reader on that bank stops paying.
#
# A THIRD READER IS NOT UNIFORMLY BETTER, AND THE SPLIT IS BY SLICE SIZE, NOT BY SHAPE.  Measured
# 2026-09-07 by running the whole decode stack at (3, 2, 1) and reading the per-op DRAM utilisation
# back: at 8 banks the rung reaches exactly the projections whose chunk divides 24 -- o_proj and
# down (96 output tiles) and the fused qkv (192) -- and the three of them moved in three different
# directions at IDENTICAL core grid, in0_block_w and dtype:
#   o_proj (13.4 MB of weight, 0.56 MB a reader)  34.70 -> 31.62 us, 70.8% -> 77.7% of DRAM peak
#   qkv    (20.1 MB,           0.84 MB a reader)  44.50 -> 44.24 us, 82.9% -> 83.3%  (a wash)
#   down   (26.8 MB,           1.12 MB a reader)  56.60 -> 60.02 us, 86.9% -> 81.9%  (a LOSS)
# so the third reader buys a third outstanding read where the bank is LATENCY-bound and fragments
# the stream where it is already bandwidth-bound.  The two regimes are separated by the bytes one
# reader owns, and the only quantity that changes across those three is K.  0.75 MB sits between
# the win and the wash; qkv is inside the noise either way and keeps the pick it was measured on.
_MAX_WEIGHT_BYTES_PER_BANK_READER = 768 * 1024
# Weight BYTES one in0 mcast block must still stream for widening the activation shard to pay for
# the extra block's synchronisation -- see in0_grid for the four measurements that set it, and for
# why the bound is in bytes rather than in the tiles it used to be counted in.
_TILE_BYTES_BF8 = 1088
_MIN_WEIGHT_BYTES_PER_IN0_BLOCK = 384 * _TILE_BYTES_BF8
# Rows below which `mm` does NOT name the core grid (see the note at its call to ttnn.linear).
_GRID_REQUEST_MIN_ROWS = 32 * TILE


def _core_count(device, k_tiles, n_tiles):
    """Largest core count (as a grid that fits the device) dividing BOTH k_tiles and n_tiles.

    The DRAM-sharded matmul gives every core an exact slice of K and of N and has NO padding
    support, so a count that does not divide both is invalid rather than merely slow.

    THE k_tiles CLAUSE IS NOT LOAD-BEARING FOR THE MATMUL, AND DROPPING IT STILL LOSES.  in0 has
    had its own rectangle since the in0_grid split, the factory takes the two shard grids as
    independent arguments, and in0 is MULTICAST to every compute core -- so on paper a compute core
    never owns a slice of K and this clause only pins the grid: it holds gate/up to 32 cores where
    64 divide N, qkv to 48 where 64 do, down and o_proj to 32 where 48 do.  Measured 2026-09-07 with
    the clause removed: every decode matmul came back at the SAME time to within noise (gate 57.69 +
    up 38.22 = 95.9 us against 95.9; down 56.38 vs 56.36; o_proj 31.55 vs 31.52; qkv 44.36 vs 44.27;
    lm_head 125.69 vs 125.68) and PCC was bit-identical -- yet device_ms went 474.65 -> 493.63, +4.0%.
    So the output core count is NOT a matmul knob at one tile row (these ops are bound on the bank
    readers, not on the cores that own N); it is a HAND-OFF knob, and widening it re-shapes every
    consumer's borrow at once.  All 19 ms of the loss was reshards appearing around ops that had been
    matching tile for tile.  Leave the clause in: it is what keeps the chain coherent.
    """
    g = device.compute_with_storage_grid_size()
    best = None
    for y in range(1, g.y + 1):
        for x in range(1, g.x + 1):
            c = x * y
            if c > _MAX_COMPUTE_CORES or k_tiles % c or n_tiles % c:
                continue
            if best is None or c > best[0]:
                best = (c, x, y)
    return best


def in0_grid(device, k_tiles, weight_tiles, tile_bytes=_TILE_BYTES_BF8):
    """Widest in0 (activation) shard whose per-core slice still amortises its mcast block.

    The activation is width-sharded one mcast block per core, so the core count IS the number of
    K blocks the matmul iterates.  Widening it narrows every in0/in1 circular buffer and measurably
    raises the weight stream's DRAM utilisation -- but each block also costs a fixed multicast and
    synchronisation, and that only pays if the block still streams a substantial slice of in1.
    MEASURED on this model, per call: gate/up (24576 weight tiles) 32 -> 48 cores took 71.0 -> 58.3
    us and 69% -> 84% of DRAM peak; down (24576) 32 -> 64 cores took 65.9 -> 56.6 us and 75% -> 87%.
    o_proj is the counter-example that sets the floor: it has HALF the weight (12288 tiles), so at 64
    cores each block carries only 192 tiles and the block overhead stops being hidden -- 34.7 -> 51.3
    us and 71% -> 48%.  So the bound is weight PER BLOCK, not a core count.

    AND THE BLOCK IS BOUNDED IN BYTES, NOT IN TILES.  All four of those were taken on bf8_b weights,
    where "384 tiles" and "418 kB" are the same sentence, and the code kept the tiles.  `up` is
    bf4_b, whose tile is 576 bytes: at the 48 cores the tile rule hands it, one block streams 279 kB
    -- BELOW down's 418 kB, and only a third above o_proj's 209 kB failure -- and the block is what
    in0_block_w is pinned to, since the mcast slice IS the activation shard width (the op asserts
    `shard_width_tiles % in0_block_w == 0`, so 2 tiles per core admits nothing wider than 2).  It
    profiles exactly where that predicts: 60.8% of DRAM peak against its bf8_b twin's 84.5% at the
    IDENTICAL shape, core count and program config, the only difference being the weight's format.
    Weighing the slice in bytes moves `up` to 32 cores (in0_block_w 3, 442 kB a block) and leaves
    every bf8_b projection's pick bit-identical, because at 1088 B/tile this threshold IS the old one.
    """
    g = device.compute_with_storage_grid_size()
    best = None
    for y in range(1, g.y + 1):
        for x in range(1, g.x + 1):
            c = x * y
            if c > _MAX_COMPUTE_CORES or k_tiles % c:
                continue
            if (weight_tiles // c) * tile_bytes < _MIN_WEIGHT_BYTES_PER_IN0_BLOCK:
                continue
            if best is None or c > best[0]:
                best = (c, x, y)
    return best


_FILL_SHARD_CACHE = {}


def _fill_shard_config(device, shp):
    """ONE HEAD PER CORE, which is what makes the folded fill correct as well as cheap.

    fill_cache splits its work into (head, seq-tile) blocks and gives each core a run of them, but
    the writer only jumps the cache's per-head stride BETWEEN cores -- inside a core it just steps
    one tile width at a time (fill_cache_multi_core_program_factory: the per-core cache_start_id
    carries `num_blocks_written / input_Ht * cache_HtWt`, and the kernel comment says outright that
    it assumes work does not spill over to the next head).  On an INTERLEAVED input the split is
    ceil(heads * seq_tiles / cores), which on this model is 10 tiles against a 16-tile head, so
    most cores straddle a head boundary and write the tail at the wrong offset -- measured PCC
    0.4585.  A HEIGHT-SHARDED input takes the other branch: the split becomes shard_height /
    TILE_HEIGHT and the core set becomes the shard grid, so one head per core makes every core's
    run exactly one head and the assumption holds.
    """
    heads, seq, width = shp[0] * shp[1], shp[2], shp[3]
    key = (heads, seq, width)
    mem = _FILL_SHARD_CACHE.get(key)
    if mem is None:
        g = device.compute_with_storage_grid_size()
        grid = ttnn.num_cores_to_corerangeset(heads, g, True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [seq, width], ttnn.ShardOrientation.ROW_MAJOR),
        )
        _FILL_SHARD_CACHE[key] = mem
    return mem


def fill_kv_prefill(kv, k, v):
    """Write a whole prefill K/V into the resident caches at sequence offset 0, in TWO launches.

    ONE STREAM PER LAUNCH WAS NEVER A REQUIREMENT, ONLY A HABIT.  `ttnn.fill_cache(cache, input,
    batch_idx)` documents a [1, n_kv, S, hd] input, so this ran a slice plus a fill per stream per
    tensor -- 4 x B = 32 launches a layer, and on the 8-stream prefill the slices alone profiled at
    78 us/layer moving bytes that were already in the right place.  But the op does not actually
    constrain the input's batch: its FILL validation only checks matching dtypes, batch_idx <
    cache.padded_shape()[0], and that the input height fits (update_cache_device_operation.cpp), and
    its factory derives the work split from input.padded_shape()[1] * input.padded_shape()[-2] and
    the destination from batch_idx * cache.padded_shape()[1] * cache_HtWt.  So folding the stream
    axis into the HEAD axis -- cache [B, n_kv, C, hd] -> [1, B*n_kv, C, hd], input [B, n_kv, S, hd]
    -> [1, B*n_kv, S, hd] -- makes batch_idx 0 address every stream's every head at offset 0, which
    is exactly the region the loop wrote.  Both reshapes merge LEADING dims and leave the last two
    alone, so they are metadata views on the same buffers and the write still lands in the resident
    cache.  The cache views are built once and kept on the slot; only k and v are viewed per call.

    Falls back to the per-stream loop if anything about the shapes or the op rejects the fold, so
    correctness never depends on this holding.
    """
    shp = [int(d) for d in k.shape]
    batch = shp[0]
    # MATCH THE CACHE DTYPE ONCE, NOT PER STREAM.  update_cache's FILL path refuses a mixed
    # precision write outright ("Input and cache tensors must have same dtype!" -- only its DECODE
    # path has the conversion kernel), so a narrowed cache has to be met here.
    # THE CACHE CANNOT GO NARROWER THAN bfloat8_b.  sdpa_decode reads it at 287 GB/s against the
    # 465 the projections reach, so halving its width looked worth ~0.5 ms/token, and the paged
    # cache family accepts BFLOAT4_B (only the fill this function calls, UpdateKVCacheOperation,
    # refuses it -- update_cache_device_operation.cpp:38).  Priced before writing that plumbing, by
    # round-tripping the prefill fill through bfloat4_b: e2e PCC 0.9635 -> 0.9366 against a 0.95
    # gate, and that is the PREFILL positions alone.  The earlier finding that widening the cache
    # bf8_b -> bf16 changed PCC by 0.0000 does not extend downward; bf8_b is the floor here.
    if k.dtype != kv.k.dtype:
        k = ttnn.typecast(k, kv.k.dtype)
    if v.dtype != kv.v.dtype:
        v = ttnn.typecast(v, kv.v.dtype)
    if batch > 1:
        try:
            flat = getattr(kv, "_flat_views", None)
            if flat is None:
                kc = [int(d) for d in kv.k.shape]
                flat = (
                    ttnn.reshape(kv.k, (1, kc[0] * kc[1], kc[2], kc[3])),
                    ttnn.reshape(kv.v, (1, kc[0] * kc[1], kc[2], kc[3])),
                )
                kv._flat_views = flat
            mem = _fill_shard_config(k.device(), shp)
            for cache_view, src in ((flat[0], k), (flat[1], v)):
                folded = ttnn.reshape(src, (1, shp[0] * shp[1], shp[2], shp[3]))
                sharded = ttnn.to_memory_config(folded, mem)
                ttnn.fill_cache(cache_view, sharded, 0)
                ttnn.deallocate(sharded)
            return
        except (RuntimeError, TypeError, AttributeError, ValueError):
            pass
    for b in range(batch):
        if batch == 1:
            kb, vb = k, v
        else:
            kb = ttnn.slice(k, (b, 0, 0, 0), (b + 1, shp[1], shp[2], shp[3]))
            vb = ttnn.slice(v, (b, 0, 0, 0), (b + 1, shp[1], shp[2], shp[3]))
        ttnn.fill_cache(kv.k, kb, b)
        ttnn.fill_cache(kv.v, vb, b)


# Bytes a full-width activation may have before interleaved L1 stops being the obviously right home.
# The audio tower's stream is [1, 1504, 1280] bf8_b = 1.9 MB against 110 x 1.5 MB of L1, so this cap
# exists only so a much longer sequence degrades to DRAM rather than crowding out SDPA's own circular
# buffers, which reserve ~1.03 MB of each core while attention runs.
# THE CAP EXCLUDES PREFILL ON PURPOSE, AND THAT IS MEASURED.  The audio tower's stream is
# [1, 1504, 1280] bf8_b = 1.9 MB; prefill's is [8, 448, 3072] bf8_b = 11.7 MB, which is only ~106 kB
# per core and looked like it should fit alongside the ~1.03 MB prefill SDPA's flash circular buffers
# reserve.  It does not: raising this to 16 MB so the prefill norms and residual adds became
# L1-resident CRASHED the pipeline -- "Statically allocated dataflow buffers in program 168 clash
# with L1 buffers on core range [0-0 - 10-9]. L1 buffer allocated at 807296 and static dataflow
# buffer region ends at 896512" (measured 2026-09-05).  The clash is with the STATIC dataflow buffer
# region, which is reserved per program regardless of what else is live, so per-core arithmetic
# against the 1.5 MB budget does not predict it.  4 MB keeps the audio tower in L1 and lets prefill
# fall back to DRAM, which is exactly what the size gate below is for.
_STREAM_L1_MAX_BYTES = 4 * 1024 * 1024

# THE DECODE ATTENTION KEEPS HiFi4 EVEN THOUGH ITS CACHE IS BLOCK-FLOAT, and that is measured, not
# inherited.  The KV cache is bf8_b (pipeline.KV_DTYPE) and q arrives narrow, so by the same argument
# this file makes for the bf8_b projections -- "8-bit operands through a HiFi4 kernel make the math
# engine take four passes over one pass worth of precision" -- HiFi2 looked like free math phases.
# It is NOT: measured 2026-09-05 across all three decode bodies (llama_attention, llama_decoder_layer,
# llama_model), HiFi2 + fp32_dest_acc_en cost decode 10.983 -> 11.048 ms/token (+0.6%) while e2e PCC
# barely moved (0.9598 -> 0.9601).  sdpa_decode is dispatch- and bandwidth-bound at this shape (one
# tile row of q against a few hundred cached positions), so cheapening the math phases buys nothing
# and the narrower fidelity path costs something.  Do not spend another round on it.

# Bytes PER ELEMENT, and the block-float entries are NOT rounded up to the next whole byte.  A
# bf8_b tile is 1024 mantissa bytes plus a 64-byte shared-exponent header, so 1.0625 B/element,
# and bf4_b is 0.5625.  Calling them 2 and 1 overstates the audio tower's FFN intermediate as
# 15.4 MB when it is 8.2 MB, which silently pushed it past ffn_config's cap and made the L1
# hand-off a no-op that still measured (profile identical to the DRAM path).
_DTYPE_BYTES = {ttnn.bfloat4_b: 0.5625, ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2, ttnn.float32: 4}


def tile_bytes(dtype):
    """Bytes ONE 32x32 tile of `dtype` occupies -- what an in0 mcast block is really weighed in."""
    return int(TILE * TILE * _DTYPE_BYTES.get(dtype, 2))


# Bytes the FFN intermediate may have before L1 residency stops paying -- see ffn_config.
_FFN_L1_MAX_BYTES = 12 * 1024 * 1024


# DECODE SDPA IS LEFT WITH NO PROGRAM CONFIG, AND THAT IS MEASURED.  It carries 27.1 ms of the
# capture at ~35 us/call and was being called with no program_config at all -- the same defaulting
# that cost the audio tower dearly, where sdpa_program_factory falls back to one-tile chunks.  So a
# full-grid SDPAProgramConfig looked free.  It is not worth taking: q is a single padded tile row at
# decode so q_chunk is pinned at TILE, max_cores_per_head_batch is not binding (8 users x 8 kv heads
# = 64 head-batch pairs against 110 cores, ~1.7 cores per pair, far under the default 16), and the
# only real axis left -- the k chunk -- measured decode 10.986 -> 10.985 ms/token at 128, i.e.
# nothing, while 512 BROKE THE TRACE outright (the decode stage stopped reporting; only encode and
# prefill came back), because a 512-wide k block's flash circular buffers do not fit alongside the
# traced program's own L1.  Neutral upside, real fragility: do not pin this op's config.


def stream_config(x, dtype=None):
    """Interleaved L1 for the layer's full-width activation while it is small enough to live there.

    THE OPS BETWEEN THE MATMULS ARE THE ONES PAYING FOR DRAM.  A layer norm and a residual add on
    the audio tower's [1, 1504, 1280] stream each read and write the whole tensor and nothing else,
    so their cost IS the placement: measured DRAM-interleaved they run at 119 GB/s (norm, 32 us/call)
    and 145 GB/s (add, 40 us/call) against 3.8 MB of traffic, an order under what the matmuls beside
    them reach.  There are four such passes per layer.  Interleaved L1 puts those pages in Tensix
    banks the workers reach over the NoC instead of through the DRAM controller; it is pure
    placement, so the values are bit-identical.

    Interleaved and NOT sharded on purpose: 1504 rows is 47 tile rows and 47 is prime, so no
    rectangle of this grid divides it, and naming a shard spec that one consumer wants is what
    forces an extra reshard at the next one rather than removing one.

    Size-gated, so a shape this was not sized for falls back to DRAM instead of failing.
    """
    n = 1
    for d in tuple(x.shape):
        n *= int(d)
    width = _DTYPE_BYTES.get(dtype if dtype is not None else x.dtype, 2)
    return ttnn.L1_MEMORY_CONFIG if n * width <= _STREAM_L1_MAX_BYTES else ttnn.DRAM_MEMORY_CONFIG


# FOLDING THE PREFILL BATCH INTO THE TILE-HEIGHT IS A WASH -- MEASURED TWICE, DO NOT RETRY.
# The projections are position-wise, so [B, S, K] -> [1, B*S, K] is a metadata view (tile-aligned
# height) and it looked like two levers at once: both matmul factories iterate batch on the OUTSIDE,
# so a 25 MB weight appeared to be re-streamed once per stream, and `mm` gates its 2-D block config
# on batch == 1 so a batched shape can never reach it.  Both halves are false here.
#   * bandwidth: folding alone moved 512x3072x8192 to 4096x3072x8192 at an IDENTICAL 15.45 ms, so
#     ttnn does NOT re-read the weight per batch entry at prefill height.  (The 932 -> 118 us the
#     decode step gets from the same reshape comes from 1-row-per-batch tile padding, not from reads.)
#   * blocks: with the fold in place block_config reached qkv and o_proj and moved them 4.975 ->
#     4.977 and 3.006 -> 2.804 ms -- 0.2 ms across the whole capture, against picking 99- and 88-core
#     plans where the plain grid request had 110.
# What this leaves standing is the SHAPE of the prefill gap: fitting 512 vs 448 rows gives a MARGINAL
# rate of ~628 TFLOP/s (essentially peak) plus ~316 us of fixed cost per gate/up call and ~178 us per
# down call.  That fixed part is NOT K-block synchronisation either (see _CB_BUDGET_BYTES), so
# whatever it is, block shaping does not reach it.


def ffn_config(rows, width, dtype=None):
    """Interleaved L1 for the FFN's WIDE intermediate, so the second projection never reads DRAM.

    THE FF1 -> FF2 HANDOFF IS THE ONE PLACE A WHOLE TENSOR IS WRITTEN AND IMMEDIATELY RE-READ.
    GUIDELINES/05 section 7 names it: FF1 writes [rows, hidden] and FF2's very next act is to stream
    that same tensor back in, so leaving it in DRAM costs a full write plus a full read of a value
    with exactly one consumer, one op later.  On the audio tower that is 7.7 MB at bf8_b, and fc2
    measured 65 us/call moving ~16 MB, i.e. 246 GB/s -- the activation half of that is what moves.

    Separate cap from `stream_config` because this tensor is FOUR TIMES the width of the layer
    stream: 110 x 1.5 MB of L1 holds it comfortably (70 kB per core against the matmul's own 700 kB
    circular-buffer budget), but the cap has to be sized for the wide tensor rather than the narrow
    one, and a longer sequence must still degrade to DRAM rather than crowd out those buffers.
    """
    width_bytes = _DTYPE_BYTES.get(dtype if dtype is not None else ttnn.bfloat8_b, 2)
    return (
        ttnn.L1_MEMORY_CONFIG if int(rows) * int(width) * width_bytes <= _FFN_L1_MAX_BYTES else ttnn.DRAM_MEMORY_CONFIG
    )


# Bytes ONE of the SwiGLU's three wide intermediates may have before L1 residency stops paying.
# SEPARATE FROM ffn_config's CAP BECAUSE THE TENANT COUNT IS DIFFERENT.  The audio tower's FFN has
# a single [rows, hidden] value in flight at a time, so its cap is sized against one tensor; a
# SwiGLU has THREE -- gate, up and their product -- and the widest moment (the `up` matmul, with
# gate already resident and the product not yet allocated) holds two of them beside that matmul's
# own ~700 kB of circular buffers.  At the LM prefill shape one of these is 3584 x 8192 x 1.0625 B
# = 31.2 MB, i.e. 284 kB on each of 110 cores, so two tenants plus the buffers come to ~1.27 MB of
# the 1.5 MB each core has.  That is the real ceiling, and the cap below is set from it rather
# than from the total L1 the board reports.
_SWIGLU_L1_MAX_BYTES = 34 * 1024 * 1024
# (rows, width) pairs whose L1 hand-off the allocator refused at runtime, so the fallback to DRAM
# is paid once rather than once per call.  The estimate above counts the tensors this module knows
# about; the trace region, the resident caches and the next op's buffers are not visible from here.
_SWIGLU_L1_REFUSED = set()


def _swiglu_handoff(rows, width, dtype):
    """Interleaved L1 for the SwiGLU's gate/up/product triple, or None to leave them in DRAM.

    THE LM's FFN INTERMEDIATE NEVER REACHED ffn_config.  `linear` had no memory_config to pass, so
    gate and up were written to DRAM, read back by the multiply, written again, and read a third
    time by the down projection -- five full passes over a tensor whose only consumers are the two
    ops after it.  Returns None (rather than DRAM_MEMORY_CONFIG) so the caller can keep the exact
    call it made before when the hand-off does not apply, which is what makes the fallback
    bit-identical to the previous behaviour instead of merely equivalent.

    THIS HAND-OFF IS WORTH 7.36 ms OF PREFILL, NOT THE 0.3 ms ITS OWN COMMIT MESSAGE CLAIMS, AND IT
    IS THE BEST TENANT IN PER-CORE L1.  Measured 2026-09-08 by turning it off at the prefill height
    alone (_SWIGLU_L1_MAX_BYTES to 12 MB, which excludes the 29 MB intermediates and leaves the
    decode height -- already excluded by the row gate -- untouched): prefill 106.49 -> 113.85 ms,
    PCC bit-identical.  So anything that wants L1 during the FFN is bidding against 7.4 ms.

    The rival is the residual chain, and the two are STRICTLY SUB-ADDITIVE -- the full 2x2 is in
    residual_add.  Short version: the chain is worth 2.72 ms on its own and -0.45 ms once this
    hand-off holds the same L1, so this one wins and there is no configuration where both pay.
    """
    rows, width = int(rows), int(width)
    if rows < _GRID_REQUEST_MIN_ROWS or (rows, width) in _SWIGLU_L1_REFUSED:
        return None
    width_bytes = _DTYPE_BYTES.get(dtype if dtype is not None else ttnn.bfloat8_b, 2)
    if rows * width * width_bytes > _SWIGLU_L1_MAX_BYTES:
        return None
    return ttnn.L1_MEMORY_CONFIG


def _linear_activation(activation):
    """The ttnn.linear `activation=` STRING for a name, keeping gelu on the approximate kernel.

    `mm`'s fallback path hands the name straight to ttnn.linear, whose own mapping reads "gelu"
    as the exact erf -- the form _fused_activation documents as a large net loss inside a pack
    loop.  Route it to ttnn's approximate spelling so both paths fuse the same kernel.
    """
    return "gelu_approx" if str(activation) == "gelu" else activation


def _apply(activation, x):
    """Run `activation` as a standalone unary (the decode/mirror path); None is a pass-through."""
    if activation is None:
        return x
    if str(activation) in ("gelu", "gelu_approx"):
        return ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh)
    return getattr(ttnn, str(activation))(x)


def _largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


# Bytes an UNSHARDED decode intermediate may have before it is cheaper to hand to the next op
# through DRAM than through L1.  These tensors are one tile row tall, so they are tens of kB where
# the weights they sit between are tens of MB; the cap only exists so a shape this was not sized for
# (a prefill height reaching the mirror, a wider vocab chunk) degrades to DRAM instead of filling L1
# underneath the matmul's own circular buffers.
_L1_HANDOFF_MAX_BYTES = 256 * 1024

# THE PREFILL ACTIVATION IS THE OTHER HALF OF THE STAGE, AND IT IS CARRIED AT DOUBLE WIDTH.
# Prefill's matmuls account for ~85 ms of the stage; the ~100 ms around them is residual adds,
# RMSNorms, rope, head splits and tilizes, and every one of those is bandwidth-bound on the SAME
# [8, 512, 3072] bf16 tensor -- 25 MB read and 25 MB written, six or more full-width passes per
# layer.  L1 cannot hold it (prefill SDPA's flash circular buffers already reserve ~1.03 MB of each
# core's 1.5 MB, so an interleaved-L1 activation fails the CB region check), which leaves the other
# axis: carry the stream at bf8_b and every one of those passes moves half the bytes.
#
# bf8_b IS THE FLOOR, NOT A STEP ON THE WAY DOWN.  These activations feed normalization statistics
# and SDPA's scores, and GUIDELINES/01 section 13 names both as tensors that must never go below
# bf8_b.  The matmuls already consume bf8_b weights, so asking for a bf8_b OUTPUT only narrows what
# the ops BETWEEN them move; the math inside each matmul is unchanged.  Decode is deliberately
# excluded -- its activation is one tile row of tens of kB against weights of tens of MB, so there
# are no bytes to save there, and it is already at its DRAM floor.
_ACT_DTYPE = ttnn.bfloat8_b


# Bytes a rope table may have before L1 residency stops being obviously free.  The prefill tables
# are [1, 1, S, head_dim] -- 128 kB each at S=512, head_dim=128 -- against a 110 x 1.5 MB budget,
# so this cap only exists to stop a much longer context silently filling L1 under the matmuls.
_ROPE_L1_MAX_BYTES = 1024 * 1024

# THE ROPE HAS ITS OWN FIDELITY PAIRING, AND NOT PASSING ONE IS NOT THE SAME AS NOT HAVING ONE.
# `rotary_embedding_hf` documents that a caller who hands it no compute_kernel_config gets
# init_device_compute_kernel_config's default -- math_fidelity=HiFi4 -- and every call site here
# handed it none, so the op has been running the model's HIGHEST fidelity on its cheapest maths.
# The rotation is two multiplies and an add; HiFi4 makes the FPU take four passes over operands
# that carry ONE pass worth of mantissa (prefill's q and k are bf8_b off the head split, decode's
# are bf16), which is exactly the argument _NORM_CFG already made for the norms.  It shows in the
# profile: prefill's q rope moves 15.6 MB in and 15.6 MB out -- a 46 us round trip at this board's
# measured bandwidth -- and takes 203.4 us, so it is not the bytes that bind.
# HiFi2 RATHER THAN LoFi because cos/sin are the one wide pair in the expression: they are the
# rotation itself, LoFi keeps ~5 mantissa bits, and this model's e2e PCC sits at 0.9559 against a
# 0.95 gate with no budget to spend on a rounding step that compounds over 32 layers.
#
# THE THIRD LM BODY IS DELIBERATELY NOT GIVEN THIS CONFIG, AND THAT IS A MEASUREMENT, NOT AN
# OVERSIGHT.  `llama_attention` calls rotary_embedding_hf with no compute_kernel_config and so
# still runs its rope at the HiFi4 default.  Handing it ROPE_CFG for consistency -- taking one of
# three bodies from HiFi4 to HiFi2 -- moved e2e PCC 0.9635 -> 0.9536, i.e. 0.0099 out of a budget
# that stands at 0.0135 above the 0.95 gate, in exchange for roughly a third of ~62 us per prefill
# layer.  Rope fidelity is one of the most PCC-DENSE knobs in this model (cos/sin really are the
# wide pair in the expression), which makes accuracy held here worth more than the microseconds
# it costs.  Leave that body on the default.
ROPE_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

# THE LM'S ATTENTION HAS THE SAME PAIRING PROBLEM, AND IT IS THE LARGEST INSTANCE OF IT LEFT.
# All three LM bodies hand scaled_dot_product_attention -- and its decode form -- a HiFi4 config,
# but every operand the op reads is block-float: q, k and v come off the head split at _ACT_DTYPE
# and the resident KV cache is bf8_b as well.  HiFi4 makes the FPU take FOUR passes over operands
# carrying one pass worth of mantissa, which is the argument _NORM_CFG and ROPE_CFG already made
# for the cheaper ops; attention is where it costs the most.  Measured in the prefill profile at
# 428 us/call on 8x32x448x128 -- ~31 TFLOP/s of causal work on a board whose block-float matmuls
# reach 280-440 on the same dtypes -- and it is the second most expensive op in the prefill layer.
#
# HiFi2 RATHER THAN LoFi, and that is a dtype fact rather than caution: bfp8_b's mantissa needs two
# FPU passes to be read exactly, so HiFi2 is the LOSSLESS pairing for this operand width where LoFi
# truncates to ~5 bits.  The truncated values here would be the scores the softmax exponentiates,
# so the error would not stay linear the way it does in a projection.  The audio tower does run its
# own SDPA at LoFi, but that attention is not causal-masked and does not feed a resident cache, and
# this model's e2e PCC sits at 0.9559 against a 0.95 gate with no budget for a compounding step.
#
# fp32_dest_acc_en STAYS TRUE.  It is a different knob from fidelity: it is what keeps flash's
# running max and running sum in fp32 across the k loop, the one place in this op where the
# accumulator's width is load-bearing rather than merely wide.
#
# PREFILL ONLY, AND THAT IS MEASURED RATHER THAN CAUTIOUS.  The decode form of the op reads the
# same bf8_b operands, so the pairing argument applies to it word for word -- but it is bound on
# DISPATCH, not on the FPU: one tile row of q against a 640-long cache is 35.4 us/call of which
# the maths is a rounding error, and pointing it here moved the trace+1cq decode 10.6521 ->
# 10.7506 ms/token, i.e. nothing but noise in the wrong direction.  So the decode call sites keep
# their own _HIFI4_CFG and only the prefill ones take this.
ATTN_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def rope_resident(cos, sin):
    """Put the prefill rope tables in L1, once per forward, and hand back the pair.

    THE TABLE IS TINY AND IS READ ONCE PER HEAD ROW.  rotary_embedding_hf's prefill mode takes the
    activation as [1, heads, S, hd] and cos/sin as [1, 1, S, hd], broadcasting them over dim 1 --
    and this model folds batch into that dim, so a [8, 32, S, hd] q becomes 256 rows that all read
    the SAME 128 kB table.  Left in DRAM that is the bulk of the op's traffic: measured at the
    prefill shape the two calls per layer moved the 34 MB of q/k in 317 + 95 us, about 107 GB/s,
    an order below the eltwise ops beside them, and dropping the math phases from four to two
    (HiFi4 -> HiFi2) did not move it -- so the residual is the read, not the math.

    Once per forward, not once per layer: every layer reuses the same pair, so the copy is amortised
    over all of them.  Best-effort -- a shape or config this cannot serve keeps the DRAM tensors.
    """
    out = []
    for t in (cos, sin):
        try:
            n = 1
            for d in tuple(t.shape):
                n *= int(d)
            wide = t.dtype in (ttnn.float32, ttnn.uint32, ttnn.int32)
            if n * (4 if wide else 2) > _ROPE_L1_MAX_BYTES or t.is_sharded():
                out.append(t)
            elif t.memory_config() == ttnn.L1_MEMORY_CONFIG:
                out.append(t)
            else:
                out.append(ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG))
        except (RuntimeError, TypeError, AttributeError):
            out.append(t)
    return out[0], out[1]


def _handoff_config(elements, dtype):
    """DRAM or interleaved L1 for a decode intermediate that must leave its shard, chosen by size.

    A projection at the decode shape produces one tile row: o_proj and down_proj hand back 49 kB and
    the SwiGLU multiply 131 kB, against weights of 12-27 MB.  Sending those through the DRAM
    controller costs a full round trip -- write, then the consumer reads it back -- for something
    that fits in a few kB per core, and the consumer is ALWAYS an op that wants L1 anyway (the
    residual add writes the next norm's L1 shard; the down projection reshards its activation into
    L1).  Interleaved, not sharded: the shard spec these consumers want differs per op, and naming
    the wrong one is what forces an extra reshard rather than removing one.  Same lever as
    sdpa_decode_out_config, applied to the other hand-offs in the decode step.
    """
    wide = dtype in (ttnn.float32, ttnn.uint32, ttnn.int32)
    if int(elements) * (4 if wide else 2) > _L1_HANDOFF_MAX_BYTES:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


class DramShardedLinear:
    """A DRAM-bank-width-sharded mirror of `weight` ([K, N], TILE) plus its matmul configs.

    `serves(m_tiles)` is False when the shape is outside what was planned for, and the caller keeps
    its own path; construction never raises, so a body can adopt this without a shape audit.
    """

    def __init__(self, device, weight, max_m_tiles=1, dtype=None):
        self.device = device
        self.ok = False
        self.max_m_tiles = max_m_tiles
        try:
            self._build(device, weight, max_m_tiles, dtype)
        except Exception:  # noqa: BLE001 - any shape this cannot serve exactly falls back
            self.ok = False

    def _build(self, device, weight, max_m_tiles, dtype=None):
        # Narrow BEFORE planning, not after: every block size below is weighed in tile_bytes(dtype),
        # so a mirror requantised afterwards would be laid out for the width it no longer has.
        if dtype is not None and weight.dtype != dtype:
            weight = ttnn.typecast(weight, dtype)
        shape = tuple(weight.shape)
        self.k, self.n = int(shape[-2]), int(shape[-1])
        k_tiles, n_tiles = self.k // TILE, self.n // TILE
        if self.k % TILE or self.n % TILE:
            return

        dram_grid = device.dram_grid_size()
        dram_cores = dram_grid.x
        plan = self._plan(device, k_tiles, n_tiles, dram_cores, max_m_tiles, tile_bytes(weight.dtype))
        if plan is None:
            return
        splits, self.workers_per_bank, cores, gx, gy = plan
        self.core_grid = ttnn.CoreGrid(y=gy, x=gx)
        self.cores = cores
        self.split_size = self.n // splits
        # THE ACTIVATION SHARD IS A SEPARATE AXIS FROM THE OUTPUT SHARD, and tying them together was
        # costing bandwidth.  The factory takes in0's and the output's shard grids as two independent
        # arguments (create_program_dram_sharded_descriptor(input_all_cores_storage,
        # output_all_cores_storage, ...)) and the ONLY constraint on the K block is
        # `Kt % in0_block_w == 0`; the output alone is what needs a count dividing N.  One shared
        # count therefore has to satisfy BOTH divisibilities, which pinned every projection whose N
        # is a large power of two to gcd(k_tiles, n_tiles) = 32 cores.  Measured consequence: the
        # fused QKV (n_tiles=192, so 48 cores are legal) reaches 82.7% of DRAM peak while gate/up,
        # o_proj and down_proj -- identical dtype, identical K -- sit at 69-75% on 32.  in0 gets the
        # widest rectangle dividing k_tiles, so its per-core slice (== in0_block_w, one mcast block
        # per core) is narrower and every in0/in1 circular buffer shrinks with it.
        in0_plan = in0_grid(device, k_tiles, k_tiles * n_tiles, tile_bytes(weight.dtype)) or (cores, gx, gy)
        self.in0_cores, in0_gx, in0_gy = in0_plan
        self.in0_grid = ttnn.CoreGrid(y=in0_gy, x=in0_gx)
        self.in0_block_w = _largest_divisor(self.k // (TILE * self.in0_cores))
        self._configs = {}

        dram_range = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, dram_grid.y - 1))}
        )
        padded = math.ceil(self.split_size / (TILE * dram_cores)) * (TILE * dram_cores)
        mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_range, (self.k, padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
        )
        base = ttnn.reshape(weight, (self.k, self.n)) if len(shape) != 2 else weight
        # Do not slice when there is only one chunk: a full-width slice is pure work, and on a
        # block-float weight it is not merely wasteful -- ttnn.slice does not handle every
        # block-float layout, so an unnecessary one turns a valid plan into a build failure.
        self.weights = [
            ttnn.to_memory_config(
                base
                if splits == 1
                else ttnn.slice(base, (0, i * self.split_size), (self.k, (i + 1) * self.split_size)),
                mem_cfg,
            )
            for i in range(splits)
        ]
        self.ok = True

    def _plan(self, device, k_tiles, n_tiles, dram_cores, max_m_tiles, tile_bytes=_TILE_BYTES_BF8):
        """Fewest power-of-two chunks that divide the compute grid AND the bank workers exactly."""
        for splits in (1 << i for i in range(int(math.log2(n_tiles)) + 1)):
            chunk_tiles = n_tiles // splits
            if chunk_tiles * splits != n_tiles:
                continue
            picked = _core_count(device, k_tiles, chunk_tiles)
            if picked is None:
                continue
            for wpb in _WORKERS_PER_BANK:
                workers = dram_cores * wpb
                if chunk_tiles % workers:
                    continue
                if max_m_tiles * (chunk_tiles // workers) > _MAX_OUT_TILES_PER_WORKER:
                    continue
                if wpb > 2 and k_tiles * chunk_tiles * tile_bytes / workers > _MAX_WEIGHT_BYTES_PER_BANK_READER:
                    continue
                return splits, wpb, picked[0], picked[1], picked[2]
        return None

    def serves(self, m_tiles):
        return self.ok and 0 < m_tiles <= self.max_m_tiles

    def act_config(self, m_tiles):
        """The exact activation shard this projection will borrow, so a PRODUCER can write it.

        The borrow check in __call__ compares memory configs by value, so the only way an upstream
        op can hand this matmul an activation it does not have to reshard is to be told the config.
        Returns None when the shape is not served, so the caller keeps whatever placement it had.
        """
        if not self.serves(m_tiles):
            return None
        return self._config_for(m_tiles)[1]

    def _config_for(self, m_tiles):
        cfg = self._configs.get(m_tiles)
        if cfg is None:
            cfg = (
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=self.in0_block_w,
                    per_core_M=m_tiles,
                    per_core_N=self.split_size // (TILE * self.cores),
                    fused_activation=None,
                    num_workers_per_dram_bank=self.workers_per_bank,
                ),
                ttnn.create_sharded_memory_config(
                    (m_tiles * TILE, self.k // self.in0_cores),
                    self.in0_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
                # NAME THE OUTPUT SHARD.  `L1_WIDTH_SHARDED_MEMORY_CONFIG` carries no shard spec, so
                # ttnn spreads the output over the WHOLE compute grid while per_core_N was sized for
                # THIS grid -- the two disagree and the circular buffers overflow L1 by a hair.
                ttnn.create_sharded_memory_config(
                    (m_tiles * TILE, self.split_size // self.cores),
                    self.core_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
                None,
            )
            self._configs[m_tiles] = cfg
        return cfg

    def __call__(self, x, compute_kernel_config=None, keep_sharded=False, out_dtype=ttnn.bfloat16):
        """`out_dtype` is NAMED rather than inherited, and bf16 is the safe default on purpose.

        ttnn.linear with no dtype takes the ACTIVATION's, so once the decode stream narrows to
        block-float every projection silently narrows with it -- including the fused QKV, whose k
        and v go straight into the resident cache, and `update_cache` refuses a block-float INPUT
        outright ("Data type of input tensor for update cache must be FLOAT32 or BFLOAT16"; the
        cache itself is bf8_b, only the tensor being written in is constrained).  So the default
        here PINS bf16 and each call site that knows its consumer opts in to `_ACT_DTYPE`.
        """
        dims = [int(x.shape[i]) for i in range(len(x.shape))]
        m = 1
        for d in dims[:-1]:
            m *= d
        program_config, act_cfg, out_cfg, _ = self._config_for(math.ceil(m / TILE))
        # THE ACTIVATION MAY ALREADY BE IN EXACTLY THIS LAYOUT.  The decode RMSNorm width-shards the
        # hidden dim over the widest rectangle dividing hidden/TILE, which is the same rule in0_grid
        # applies to k_tiles -- so norm output and projection input agree tile for tile, and the
        # reshard pair between them is pure overhead.  Borrowed, so it is NOT deallocated below: one
        # norm shard feeds both halves of a SwiGLU.
        borrowed = dims[-1] == self.k and x.memory_config() == act_cfg
        act = x if borrowed else ttnn.to_memory_config(ttnn.reshape(x, (1, 1, m, self.k)), act_cfg)
        parts = [
            ttnn.linear(
                act,
                w,
                program_config=program_config,
                memory_config=out_cfg,
                compute_kernel_config=compute_kernel_config,
                dtype=out_dtype,
            )
            for w in self.weights
        ]
        if not borrowed:
            ttnn.deallocate(act)
        # HAND THE L1 SHARD STRAIGHT TO A CONSUMER THAT WANTS ONE.  The default tail below pushes the
        # result back to DRAM, which is right for a consumer that reads interleaved -- but
        # nlp_create_qkv_heads_decode picks its program factory FROM THE INPUT'S LAYOUT
        # (nlp_create_qkv_heads_decode_device_operation.cpp select_program_factory: width-sharded
        # ROW_MAJOR TILE input -> the SHARDED factory, anything else -> the INTERLEAVED one), so
        # round-tripping through DRAM both costs a ShardedToInterleaved and forces that op to re-read
        # the fused projection from DRAM on its 8 per-user cores.  Returns the raw (1, 1, m, n) shard,
        # which is already the rank-4 shape that op requires.
        if keep_sharded and len(parts) == 1:
            return parts[0]
        # The full output is m x n whether it arrives as one part or as `splits` of them, so size the
        # hand-off from that rather than from a single chunk.
        handoff = _handoff_config(m * self.n, parts[0].dtype)
        out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=-1, memory_config=handoff)
        if len(parts) == 1:
            out = ttnn.to_memory_config(out, handoff)
        return ttnn.reshape(out, tuple(dims[:-1]) + (self.n,))


def attach(device, weight, max_m_tiles=1, dtype=None):
    """Build a mirror for `weight`, or return None when the shape cannot be served exactly.

    `dtype` NARROWS THE MIRROR ALONE, AND THAT IS THE POINT.  The mirror is the DECODE copy of a
    weight -- `serves()` is false above one tile row, so prefill never reads it -- and decode and
    prefill are bound by different things: a decode projection is 12 cores reading a whole weight
    out of DRAM at 78-87% of peak, so its cost IS the stored width, while the same matmul at
    prefill height is compute-bound and reads that weight once for 3328 rows (measured: gate at
    bfloat8_b 362.3 us against up at bfloat4_b 360.0 us -- the width buys prefill NOTHING).
    Narrowing a weight globally therefore pays in one regime and charges e2e PCC in both.  Passing
    the narrow dtype here spends the accuracy only where the bytes are actually on the clock.
    """
    mirror = DramShardedLinear(device, weight, max_m_tiles=max_m_tiles, dtype=dtype)
    return mirror if mirror.ok else None


def linear(
    x,
    weight,
    mirror,
    compute_kernel_config=None,
    core_grid=None,
    activation=None,
    keep_sharded=False,
    memory_config=None,
    out_dtype=ttnn.bfloat16,
):
    """Project through the DRAM-sharded mirror when it serves this shape, else the plain path.

    `activation` is fused into the MATMUL on either path -- the mirror bakes it into its program
    config's fused_activation, and plain ttnn.linear takes the same activation kwarg -- so the
    caller must NOT also apply it afterwards.  The mirror was built with it, so passing a
    different one here would silently disagree; assert instead of quietly running the wrong maths.
    """
    if mirror is not None:
        m = 1
        for d in tuple(x.shape)[:-1]:
            m *= d
        if mirror.serves(math.ceil(m / TILE)):
            # MEASURED: fusing a non-RELU activation into the DRAM-SHARDED kernel is a LOSS.
            # That factory sends anything but RELU down a separate SFPU path in DEST
            # (matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:513), and at the
            # decode shape -- one tile row, the matmul pinned to the 32 cores that divide K and N
            # -- that costs more than the standalone unary it replaces, which runs interleaved
            # across the WHOLE grid.  Measured on this model: fused silu here cost decode
            # 14.755 -> 15.173 ms/token (+2.8%).  So the mirror stays plain and the activation is
            # applied after it.
            return _apply(
                activation,
                mirror(
                    x,
                    compute_kernel_config=compute_kernel_config,
                    keep_sharded=keep_sharded,
                    out_dtype=out_dtype,
                ),
            )
    # The PLAIN path is the opposite: prefill hands ttnn.linear hundreds of tile rows, so fusing
    # removes a full-width read-modify-write of the [rows, intermediate] tensor.  Same measurement:
    # prefill 210.82 -> 196.35 ms (-6.9%).
    rows = 1
    for d in tuple(x.shape)[:-1]:
        rows *= d
    # AND ASK FOR THE BLOCKS HERE TOO.  This is the FFN's call site, and it was the one place that
    # still only named a core grid: `mm` grew the 2-D block config for the attention projections and
    # swiglu never got it, so gate/up/down -- the majority of the LM's prefill flops -- ran on the
    # 1-D mcast path with a one-tile K block while qkv/o_proj next door did not.  Same helper, same
    # guards; a shape it cannot serve falls through to the grid request below unchanged.
    blocked = _block_linear(
        x, weight, compute_kernel_config, activation=activation, rows=rows, memory_config=memory_config
    )
    if blocked is not None:
        return blocked
    # LOSING THE BLOCK CONFIG MUST NOT ALSO LOSE THE FOLD.  These two levers were tied together
    # only because _block_linear happened to own both: a shape it declines dropped to a grid
    # request holding the ORIGINAL [B, S, K], and ttnn reads that leading dim as batch, so the
    # 1-D factory plans against S rather than B*S and runs the whole grid once per stream.
    # Measured when an L1 hand-off made down's 2-D config unplaceable: the profile went from one
    # `3584 x 8192 x 3072` to eight `448 x 8192 x 3072` and the op's roofline gap went 1.86 ->
    # 4.73 ms -- most of which is the fold, not the config.  The fold is a metadata view on a
    # tile-aligned M (see _block_linear), so it costs nothing to keep on this path too.
    folded, dims = x, [int(d) for d in x.shape]
    if len(dims) > 2 and rows != dims[-2] and dims[-2] % TILE == 0:
        folded = ttnn.reshape(x, (1, rows, dims[-1]))
    kwargs = {} if memory_config is None else {"memory_config": memory_config}
    out = ttnn.linear(
        folded,
        weight,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        activation=_linear_activation(activation),
        # Same regime boundary `mm` uses: narrow the output only where the tensor is big enough for
        # the bytes to matter, so a decode shape that misses its mirror still lands in bf16.
        dtype=_ACT_DTYPE if rows >= _GRID_REQUEST_MIN_ROWS else None,
        **kwargs,
    )
    if folded is x:
        return out
    return ttnn.reshape(out, tuple(dims[:-1]) + (int(tuple(weight.shape)[-1]),))


def _mirror_serves(x, mirror):
    """Whether the DRAM-sharded mirror will take this activation's height (i.e. the decode shape)."""
    if mirror is None:
        return False
    m = 1
    for d in tuple(x.shape)[:-1]:
        m *= d
    try:
        return bool(mirror.serves(math.ceil(m / TILE)))
    except (AttributeError, TypeError, ValueError):
        return False


def _restore_rank(out, rank):
    """Put a kept output shard back at the caller's rank, or pass an unsharded result through.

    THE RESIDUAL ADD WANTS THE SHARD, NOT A COPY OF IT.  The DRAM-sharded mirror's default tail
    converts its own width-sharded output to interleaved L1 before returning -- profiled at
    1.34 us/call on 32 cores, twice per layer per token (o_proj and down_proj) -- and for both of
    those the ONLY consumer is residual_add, which names the norm's 48-core shard as its OUTPUT and
    so has to gather whichever layout it is handed.  The conversion is a whole launch spent building
    a layout nothing reads.

    `keep_sharded` returns the raw (1, 1, m, n) shard because that is what the fused-QKV split
    requires, so the rank the caller handed in is restored here instead: dropping a leading one-dim
    is a metadata view on a width shard -- the same view rms_norm's own keep_sharded tail relies on
    -- so the shard spec survives it and the residual add still sees [1, B, hidden].

    Guarded on is_sharded() so the PREFILL path is untouched: there the mirror does not serve, the
    2-D block config returns an interleaved tensor, and this is the identity.
    """
    if not out.is_sharded():
        return out
    dims = [int(d) for d in out.shape]
    if len(dims) <= rank:
        return out
    return ttnn.reshape(out, tuple(dims[-rank:]))


def proj_delta(device, x, weight, compute_kernel_config, mirror=None, bias=None, memory_config=None):
    """A projection whose only consumer is the residual add, so it keeps its output shard.

    o_proj's call site, shared by every attention body: same lever as the down projection at the end
    of `swiglu`, and the same reason -- see _restore_rank for the measurement.  Inert at prefill
    height, where `serves()` is false and `mm`'s keep_sharded never reaches the mirror.
    """
    rank = len([int(d) for d in x.shape])
    return _restore_rank(
        mm(
            device,
            x,
            weight,
            compute_kernel_config,
            bias=bias,
            mirror=mirror,
            keep_sharded=True,
            memory_config=memory_config,
            out_dtype=_ACT_DTYPE,
        ),
        rank,
    )


# silu as a per-input activation the binary op can absorb, or None on a build that cannot spell it.
try:
    _SILU_ACT = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
except (AttributeError, TypeError, ValueError):  # pragma: no cover - depends on the ttnn build
    _SILU_ACT = None


def _silu_mul_kernel():
    """Load the hand-written SwiGLU-product kernel from ../_kernels, or None if unavailable.

    Same standalone-by-path problem this file itself has, one directory over: the stubs carry no
    package context, so the sibling `_kernels` tree has to be imported by file path too.  Returns
    None rather than raising, so a checkout without the kernel keeps the plain ttnn call.
    """
    import importlib.util
    import pathlib
    import sys

    key = "_voxtral_kernels__silu_mul"
    if key in sys.modules:
        return sys.modules[key]
    path = pathlib.Path(__file__).resolve().parent.parent / "_kernels" / "silu_mul.py"
    if not path.exists():
        sys.modules[key] = None
        return None
    try:
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 - a kernel this build cannot import must not break the model
        sys.modules[key] = None
        return None
    return mod


def _multiply_with_silu(gate, up, memory_config, deferred):
    """gate * up, with silu(gate) folded into the multiply's own unpack when it was deferred.

    AND THE SIGMOID INSIDE IT RUNS APPROXIMATE.  Once the silu rides on this op, the multiply is no
    longer a bandwidth-bound eltwise: at the prefill shape a bare add over the same [8, 416, 3072]
    stream runs at 725 GB/s, which scaled to the 8192-wide intermediate is ~120 us, and this call
    measures 307 us -- the ~190 us difference is the SFPU evaluating an exact sigmoid over 27 M
    elements, and it is the single largest non-matmul cost in prefill.  `fast_and_approximate_mode`
    is the binary op's own route to the SFPU's approximate path (the same axis as
    math_approx_mode on a compute config, which the matmul-fused form was measured not to benefit
    from -- there the cost was the pack loop, here it is the transcendental itself).  Only the
    activated form asks for it: a plain multiply has no transcendental to approximate.

    THE 307 us ABOVE IS RIGHT; THE 101.3 THAT ONCE REPLACED IT WAS A PER-LAYER FIGURE DIVIDED BY
    THE LAYER COUNT A SECOND TIME.  A run on 2026-09-07 called 307 stale and put 101.3 us/call and a
    37.1 us counterfactual in its place; re-measured 2026-09-08 on a fresh capture, the prefill
    instance reads 307.1 / 307.4 / 307.2 us on the three profiled layers and the SAME counterfactual
    -- the activation removed outright and the model re-profiled -- reads 98.1-99.0 us.  Both of the
    old numbers are almost exactly a third of the real ones, which is the arithmetic that produced
    them.  Re-measure before trusting a comment, INCLUDING a comment that says it re-measured.
        silu on THIS multiply's unpack   307.2 us/call   (+209 us over the bare 98.2)
        silu in the gate MATMUL's pack   593.2 us/call   (+231 us over the bare 361.8, and the
                                                          matmul's FPU utilisation 76.2% -> 46.4%)
    The placement verdict does NOT change -- the multiply still wins, now by 22 us a layer rather
    than 167 -- and moving it to the matmul was tried anyway (2026-09-07, scoped to prefill height
    so decode kept the deferred form): prefill 106.49 -> 109.29 ms, device_ms 474.65 -> 474.93.
    Reverted.  The unpack really is the cheap host -- the SFPU work is the same either way, but in
    the pack loop it serialises against the matmul's output schedule where the whole grid cannot
    hide it.  Do not move this activation again.

    WHAT THE CORRECTED NUMBER BUYS IS A SIZE, AND THE SIZE IS WORTH KNOWING: the silu is 209 us a
    LAYER, 6.3 ms of prefill over 30 layers and 9.56 ms of device_ms (479.64 -> 470.08 with it
    removed).  It is the largest single non-matmul cost in the model.  AND IT IS AT ITS FLOOR.  The
    SFPU has a cheaper sigmoid and SILU cannot reach it: `UnaryOpType::SIGMOID` carries
    (vector_mode, approximate) and lowers to `sigmoid_tile<VectorMode::RC, 1>` --
    ckernel_sfpu_sigmoid_appx.h, ONE `lut` instruction plus an add -- while `UnaryOpType::SILU`
    carries no parameter at all and ckernel_sfpu_silu.h pins it to `_sfpu_sigmoid_` in a comment of
    its own.  So the LUT was reached the only way ttnn allows, by splitting the op:
    multiply(g, g, a_activations=[SIGMOID_approx]) is silu(g) in one binary op, then multiply by u.
    Measured 2026-09-08: e2e PCC 0.9576 -> **0.8214** against a 0.95 gate.  That LUT is a
    3-coefficient piecewise fit and this model has 0.0076 of PCC margin.
    THE SIGMOID IS AT ITS FLOOR; THE OP IS NOT, AND AN EARLIER VERSION OF THIS DOCSTRING CONFLATED
    THE TWO.  It argued that the kernel rungs were closed because "a hand-written tt-lang or
    Metalium kernel has exactly the same two sigmoids to choose between" and the exact one already
    runs on its cheap path here (binary_ng derives fp32_dest_acc_en from the data formats,
    binary_ng_program_factory.cpp:1205, and bf8_b operands never set it, so `_sfpu_sigmoid_` takes
    the `_sfpu_exp_21f_bf16_` branch and a SINGLE reciprocal iteration rather than
    `_sfpu_exp_accurate_` and two).  Every clause of that is still true and the CONCLUSION was
    wrong.  A kernel does not have to beat the sigmoid to beat the op -- `_kernels/silu_mul.py`
    keeps the exact sigmoid, byte for byte, and reaches **278.8 us/call** against 307.2 by deleting
    the FRAMING around it: the reciprocal's `v_if (t < 0)` NaN guard (unreachable once the exp
    argument is capped, one branch-free SFPSWAP replacing SETCC/MAD/ENDIF) and one of the library
    path's two bf16 roundings, with silu and the product formed in ONE SFPU pass over DEST instead
    of an activation on operand A's unpack plus a separate binary op.  e2e PCC came out slightly
    BETTER, 0.95762 -> 0.95794.  It is ON by default; see the cpp comment at the call site.

    THE TT-LANG RUNG IS THE ONE THAT IS MEASURED CLOSED.  `_kernels/ttl_silu_mul.py` is a real ttl
    kernel for this exact op, run on the full 11x10 grid: 409.8 us/call against ttnn's 307.2 (33%
    slower) and e2e PCC 0.9502707 against 0.9576212.  It is kept but DISABLED (env
    VOXTRAL_TTL_SILU_MUL=1), and its docstring carries both the reason it loses -- ttl emits a
    one-tile-at-a-time pipeline, paying init_sfpu, the DEST acquire/release pair and a write barrier
    PER TILE where binary_ng amortises them over blocks -- and the four ttl 1.0.1 gotchas the
    authoring cost, so the next round does not pay them again.  Its compute body is `silu_tile()`
    then `mul_binary_tile()`: byte for byte what binary_ng already runs.  That is the difference
    between it and the Metalium kernel that won -- ttl could not express the block framing, which
    was the whole prize.

    THE MATH FIDELITY OF THIS OP IS NOT A KNOB, AND THE PROFILE MAKES IT LOOK LIKE ONE.  Tracy
    reports fidelity=hifi4 for both instances of this multiply (prefill 416x8192 and decode
    32x8192), which reads as the model's highest fidelity spent on ONE product per element -- four
    FPU passes over operands carrying one pass worth of mantissa, exactly the pairing argument that
    ROPE_CFG, _NORM_CFG and ATTN_CFG each acted on.  It is not reachable: `BinaryNgDeviceOperation`
    DOES carry a `compute_kernel_config` attribute (binary_ng_device_operation.hpp), but NO eltwise
    Python binding exposes it -- binary_nanobind.cpp offers dtype, memory_config, output_tensor, the
    three activation spans, sub_core_grids and sub_device_id, and nothing else.  So the HiFi4 comes
    from the device op's own default and there is no argument to override it with; a
    compute_kernel_config= kwarg here is a TypeError, not a lever.  Do not spend a round on it.

    AND WHERE THE SAME BINDING DOES EXIST, IT IS STILL INERT -- SO THE TAG IS THE PROFILER'S, NOT A
    KNOB.  The one place to test that cleanly is the KV cache write: `paged_fused_update_cache` and
    `paged_update_cache` BOTH take `compute_kernel_config`, both profile at HiFi4 (inherited from
    init_device_compute_kernel_config, since no call site passed one), and the op's entire job is to
    move one position's k and v into the resident cache and narrow them to its dtype -- no
    accumulation, every element written once.  Measured 2026-09-07 with a LoFi + math_approx config
    threaded to all three LM bodies: **8.22 us/call against 8.23**, PCC bit-identical, device_ms
    474.53 against 474.65.  Nothing.  The 8.2 us is the 8 scattered partial-tile DRAM writes each
    core issues -- the same conclusion the grid knob reached from the other side -- and neither
    fidelity nor fan-out reaches it.  Generalise: on a datamove-shaped op a fidelity tag reports the
    device op's default, and having the binding does not make it a lever.
    """
    if not deferred:
        return ttnn.multiply(gate, up, memory_config=memory_config)
    # THE cpp RUNG FOR THIS OP -- see _kernels/silu_mul.py.  What it removes is NOT the sigmoid
    # (both of the SFPU's LUT sigmoids are measured too coarse for this model, the 6-entry `lut2`
    # included) but the reciprocal's unreachable NaN guard and one of the library path's two bf16
    # roundings.  Anything it cannot serve exactly keeps the ttnn call below, so this is additive.
    km = _silu_mul_kernel()
    if km is not None and km.serves(gate, up, memory_config):
        try:
            return km.silu_mul(gate, up, memory_config)
        except Exception as exc:  # noqa: BLE001 - the ttnn call below is the contract; this is a bonus
            km.note(f"silu_mul refused: {type(exc).__name__}: {exc}")
    for approx in (True, False):
        try:
            return ttnn.multiply(
                gate,
                up,
                memory_config=memory_config,
                input_tensor_a_activations=_SILU_ACT,
                **({"fast_and_approximate_mode": True} if approx else {}),
            )
        except (RuntimeError, TypeError, ValueError):
            continue
    # A build whose binary op will not take per-input activations: pay the standalone unary.
    return ttnn.multiply(_apply("silu", gate), up, memory_config=memory_config)


def swiglu(x, gate_w, gate_ds, up_w, up_ds, down_w, down_ds, compute_kernel_config, core_grid):
    """gate/up -> silu -> multiply -> down, keeping the two halves in L1 when the mirrors serve.

    THE TWO HALVES WERE ROUND-TRIPPING THROUGH DRAM FOR NOTHING.  Each projection ended with a
    ShardedToInterleaved (3.1 us measured) purely so the standalone silu and multiply could read
    interleaved tensors, and those two then moved the [1, B, intermediate] activation through DRAM
    again.  gate and up produce the SAME shard spec, so silu and the multiply both run on the shard;
    the multiply is the only op that must leave L1, and asking it for a DRAM output makes it replace
    BOTH ShardedToInterleaved with itself.  down still reshards -- its in0 rectangle is wider than
    gate/up's output rectangle -- so nothing here depends on a shard-to-shard conversion.

    Prefill is untouched: keep_sharded only reaches the mirror path, and serves() is decode-only, so
    on prefill both halves come back interleaved and the original multiply runs.

    AND ON THE DECODE PATH THE silu IS NOT A LAUNCH OF ITS OWN.  `linear` documents why it cannot be
    fused into the DRAM-SHARDED matmul (that factory sends anything but RELU down a separate DEST
    path, measured +2.8% on decode), so it came back as a standalone unary -- a full read-modify-
    write of the [1, B, intermediate] shard whose ONLY consumer is the multiply on the very next
    line, which is about to read that same tensor again.  ttnn's binary ops take per-INPUT
    activations (binary_nanobind.cpp: input_tensor_a_activations), so the silu can ride into the
    multiply's own unpack instead, and the launch disappears rather than moving.  Same maths -- the
    fused form runs the identical SFPU op on operand A before the multiply.

    AND PREFILL NOW RIDES THE SAME MULTIPLY.  It used to fuse the silu into the MATMUL's program
    config instead -- a measured win at the time (prefill 210.82 -> 196.35 ms) because the thing it
    replaced was a standalone unary over the whole [rows, intermediate] tensor.  But the multiply is
    the better of the two hosts on either path, because it is the one op that reads gate anyway, and
    the pack loop is the one place the activation cannot spread across the grid.  So the choice is
    no longer gated on the mirror; it is unconditional, and the gate matmul asks for no activation.

    AND AT PREFILL HEIGHT THE THREE WIDE INTERMEDIATES NOW STAY IN L1.  `linear` had no
    memory_config to pass, so on that path gate and up were written to DRAM, read back by the
    multiply, written again, and read a third time by down -- five full passes over a tensor with
    exactly two consumers, both one op away.  See _swiglu_handoff for the L1 budget; a shape the
    allocator refuses falls back to the DRAM call this used to make and is remembered so the
    exception is paid once.
    """
    rows = 1
    for d in tuple(x.shape)[:-1]:
        rows *= int(d)
    ffn_mem = _swiglu_handoff(rows, int(gate_w.shape[-1]), _ACT_DTYPE)
    if ffn_mem is None:
        return _swiglu_body(x, gate_w, gate_ds, up_w, up_ds, down_w, down_ds, compute_kernel_config, core_grid, None)
    try:
        return _swiglu_body(x, gate_w, gate_ds, up_w, up_ds, down_w, down_ds, compute_kernel_config, core_grid, ffn_mem)
    except RuntimeError:
        _SWIGLU_L1_REFUSED.add((rows, int(gate_w.shape[-1])))
        return _swiglu_body(x, gate_w, gate_ds, up_w, up_ds, down_w, down_ds, compute_kernel_config, core_grid, None)


def _swiglu_body(x, gate_w, gate_ds, up_w, up_ds, down_w, down_ds, compute_kernel_config, core_grid, ffn_mem):
    """The SwiGLU itself; `ffn_mem` is the hand-off placement for gate/up/product, or None for DRAM."""
    rank = len([int(d) for d in x.shape])
    # THE MULTIPLY IS THE RIGHT PLACE FOR THE silu ON BOTH PATHS, NOT JUST THE MIRROR'S.  This used
    # to be gated on `_mirror_serves` because the two paths had opposite reasons: decode could not
    # fuse into its DRAM-sharded matmul at all, while prefill's fused_activation was a measured win
    # against a STANDALONE unary over the [rows, intermediate] tensor.  But those are not the only
    # two placements, and the third is strictly better than either: the multiply is ALREADY reading
    # gate, so riding in on its unpack costs no pass at all, where the pack loop is a place the
    # activation cannot be spread across the grid.  Prefill measures gate and up at the SAME
    # 3584x3072x8192 shape, and the fused one runs 644.4 us/call against 433.5 -- 280 TFLOP/s
    # against 416 -- with under 20 us of that explained by bf8_b's extra weight bytes over up's
    # bf4_b.  math_approx_mode was tried first and moved nothing (see llama_m_l_p), which is what
    # says the cost is the pack loop rather than which sigmoid runs in it.
    deferred = _SILU_ACT is not None
    gate = linear(
        x,
        gate_w,
        gate_ds,
        compute_kernel_config,
        core_grid,
        activation=None if deferred else "silu",
        keep_sharded=True,
        memory_config=ffn_mem,
        out_dtype=_ACT_DTYPE,
    )
    up = linear(
        x,
        up_w,
        up_ds,
        compute_kernel_config,
        core_grid,
        keep_sharded=True,
        memory_config=ffn_mem,
        out_dtype=_ACT_DTYPE,
    )
    if gate.is_sharded():
        # LEAVE THE SHARD, BUT NOT ALL THE WAY TO DRAM.  The multiply has to produce something
        # unsharded (down's in0 rectangle is wider than gate/up's output rectangle, so there is no
        # shard-to-shard conversion to inherit), and that is what lets it replace both
        # ShardedToInterleaved ops.  But "unsharded" does not have to mean DRAM: at the decode shape
        # this is 131 kB, and down_proj's very next act is to reshard it back into L1.
        n = 1
        for d in tuple(gate.shape):
            n *= int(d)
        # WRITE THE SHARD down IS ABOUT TO BORROW, RATHER THAN ONE IT HAS TO REBUILD.  The comment
        # above is right that the product cannot stay on gate/up's rectangle -- down's in0 grid is
        # wider -- but "leave the shard" and "leave it interleaved" are not the same choice.  The
        # multiply has to gather across cores either way, and the profile shows what interleaved
        # costs: the product comes out L1-interleaved and down's very next act is an
        # InterleavedToSharded on 64 cores, 1.11 us/call, once per layer per token.  Asking the
        # multiply for down's own activation config folds that launch into the gather it was already
        # doing, and the borrow check in DramShardedLinear.__call__ then compares equal.
        rows = max(1, n // int(gate.shape[-1]))
        handoff = _handoff_config(n, gate.dtype)
        wanted = down_ds.act_config(math.ceil(rows / TILE)) if down_ds is not None else None
        h = None
        if wanted is not None and wanted != handoff:
            try:
                h = _multiply_with_silu(gate, up, wanted, deferred)
            except (RuntimeError, TypeError, ValueError):
                # A build or shape whose binary op will not pack this rectangle: the interleaved
                # hand-off below is the placement this call has always made, so nothing is lost.
                h = None
        if h is None:
            h = _multiply_with_silu(gate, up, handoff, deferred)
        # FREE THE HALVES.  multiply builds a new DRAM tensor rather than viewing its inputs, so this
        # is safe, and it matters: the down projection sizes its circular buffers against whatever L1
        # is still free on these cores.
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        dims = [int(d) for d in h.shape]
        if len(dims) > rank:
            h = ttnn.reshape(h, tuple(dims[-rank:]))
    else:
        # THE PRODUCT GOES TO DRAM EVEN WHEN THE HALVES DID NOT, AND THAT IS MEASURED.  Keeping it
        # in L1 too leaves 284 kB/core resident while `down` is trying to place ~700 kB of circular
        # buffers, and the 2-D config is what loses that argument: ttnn refused it, _block_linear
        # blacklisted the SHAPE, and every later call fell to the 1-D path -- where the activation
        # is not folded either, so the profile went from one `3584 x 8192 x 3072` to eight
        # `448 x 8192 x 3072` and the op's roofline gap went 1.86 -> 4.73 ms.  gate and up are the
        # two passes worth having; the product's consumer wants the cores more than the placement.
        # NAMED, NOT DEFAULTED: ttnn's binary ops inherit operand A's placement, so with the halves
        # in L1 an unqualified multiply keeps the product there too and the change above would have
        # been inert.
        # ...AND THE silu RIDES IN ON THIS MULTIPLY'S UNPACK, which is why the gate matmul above no
        # longer carries a fused_activation.  _multiply_with_silu falls back to the standalone unary
        # on a build whose binary op will not take per-input activations, so nothing is dropped.
        h = _multiply_with_silu(gate, up, None if ffn_mem is None else ttnn.DRAM_MEMORY_CONFIG, deferred)
        if ffn_mem is not None:
            # SAME REASON AS THE SHARDED BRANCH ABOVE, AND MORE PRESSING HERE: at prefill height the
            # two halves are 31 MB EACH of L1, and down's circular buffers are sized against
            # whatever is still free on these cores.  multiply builds a new tensor rather than
            # viewing its inputs, so releasing them the moment it returns is safe.
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
    # HAND THE DELTA TO THE RESIDUAL ADD AS A SHARD -- see _restore_rank.
    return _restore_rank(
        linear(h, down_w, down_ds, compute_kernel_config, core_grid, keep_sharded=True, out_dtype=_ACT_DTYPE),
        rank,
    )


# out_subblock (h, w) candidates, widest DEST footprint first.  This is tt-metal's own
# SUBBLOCK_HW_CHOICES, transposed: that table is ordered (w, h) -- get_subblock_sizes reads
# out_subblock_w from element 0 -- and it carries the ODD widths (7, 5, 3) that a short list of
# powers of two does not.  They matter: fc1's per-core width is 15 tiles, which admits (1,5) and
# nothing wider, and dropping to (1,1) there costs half the FPU rate.
_SUBBLOCK_CHOICES = (
    (2, 4),
    (4, 2),
    (1, 8),
    (8, 1),
    (1, 7),
    (7, 1),
    (2, 3),
    (3, 2),
    (1, 6),
    (6, 1),
    (1, 5),
    (5, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 3),
    (3, 1),
    (1, 2),
    (2, 1),
    (1, 1),
)
# Fraction of the best cost inside which two candidates count as equal, so the tie goes to the one
# occupying MORE cores.  The cost model cannot separate a 100-core and a 110-core plan that do the
# same per-core work, and measured on fc1 the wider grid was 6% faster.
#
# THE WINDOW HAS TO BE WIDER THAN THE ONE TERM THE MODEL IS LEAST SURE OF.  `reuse` is a guess at how
# much the unpacker costs a narrow DEST subblock, and it is the ONLY thing separating the two plans
# on this model's two dominant LM shapes: at 11 x 9 gate/up gets per_core 12 x 24 with an out
# subblock of 2x3 (reuse 1.0), at 11 x 10 it gets 11 x 24 -- 8.3% LESS per-core work -- but
# per_core_M 11 is prime, so the subblock drops to 1x6 (reuse 0.857) and the modelled cost rises 6%.
# A 1% window makes that unmeasurable guess decide the grid outright. Widened so a plan doing
# strictly less per-core work on 11 more cores is allowed through; only gate/up and qkv move, and
# whether the reuse penalty is real is then a measurement rather than a constant.
#
# AND IT IS NOW MEASURED, ON THE SIDE THE COMMENT ABOVE LEFT OPEN.  Narrowing the window to 0.01 --
# which hands the grid to the model's own lowest-cost plan, 11 x 9 with a 2x3 subblock at reuse 1.0
# -- cost prefill 106.53 -> 112.43 ms (+5.5%) and encode 16.92 -> 17.27, measured 2026-09-06.  The
# 11 cores are worth more than the DEST subblock the model prices them against, so the wide window
# stays: `reuse` overstates what a 1xN subblock costs on these shapes.
_COST_TIE = 0.10
# Bytes of circular buffer one core may hold for a 2-D mcast matmul.  Blackhole has 1,572,864 B of
# L1 per core; this is deliberately well under half of it because the estimate below counts only
# in0/in1/out/interm and the factory also reserves space for the reader/writer and the semaphores.
# 700 kB IS NOT MERELY CAUTIOUS, IT IS THE MEASURED OPTIMUM.  Raising it to 1100 kB doubles the K
# block these shapes can afford (the audio tower's fc1 goes in0_block_w 10 -> 20), which the cost
# model says should hide the multicast and semaphore round.  MEASURED 2026-09-05: fc2 1504x5120x1280
# went 2.206 -> 3.432 ms (+56%) and encode 19.37 -> 19.58 -- clearly WORSE.  The wider circular
# buffers push the real per-core footprint past what the factory has left after the reader/writer and
# the semaphores, and these matmuls are not bound on K-block synchronisation in the first place.
#
# ...AND IT DOES NOT MOVE FROM THE OTHER SIDE EITHER.  A bigger allowance can be spent two ways and
# BOTH have now been measured losing, which is what makes 700 kB an empirical ceiling rather than a
# model parameter:
#   * on the K BLOCK (the in0 + in1 term) -- the 1100 kB run above, fc2 +56%.
#   * on the OUT BLOCK, which is the term that decides RE-READS: the factory runs the whole K
#     reduction inside each out block, so an out_block_w narrower than per_core_N makes the core
#     re-stream its entire in0 slice once per N block, and gate/up pays exactly that (12x8 against a
#     per_core_N of 24, three passes over the activation).  Widening it was tried twice.  Shrinking
#     the K block to afford it inside 700 kB gives 12x12 at in0_block_w 4 and takes the
#     synchronisation rounds 36 -> 48: 510.18 -> 511.64 ms.  Raising the total to 900 kB with a
#     separate 640 kB ceiling on the K-block term -- so the extra room can ONLY buy the out block,
#     and every encoder shape provably keeps its plan -- gives 12x12 at in0_block_w 8, which is
#     better on BOTH axes (24 rounds, operand traffic 5760 -> 4608 tiles) and still measured
#     509.23 -> 511.41 ms.
# Cutting re-reads by a third and halving the rounds still lost, so the binding constraint is the
# REAL per-core L1 footprint -- which this estimate undercounts -- and not operand traffic at all.
# Do not raise this again without a way to measure the factory's true footprint.
_CB_BUDGET_BYTES = 700 * 1024
# Tile-matmul-equivalents one K block costs in multicast plus semaphore synchronisation.  Used only
# to RANK candidates against each other, so its exactness matters far less than its sign: it is what
# stops the search from picking a one-tile K block, which is the shape of the config ttnn was
# choosing on its own.
_KBLOCK_OVERHEAD = 60.0
# Which output dim the 2-D factory multicasts along.  Its own axis, invisible to the cost model
# below (which is symmetric in the two output dims), so only a measurement can choose it.
#
# FALSE IS THE MEASURED ANSWER ON THIS BOARD, AND IT IS NOT CLOSE.  Measured 2026-09-05 with the
# search ranges swapped to match (see block_config): prefill 136.77 -> 158.91 ms (+16.2%) and
# encode 16.94 -> 20.40 ms (+20.4%).  The reason is the grid, not the multicast: transposing makes
# the M blocks fit the X extent and the N blocks the Y extent, and on an 11 x 10 grid every one of
# these shapes then lands on 10 x 10 = 100 cores instead of 110, because M=112 tiles cannot use the
# 11th column without leaving it short.  Losing 9% of the grid is bad enough on its own, but the
# narrower N extent also forces a wider per_core_N, which eats the circular-buffer budget and drops
# gate/up's K block from 8 tiles to 3.  Keep the flag -- the axis is real and a squarer grid could
# flip it -- but leave it False here.
#
# RE-MEASURED AT THE SHAPE THAT REPLACED THE ONE ABOVE, AND IT IS STILL FALSE -- FOR A SHARPER
# REASON.  That measurement was taken at m_tiles=112 (PREFILL_C 448); RUN72 moved the model to 416,
# so m_tiles is 104 and the "cannot use the 11th column" argument no longer applies -- at 104 the
# LM shapes DO keep all 110 cores under transposition, and the cost model actually prefers it
# (5-9% cheaper on the two dominant shapes) because per_core_M stops being prime.  Measured
# 2026-09-07 anyway, and the flip is far worse than before, +3.8% of device_ms:
#   down   3328x8192x3072  sb 1x3 -> 1x5, 389.5 -> 383.8 us, 70.7% -> 71.8% of FPU  (a small WIN)
#   o_proj 3328x4096x3072  sb 1x3 -> 1x5, 212.0 -> 216.0 us, 64.9% -> 63.8%         (a small LOSS)
#   gate/up 3328x3072x8192 sb 1x6 -> 2x2, 361.0 -> 980.2 us, 76.1% -> 29.8%         (CATASTROPHIC)
#   qkv    3328x3072x6144  sb 1x6 -> 2x4, 304.7 -> 727.1 us, 68.7% -> 30.7%
# THE DISCRIMINANT IS N, NOT M.  Transposing gives the N blocks the Y extent (10) instead of the X
# extent (11), so a wide-N shape's per_core_N jumps -- 8192 tiles over 10 is 26 against 24 -- and at
# that width the out block no longer fits _CB_BUDGET_BYTES, so the core re-streams its whole in0
# slice once per N block.  The two shapes it helps are exactly the two whose N is 96 tiles, and
# their combined gain (-5.7 us and +4.0 us a layer) does not come close to paying for the two it
# wrecks.  A per-shape orientation search is therefore also not worth building: it would be worth
# about 1.7 us a layer.  The encoder confirms the same story from the other side -- every 1504-row
# shape fell to 100 cores and fc2 went 62.3 -> 107.0 us.
_TRANSPOSE_MCAST = False
_BLOCK_CFG_CACHE = {}


def _subblock(block_h, block_w, max_dest_tiles=8):
    for h, w in _SUBBLOCK_CHOICES:
        if h * w <= max_dest_tiles and block_h % h == 0 and block_w % w == 0:
            return h, w
    return 1, 1


def _cb_bytes(block_h, block_w, in0_block_w, tile_bytes, interm_bytes):
    """in0 + in1 double-buffered, plus the output block and its accumulator, for one core.

    Sized from the OUT BLOCK, not from what the core owns in total: the factory iterates out blocks
    and only one is resident, which is the whole reason out_block is a separate field from
    per_core_M/N.  Tying the two together is what made every prefill shape unservable -- at
    per_core 13x18 the output term alone is 734 kB.
    """
    return (
        2 * block_h * in0_block_w * tile_bytes
        + 2 * in0_block_w * block_w * tile_bytes
        + block_h * block_w * (tile_bytes + interm_bytes)
    )


def _divisors(n):
    return [d for d in range(int(n), 0, -1) if n % d == 0]


def _fused_activation(activation):
    """The program-config form of an activation name, for fusing a unary into the pack schedule.

    A FUSED GELU MUST BE THE APPROXIMATE ONE, WHICH IS NOT WHAT THE STRING MAPPING GIVES.  ttnn's
    `string_to_unary_with_param` maps "gelu" to GELU with its approximate flag FALSE, matching a
    standalone `ttnn.gelu`; fusing THAT form is a large net loss, because the exact erf runs
    inside the matmul's pack loop where it cannot be spread over the grid the way a standalone
    unary is.  Measured on the audio tower's fc1 (1504x1280x5120 bf8_b): 80.2 us/call bare, 209.9
    us/call with exact GELU fused -- 94 TFLOP/s against fc2's 303 on identical dtypes and flops --
    where the standalone unary it was meant to replace cost only 107 us.  The approximate flag is
    the form GUIDELINES/05 section 2 measures at +0.8 us over a bare matmul, so that is the one
    worth fusing, and the e2e PCC gate is what licenses the swap from erf to the tanh form.
    Returns None for a name this build cannot express, and the caller then keeps the standalone
    unary.
    """
    if activation is None:
        return None
    name = str(activation)
    try:
        if name in ("gelu", "gelu_approx"):
            return [ttnn.UnaryOpType.GELU, True]
        if name == "gelu_exact":
            return [ttnn.UnaryOpType.GELU, False]
        return ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, name.upper()))
    except (AttributeError, TypeError, ValueError):
        return None


def block_config(
    device, m_tiles, k_tiles, n_tiles, tile_bytes=1088, interm_bytes=2048, activation=None, max_dest_tiles=8
):
    """A 2-D mcast program config that spreads BOTH output dims and streams K in wide blocks.

    WHY THIS EXISTS: NAMING THE GRID PUTS THESE MATMULS ON THE 1-D MCAST PATH, WHICH IS THE WRONG
    SHAPE FOR THEM.  `ttnn.linear(core_grid=...)` routes to create_matmul_program_config, and that
    function splits ONE dimension across the whole grid and gives every core the FULL other one
    (matmul_program_config.cpp: `is_tall = m_tiles > n_tiles`, then either per_core_N = n_tiles or
    per_core_M = m_tiles), with `in0_block_w = div_up(k_tiles, num_cores)` -- which on these shapes
    is 1 or 2 tiles.  Measured consequence on the audio tower (M = 1504 = 47 tile rows):
      - qkv   1504x1280x3840  is_wide -> per_core_M = 47, per_core_N = 2,  in0_block_w = 1
      - fc1   1504x1280x5120  is_wide -> per_core_M = 47, per_core_N = 2,  in0_block_w = 1
      - fc2   1504x5120x1280  is_tall -> per_core_M = 1,  per_core_N = 40, in0_block_w = 2
    The tall ones then use only 47 of 110 cores (there are 47 M blocks), and every one of them
    iterates K 40 to 80 times for a single tile of reuse.  The profile agrees: 24-42% of LoFi FLOP
    peak with Output Subblock = 1x1, against 68% for the same model's prefill projections.

    WHAT IT PICKS.  Cost of a candidate, per core, in tile-matmul-equivalents:
        per_core_M * per_core_N * k_tiles / dest_reuse
      + (per_core_M/out_block_h) * (per_core_N/out_block_w) * (k_tiles/in0_block_w) * overhead
    `dest_reuse` = min(1, h*w/(h+w)) is the unpacker's side of the DEST block: an out_subblock of
    (1,1) needs two tile loads per tile-matmul and runs the FPU at half rate, (1,4) needs 1.25 and
    (2,4) is not load-bound at all.  The second term is the multicast and semaphore round the core
    pays per (out block, K block) pair, which is what makes a one-tile K block so expensive.
    Candidates that would leave a whole row or column of the grid with no work are rejected rather
    than scored, so the count is real occupancy and not grid size.

    THE MULTICAST ORIENTATION IS A SEPARATE AXIS, AND THE COST MODEL CANNOT SEE IT.  The scoring
    below is symmetric in the two output dims, so it has no opinion on which of them the factory
    multicasts along; only a measurement can choose.  With transpose_mcast the roles swap, so the M
    blocks have to fit the grid's X extent and the N blocks its Y extent -- which is why the search
    ranges swap with the flag rather than the flag simply being passed through.  On a non-square
    grid that is not cosmetic: at 11 x 10, M=112 tiles over 11 gives per_core_M 11 and N=256 over 10
    gives per_core_N 26, against 12 and 24 the other way round.

    Returns None when nothing fits, so the caller keeps its own path.
    """
    key = (
        int(m_tiles),
        int(k_tiles),
        int(n_tiles),
        int(tile_bytes),
        int(interm_bytes),
        str(activation),
        bool(_TRANSPOSE_MCAST),
        int(max_dest_tiles),
    )
    if key in _BLOCK_CFG_CACHE:
        return _BLOCK_CFG_CACHE[key]
    g = device.compute_with_storage_grid_size()
    # With transpose_mcast the factory counts M blocks along X and N blocks along Y, so the extent
    # each dim must fit inside swaps.
    m_extent, n_extent = (int(g.x), int(g.y)) if _TRANSPOSE_MCAST else (int(g.y), int(g.x))
    k_divs = _divisors(k_tiles)
    best = None
    for gy in range(1, m_extent + 1):
        per_core_m = -(-m_tiles // gy)
        if gy > 1 and per_core_m * (gy - 1) >= m_tiles:
            continue
        for gx in range(1, n_extent + 1):
            per_core_n = -(-n_tiles // gx)
            if gx > 1 and per_core_n * (gx - 1) >= n_tiles:
                continue
            for block_h in _divisors(per_core_m):
                for block_w in _divisors(per_core_n):
                    h, w = _subblock(block_h, block_w, max_dest_tiles)
                    reuse = min(1.0, (h * w) / float(h + w))
                    blocks = (per_core_m // block_h) * (per_core_n // block_w)
                    for in0_block_w in k_divs:
                        if _cb_bytes(block_h, block_w, in0_block_w, tile_bytes, interm_bytes) > _CB_BUDGET_BYTES:
                            continue
                        cost = (
                            per_core_m * per_core_n * k_tiles / reuse
                            + blocks * (k_tiles / in0_block_w) * _KBLOCK_OVERHEAD
                        )
                        work = per_core_m * per_core_n * k_tiles
                        cand = (cost, gx, gy, per_core_m, per_core_n, block_h, block_w, in0_block_w, h, w, work)
                        if (
                            best is None
                            or cost < best[0] * (1.0 - _COST_TIE)
                            or (cost < best[0] * (1.0 + _COST_TIE) and gx * gy > best[1] * best[2])
                            # MORE CORES FOR NO MORE PER-CORE WORK IS TAKEN WHATEVER `reuse` SAYS.
                            # The window above compares reuse-ADJUSTED cost, so it can only forgive a
                            # subblock penalty smaller than itself -- and on the two projections whose
                            # N is 96 tiles the penalty is bigger than that: 11 x 9 gives per_core
                            # 12x9 with a 2x3 subblock (reuse 1.0) and 11 x 10 gives 11x9 with 1x3
                            # (reuse 0.75), so the modelled cost rises 21% on a plan that does 8.3%
                            # LESS work per core across 11 MORE cores.  Narrowing _COST_TIE to hand
                            # the grid to the model's own favourite was already measured LOSING 5.5%
                            # of prefill on gate/up, which is the same trade one shape over: `reuse`
                            # overstates a 1xN subblock, so it must not be able to veto real cores.
                            # RAW work is the term the model is sure of -- it is the tile-matmul count
                            # -- so a wider grid is accepted whenever that term does not get worse.
                            or (gx * gy > best[1] * best[2] and work <= best[10])
                        ):
                            best = cand
                        break
    cfg = None
    if best is not None:
        _, gx, gy, per_core_m, per_core_n, block_h, block_w, in0_block_w, h, w, _work = best
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            # The grid is named in (x, y) order, so when the orientation is transposed the M blocks
            # -- searched against the X extent above -- are what x has to carry.
            compute_with_storage_grid_size=(gy, gx) if _TRANSPOSE_MCAST else (gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=h,
            out_subblock_w=w,
            out_block_h=block_h,
            out_block_w=block_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=_TRANSPOSE_MCAST,
            fused_activation=_fused_activation(activation),
        )
    _BLOCK_CFG_CACHE[key] = cfg
    return cfg


# Shapes whose 2-D config ttnn refused at runtime; they fall back for the rest of the run so a
# rejected shape costs one exception, not one per call.
_BLOCK_CFG_REFUSED = set()

# THE GUARD IS ABOUT OUTPUT TILES, AND THE ROW COUNT WAS ONLY EVER A PROXY FOR THEM.
# _GRID_REQUEST_MIN_ROWS exists to keep the DECODE shape off the 2-D path: at one tile row there is
# a single row of blocks, so no block config can buy DEST reuse and spreading the launch costs more
# than it recovers.  But `mm` already says in its own comment that the question is "enough OUTPUT
# tiles to keep a full grid busy, not rows alone", and the row form of the test excluded the
# multi-modal projector -- 384 rows against a 3072-wide weight is 12 x 96 = 1152 output tiles, about
# ten per core.  Those two matmuls therefore ran on the path block_config exists to replace: naming
# a core grid routes to create_matmul_program_config, which at 12 x 96 takes the is_wide branch,
# hands every core per_core_M = 12 / per_core_N = 1 with in0_block_w = div_up(160, cores) = 2, and
# so iterates K eighty times against a 1x1 DEST subblock.
# ADMIT A BELOW-THRESHOLD SHAPE ONLY ON BOTH COUNTS.  m_tiles > 1 keeps out every single-tile-row
# shape whatever its width -- the decode projections and the LM head, whose 1024 output tiles would
# otherwise satisfy an occupancy test on their own -- and the per-core floor keeps out anything too
# small to give each core a block to own.
_BLOCK_MIN_OUT_TILES_PER_CORE = 4


def _blocks_worth_it(device, m_tiles, n_tiles):
    """Whether a shape under the row threshold still has enough output blocks to spread."""
    g = device.compute_with_storage_grid_size()
    return m_tiles > 1 and m_tiles * n_tiles >= int(g.x) * int(g.y) * _BLOCK_MIN_OUT_TILES_PER_CORE


def _block_linear(x, weight, compute_kernel_config, bias=None, activation=None, memory_config=None, rows=None):
    """ttnn.linear under a 2-D mcast block config, or None when this shape cannot take one.

    Shared by `mm` (the attention projections) and `linear` (the FFN), because the reason for
    wanting it is the same at both call sites and only one of them used to have it: naming a core
    grid routes to create_matmul_program_config, which splits ONE output dim across the grid, hands
    every core the whole other one, and sets in0_block_w = div_up(k_tiles, cores) -- one or two
    tiles.  See block_config for the measurements.

    The guards are the ones `mm` already established: a weight that is not tile-aligned has no exact
    block decomposition, and below _GRID_REQUEST_MIN_ROWS the shape is presumed to be the decode one,
    where spreading a single tile row costs more launch than it recovers -- see _blocks_worth_it for
    the output-tile test that lets a genuinely wide sub-threshold shape through anyway.  A shape ttnn
    refuses is remembered so the exception is paid once.

    A LEADING BATCH IS FOLDED, NOT REFUSED.  Handing the 2-D factory a stacked activation makes it
    run the whole grid once PER BATCH ENTRY, so the blocks end up sized against a height the kernel
    never sees in one pass -- measured on this model's 8-stream prefill at 0.8% WORSE, which is why
    `mm` used to skip batched shapes outright.  But skipping is not the only answer: when the M dim
    is tile-aligned, [B, M, K] and [1, B*M, K] have the IDENTICAL physical tile order (batch b's
    tile row i sits at (b*M/32 + i) whichever way it is labelled), so the fold is a metadata view
    and the matmul becomes ONE pass over a B-times-taller activation against ONE read of the
    weight.  The attention projections already did this inside their own stubs, which is why they
    reached the block path and the FFN -- the majority of the LM's prefill flops -- never did.
    """
    if rows is None:
        rows = 1
        for d in tuple(x.shape)[:-1]:
            rows *= int(d)
    wshape = tuple(weight.shape)
    if int(wshape[-2]) % TILE or int(wshape[-1]) % TILE:
        return None
    m_tiles = math.ceil(rows / TILE)
    k_tiles, n_tiles = int(wshape[-2]) // TILE, int(wshape[-1]) // TILE
    if rows < _GRID_REQUEST_MIN_ROWS and not _blocks_worth_it(x.device(), m_tiles, n_tiles):
        return None
    dims = [int(d) for d in x.shape]
    batch = 1
    for d in dims[:-2]:
        batch *= d
    folded = x
    if batch != 1:
        if len(dims) < 3 or dims[-2] % TILE:
            return None
        folded = ttnn.reshape(x, (1, rows, dims[-1]))
    key = (m_tiles, k_tiles, n_tiles, str(activation))
    if key in _BLOCK_CFG_REFUSED:
        return None
    # THE DEST BUDGET IS THE CALLER'S TO DECLARE, NOT A CONSTANT.  An out subblock lives in DEST, and
    # fp32 accumulation halves how many tiles fit there -- so a caller that asks for fp32_dest_acc_en
    # cannot be handed the 8-tile subblock the search assumes by default.  Read it off the config
    # that will actually run the kernel rather than making the search guess.  Every projection on
    # this path today accumulates in fp16, so this changes no existing plan; it is what makes an
    # fp32-DEST caller expressible at all.
    cfg = block_config(
        x.device(),
        m_tiles,
        k_tiles,
        n_tiles,
        activation=activation,
        max_dest_tiles=4 if getattr(compute_kernel_config, "fp32_dest_acc_en", False) else 8,
    )
    if cfg is None:
        return None
    # AN OUTPUT PLACEMENT THE ALLOCATOR REFUSES IS NOT A REFUSED BLOCK CONFIG.  A caller asking for
    # an L1 hand-off is asking for something the block path is free to decline on its own; conflating
    # the two would blacklist the SHAPE on the first tight allocation and send every later call --
    # including the ones that never wanted L1 -- down the 1-D mcast path this helper exists to avoid.
    # So an L1 request is retried in DRAM first, and only a failure with no placement left standing
    # is recorded against the shape.
    placements = [memory_config] if memory_config is None else [memory_config, None]
    for placement in placements:
        try:
            out = ttnn.linear(
                folded,
                weight,
                bias=bias,
                compute_kernel_config=compute_kernel_config,
                program_config=cfg,
                # THE OUTPUT WIDTH FOLLOWS THE REGIME, NOT THIS PATH.  `mm` and `linear` both narrow
                # the output only at or above _GRID_REQUEST_MIN_ROWS -- "narrow the output only where
                # there are bytes to save" -- so hard-coding _ACT_DTYPE here meant admitting a
                # smaller shape to the 2-D path silently also narrowed its stored activation.  That
                # bundles a dtype change into what is meant to be a pure scheduling lever, and this
                # model's e2e PCC budget is measured in thousandths; the block config has to be
                # attributable on its own.
                dtype=_ACT_DTYPE if rows >= _GRID_REQUEST_MIN_ROWS else None,
                memory_config=placement,
            )
            break
        except (RuntimeError, TypeError, ValueError):
            if placement is None:
                _BLOCK_CFG_REFUSED.add(key)
                return None
    # Hand the caller back the rank it gave us; same view argument as the fold above.
    return out if batch == 1 else ttnn.reshape(out, tuple(dims[:-1]) + (int(wshape[-1]),))


def mm(
    device,
    x,
    weight,
    compute_kernel_config=None,
    bias=None,
    mirror=None,
    keep_sharded=False,
    activation=None,
    memory_config=None,
    out_dtype=ttnn.bfloat16,
):
    """ttnn.linear routed by the height of the activation: one call site, both regimes.

    `activation` IS APPLIED BY THE MATMUL, NOT AFTER IT.  A standalone unary on the FFN's wide
    intermediate is a full-width read-modify-write of a tensor the matmul had just finished
    writing -- measured on the audio tower, `ttnn.gelu` on a [1504, 5120] bf8_b activation cost
    107 us/layer at 153 GB/s, moving 16 MB that were already in the packer's hands.  Fusing it
    into the program config's `fused_activation` runs it inside the pack schedule instead
    (GUIDELINES/05 section 2).  The caller must NOT also apply it afterwards.

    `mirror` (optional) is the DRAM-bank-sharded decode copy of `weight`; when it serves the shape
    it wins, because at one tile row the cost is entirely the weight read.  Otherwise:

    These projections profile grid=partial with no grid hint at all, so ttnn routes them on its
    own.  Asking for the grid by name is the safe form of the occupancy lever: ttnn still sizes
    the blocks, so it cannot produce the L1 circular-buffer overflow a hand-written
    program_config does at these widths.

    GATED ON HEIGHT.  The grid request is regime-dependent: it wins on prefill (thousands of
    rows to spread) and LOSES on decode, where a single tile row spread over 110 cores costs more
    launch than it recovers.  Only ask when there is at least one tile row per core.
    """
    g = device.compute_with_storage_grid_size()
    m = 1
    for d in tuple(x.shape)[:-1]:
        m *= d
    if mirror is not None and bias is None and mirror.serves(math.ceil(m / TILE)):
        # keep_sharded is inert off the mirror path: prefill never reaches here (serves() is
        # decode-only), so a caller can ask for it unconditionally at a shared call site.  The
        # activation stays a standalone unary here for the reason `linear` documents: the
        # DRAM-sharded factory sends anything but RELU down a separate DEST path, which at one
        # tile row costs more than the interleaved unary it replaces.
        return _apply(
            activation,
            mirror(
                x,
                compute_kernel_config=compute_kernel_config,
                keep_sharded=keep_sharded,
                out_dtype=out_dtype,
            ),
        )
    # THE THRESHOLD WAS "ONE TILE ROW PER CORE", WHICH IS TOO STRICT.  Asking for the grid loses
    # at the decode shape (a single tile row spread over 110 cores costs more launch than it
    # recovers) but the break-even is nowhere near one row PER CORE -- the audio tower's FFN runs
    # 1504 rows (47 tile rows) against a 5120-wide weight, which is thousands of output tiles to
    # spread, and the old bound (110 * 32 = 3520 rows) silently excluded it.  Ask once there are
    # enough tile rows to keep a full grid busy on the OUTPUT tiles, not on the rows alone.
    # NARROW THE OUTPUT ONLY WHERE THERE ARE BYTES TO SAVE.  _GRID_REQUEST_MIN_ROWS already marks
    # the boundary between the two regimes -- below it is the decode shape, whose activation is
    # tens of kB and whose consumers all want the shard, not a narrower dtype.
    if m < _GRID_REQUEST_MIN_ROWS:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            compute_kernel_config=compute_kernel_config,
            activation=_linear_activation(activation),
        )
    # ASK FOR THE BLOCKS, NOT JUST THE GRID -- see block_config for why naming the grid alone lands
    # these on the 1-D mcast path with a one-tile K block and 1x1 DEST subblocks.  Best-effort: a
    # shape ttnn refuses (an L1 circular-buffer overflow, a batch it will not broadcast) falls back
    # to the grid request below.  M IS THE PADDED HEIGHT, NOT THE LOGICAL ONE -- the audio tower's
    # activation is 1500 rows, which the matmul pads to 47 tiles.
    blocked = _block_linear(
        x, weight, compute_kernel_config, bias=bias, activation=activation, memory_config=memory_config, rows=m
    )
    if blocked is not None:
        return blocked
    return ttnn.linear(
        x,
        weight,
        bias=bias,
        compute_kernel_config=compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=g.y, x=g.x),
        activation=_linear_activation(activation),
        dtype=_ACT_DTYPE,
        memory_config=memory_config,
    )


# --------------------------------------------------------------- fused QKV + head split
# THE HEAD SPLIT WAS THE MOST EXPENSIVE DATAMOVE IN THE MODEL.  `reshape(x, (B, S, nh, hd))`
# splits the LAST dim, which reorders data inside every tile, so on TILE layout it is a full
# re-tilization rather than a view -- measured 373 us/call on a 1504x1280 activation, ~10 GB/s.
# The pair (reshape + transpose) runs three times per attention, and ttnn has one op that does
# the whole thing: nlp_create_qkv_heads.  It wants the projections FUSED, which is what we want
# anyway (GUIDELINES/03 section 1) -- one weight read and one launch instead of three.


def fuse_qkv(qw, kw, vw, qb=None, kb=None, vb=None, scale=1.0):
    """Concatenate three [in, out] projections into one fused weight (+ bias), on the host.

    `scale` folds the attention scaling into Q so the runtime multiply disappears; the caller then
    keeps passing scale=1.0 to SDPA exactly as before.  Missing biases become zeros, because the
    fused matmul has one bias or none.
    """
    import torch

    w = torch.cat([qw * scale, kw, vw], dim=-1)
    if qb is None and kb is None and vb is None:
        return w, None

    def _b(bias, width, factor=1.0):
        return torch.zeros(width, dtype=w.dtype) if bias is None else bias.reshape(-1) * factor

    b = torch.cat(
        [_b(qb, qw.shape[-1], scale), _b(kb, kw.shape[-1]), _b(vb, vw.shape[-1])],
        dim=-1,
    )
    return w, b


# Bytes of q+k+v under which the head split leaves its three outputs in L1 for SDPA -- see below.
_SDPA_L1_MAX_BYTES = 24 * 1024 * 1024
# Bytes of K PLUS V under which they are moved into L1 on their own when the three-way budget above
# refused the split.  Counted in the REAL dtype, unlike _SDPA_L1_MAX_BYTES: this cap is being derived
# here rather than inherited, and the failure it has to stay clear of is measured -- 23.4 MB of real
# L1 tenancy at this point in the layer makes the next rms_norm unable to place its dataflow buffers.
# K and V are the only two operands flash re-reads, and on a grouped-query model they are a small
# fraction of the split (8 kv heads against 32 q heads), so the pair fits far inside that limit.
_KV_L1_MAX_BYTES = 12 * 1024 * 1024


# Bytes an attention OUTPUT may have before L1 residency stops paying -- see attn_out_config.  Sized
# for ONE tenant, not three: `release` has already dropped q/k/v by the time the head concat runs, so
# the widest moment on this path is flash's own output beside the concat's, and at the LM prefill
# shape that is 14.5 MB, i.e. 132 kB on each of 110 cores.
_ATTN_OUT_L1_MAX_BYTES = 16 * 1024 * 1024
# (shape) pairs the allocator refused, so the fallback to DRAM is paid once rather than per call.
_ATTN_OUT_L1_REFUSED = set()


def rows_of(t):
    """Everything but the last dim, multiplied out -- the height a matmul actually sees."""
    n = 1
    for d in tuple(t.shape)[:-1]:
        n *= int(d)
    return n


def qkv_out_config(rows, width, dtype=None):
    """Interleaved L1 for the FUSED QKV projection, whose only consumer is the head split.

    THE SPLIT MOVED TO L1 AND ITS INPUT DID NOT.  nlp_create_qkv_heads now writes q/k/v into L1, but
    it was still READING the projection out of DRAM -- so the fused [B, S, (nh + 2*nkv)*hd] value
    was written through the DRAM controller and pulled straight back by the very next op.

    Shares _SDPA_L1_MAX_BYTES because it is the same tensor twice over: the projection is exactly
    the size of the three outputs it becomes, and the two coexist only for the duration of the split
    itself -- the caller releases the projection the moment it returns.  Gated on the prefill height
    so the decode path (which goes through the DRAM-sharded mirror and ignores this argument anyway)
    cannot be reached by it.  Returns None rather than DRAM_MEMORY_CONFIG so a caller that does not
    apply keeps the exact call it made before.
    """
    if int(rows) < _GRID_REQUEST_MIN_ROWS:
        return None
    width_bytes = _DTYPE_BYTES.get(dtype if dtype is not None else _ACT_DTYPE, 2)
    return ttnn.L1_MEMORY_CONFIG if int(rows) * int(width) * width_bytes <= _SDPA_L1_MAX_BYTES else None


def attn_out_config(t):
    """Interleaved L1 for flash's output and the concat that reads it, or None to leave it in DRAM.

    THE OPERANDS MOVED TO L1 AND THE RESULT DID NOT.  q/k/v come out of the head split in L1 now,
    but flash still defaulted its output to DRAM, so the attention wrote [b, nqh, s, hd] out through
    the DRAM controller, concatenate_heads read it back and wrote the same bytes again, and o_proj
    read them a third time -- three full passes over a value whose only consumers are the two ops
    after it.  Returns None rather than DRAM_MEMORY_CONFIG so a caller that does not apply keeps the
    exact call it made before.
    """
    shp = tuple(int(d) for d in t.shape)
    if shp in _ATTN_OUT_L1_REFUSED:
        return None
    return ttnn.L1_MEMORY_CONFIG if _tensor_bytes(t) <= _ATTN_OUT_L1_MAX_BYTES else None


def sdpa_prefill(q, k, v, *, scale, program_config, compute_kernel_config, causal=True):
    """Flash attention over a full sequence, landing its output where the head concat will read it.

    `causal` DEFAULTS TO THE LM's ANSWER BUT IS THE CALLER'S TO GIVE.  The placement this helper
    exists for -- flash writing L1 so the concat and the output projection do not each pull the
    same bytes back through the DRAM controller -- has nothing to do with the mask, but the flag
    was baked in, so the audio tower (a bidirectional encoder, is_causal=False) could not reach it
    and kept the three-pass DRAM chain the docstring on attn_out_config describes. Widening the
    contract rather than copying the body keeps ONE place where the L1 request and its refusal
    cache live.
    """
    mem = attn_out_config(q)
    if mem is not None:
        try:
            return ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                scale=scale,
                memory_config=mem,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
        except RuntimeError:
            # The budget above counts this tensor; it cannot see the trace region, the resident
            # caches or flash's own circular buffers, so the placement is a request.
            _ATTN_OUT_L1_REFUSED.add(tuple(int(d) for d in q.shape))
    return ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def concat_heads(attn_out):
    """`concatenate_heads`, keeping the projection's activation in L1 when flash left one there."""
    mem = attn_out_config(attn_out)
    if mem is not None:
        try:
            return ttnn.transformer.concatenate_heads(attn_out, memory_config=mem)
        except (RuntimeError, TypeError):
            _ATTN_OUT_L1_REFUSED.add(tuple(int(d) for d in attn_out.shape))
    return ttnn.transformer.concatenate_heads(attn_out)


def release(*tensors):
    """Drop device tensors whose last consumer has just run, ignoring anything already gone.

    A ttnn tensor is freed when its buffer is deallocated, not when the Python name goes out of
    scope, and a decoder body binds its q/k/v to loop LOCALS -- so the split stays resident until
    the next iteration rebinds the name, which is four ops later than its last read.  That is the
    whole reason the three-way L1 split did not fit: the tenancy being measured was not the
    attention's, it was the attention's plus everything that runs after it in the same iteration.
    Tolerant by design, so a caller may hand it a tensor another path already released.
    """
    for t in tensors:
        if t is None:
            continue
        try:
            ttnn.deallocate(t)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass


def _tensor_bytes(t):
    n = 1
    for d in tuple(t.shape):
        n *= int(d)
    return n * _DTYPE_BYTES.get(t.dtype, 2)


def qkv_heads(qkv, num_heads, num_kv_heads=None):
    """Split a fused [B, S, (nh + 2*nkv) * hd] projection into per-head Q/K/V in ONE op.

    LAND THEM IN L1 WHEN THEY FIT, because flash attention does not read K and V once.  The kernel
    loops q chunks on the OUTSIDE and streams the whole of K and V for each one, so at the audio
    tower's 1504-long sequence with 256-wide chunks every K and V tile is pulled six times over:
    3.85 MB each becomes ~46 MB of reads per call against a 3.85 MB tensor.  Through the DRAM
    controller that is most of the op (measured 301 us/call at ~166 GB/s effective); out of L1 the
    workers reach the same tiles over the NOC instead.  It costs NOTHING extra -- this is the
    memory_config the split already had to choose, not a new op -- and it is pure placement, so the
    values are bit-identical.  Gated on size so a shape that would not fit degrades to DRAM rather
    than crowding out SDPA's own circular buffers.
    """
    dims = [int(qkv.shape[i]) for i in range(len(qkv.shape))]
    b, s, w = dims[0], dims[-2], dims[-1]
    # CHARGED IN THE REAL DTYPE, AND THE THING THAT MADE THAT UNAFFORDABLE IS FIXED.  The `* 2` here
    # was a bf16 width on a stream that is _ACT_DTYPE, so the LM's prefill split was charged 44 MB
    # against a 24 MB cap when it really occupies 23.4 MB.  Correcting it alone was measured a loss
    # on 2026-09-06: the split fitted, and then the NEXT rms_norm raised "statically allocated
    # dataflow buffers in program 156 clash with L1 buffers on core range [0-0 - 10-9]" -- an L1
    # buffer at 876032 against a dataflow region ending at 896512, i.e. short by 20 kB.
    #
    # It was short because q/k/v OUTLIVE the attention.  The caller binds them to loop locals and
    # Python does not drop those until the next iteration rebinds them, so 213 kB/core of split that
    # is dead the instant SDPA returns was still resident through concatenate_heads, o_proj, the
    # residual add and that norm.  `release` below hands the callers a way to end those lifetimes at
    # the op that actually ends them, which is what buys the 20 kB back and a great deal more.
    #
    # What the placement is worth: the split writes its 21.7 MB into L1 instead of DRAM, the two
    # separate K/V copies below disappear (they exist only to rescue the operands flash re-reads),
    # both ropes then read AND write L1 (rotary_embedding_hf defaults its output to its input's
    # config), and SDPA reads all three operands out of Tensix banks over the NoC.
    mem = (
        ttnn.L1_MEMORY_CONFIG
        if b * s * w * _DTYPE_BYTES.get(qkv.dtype, 2) <= _SDPA_L1_MAX_BYTES
        else ttnn.DRAM_MEMORY_CONFIG
    )
    folded = ttnn.reshape(qkv, (b, 1, s, w))
    kwargs = dict(
        num_heads=num_heads,
        num_kv_heads=num_heads if num_kv_heads is None else num_kv_heads,
        transpose_k_heads=False,
    )
    try:
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(folded, memory_config=mem, **kwargs)
    except RuntimeError:
        # A SHAPE THE ALLOCATOR REFUSES IN L1 STILL HAS TO SPLIT.  The budget above counts the three
        # outputs; it cannot see the trace region, the resident caches or this op's own buffers, so
        # the placement is a request rather than a guarantee and the DRAM form is the same op.
        if mem is ttnn.DRAM_MEMORY_CONFIG:
            raise
        mem = ttnn.DRAM_MEMORY_CONFIG
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(folded, memory_config=mem, **kwargs)
    if mem is ttnn.DRAM_MEMORY_CONFIG:
        # THE SPLIT DID NOT FIT, BUT THE TWO OPERANDS THAT ARE RE-READ DO.  q is read once per call;
        # K and V are read once per (q chunk, q head), and this model is grouped-query, so each kv
        # head is pulled by four q heads and again by every q chunk.  On the LM's prefill shape that
        # turns a 7.8 MB pair into ~47 MB of DRAM reads and is what holds this op at ~183 GB/s
        # effective while the residual adds one line away reach 412.  Moving ONLY the pair costs two
        # launches over 7.8 MB and leaves 71 kB/core resident -- a third of the footprint that made
        # the three-way placement unplaceable -- so it buys most of the re-read back at a fraction
        # of the L1 the whole split wanted.  Pure placement: the values are bit-identical, and a
        # refusal falls back to the DRAM tensors the split already produced.
        if _tensor_bytes(k) + _tensor_bytes(v) <= _KV_L1_MAX_BYTES:
            try:
                k_l1 = ttnn.to_memory_config(k, ttnn.L1_MEMORY_CONFIG)
                v_l1 = ttnn.to_memory_config(v, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                k, v = k_l1, v_l1
            except RuntimeError:
                pass
    return q, k, v


_SDPA_CFGS = {}
_SDPA_MAX_CHUNK = 256
# Fraction of the grid the flash work units must keep busy before sdpa_config stops halving the q
# chunk.  2/3 is the point where one more halving stops paying: it doubles the per-unit loop and CB
# overhead to recover less than half a round.
_SDPA_MIN_OCCUPANCY = 2.0 / 3.0
# The widest k block that is worth asking for, and the q_chunk * k_chunk budget it has to live
# inside.  32768 is not a guess: 128 x 256 is the shape the audio tower runs today and 128 x 512
# is the shape that raised "circular buffers in program 74 clash with L1 buffers" by 3968 B, so
# the fitting product is bounded above by twice the current one and below by the current one.
# Anything at or under the current product is known to fit.
# 512 IS AN OPTIMUM, NOT A CEILING TO KEEP RAISING.  At the same 32768 product the next step along
# the same gradient -- q 32, k 1024, which trades two more halvings of q for one more doubling of k
# and even raises occupancy to 95% -- measured encode 20.00 -> 24.80 ms, far WORSE.  So "wider k is
# better" holds only while the q chunk still has enough rows to amortise its own setup; at 32 rows
# (one tile) the per-unit cost the q chunk was supposed to be cheap at dominates instead.  Both
# neighbours of 64 x 512 are measured and both are worse: 128 x 256 is +6.9% and 32 x 1024 is +24%.
_SDPA_MAX_K_CHUNK = 512
_SDPA_CHUNK_PRODUCT = 32768
# Tile-equivalents one (q chunk, k chunk) PAIR costs a core in loop setup, circular-buffer turns and
# the running-max/running-sum rescale, over and above the qc*kc/1024 tiles of score it computes.
# It only has to RANK the causal candidates below against each other, so its exactness matters less
# than the fact that it is non-zero -- without it the area model would always pick the smallest chunk,
# which is the very thing that makes ttnn's one-tile default disastrous on a long sequence.
#
# FITTED, NOT GUESSED, AND IT IS DOING TWO JOBS.  Two points on this model's LM prefill
# (b8 h32 s416) pin it: 256x256 costs 192 score tiles + 3 pairs and measures 288.4 us/call, 128x128
# costs 160 + 10 and measures 256.4, which solves to 1.469 us a score tile and 2.14 us a pair -- a
# ratio of 1.46 tiles a pair.  It was first set to 3.0 by eye, which is enough to prefer 128 over
# 256 but not enough to reach the pick below.
#
# WHAT IT IS STANDING IN FOR.  Narrowing the Q chunk also multiplies the GRID ROUNDS, and that cost
# is not a pair cost.  The third measured point shows it: 64x128 (128 tiles, 16 pairs, 17 rounds
# against 128x128's 10) came in at 249.8 us -- a real win, but only -2.6% where the pair-only model
# had predicted -13%.  Three samples cannot separate the two terms (pairs and rounds move together
# across them: an exact 3-term fit returns a NEGATIVE pair cost, which is over-fitting, not physics),
# so the rounds cost stays folded into this constant.  The consequence to respect: this number
# UNDER-prices narrowing q, so treat a pick it makes below 64 as unmeasured rather than trusted.
_SDPA_CAUSAL_PAIR_OVERHEAD = 1.46
# How many k chunks one q chunk may visit.  THIS IS THE PCC KNOB, AND IT IS THE ONE THE TWO CHUNK
# AXES DO NOT SHARE.  Flash rescales its running max and running sum once per k chunk it folds in,
# so the DEPTH of that chain -- ceil(seq_k / k_chunk) -- is what costs accuracy, and it depends on
# the K chunk alone.  Measured: going 256x256 -> 128x128 took the depth 2 -> 4 and e2e PCC
# 0.9636 -> 0.9601 against a 0.95 gate.  Narrowing the Q chunk instead cuts score area at CONSTANT
# depth, which is why the search below is over both axes rather than one square chunk.  Held at the
# depth already paid for, so the area search cannot quietly spend more accuracy.
#
# HOLDING IT AT 4 IS NOT COSTING MUCH, WHICH IS WHY IT CAN STAY A GUARD RATHER THAN A SWEEP.  The
# candidates it blocks here are k=64 (depth 7), and on the honest reading they are a bad trade
# anyway: 128x64 takes the score area only 160 -> 152 tiles (-5%) while very nearly doubling the
# pairs 10 -> 19, and 64x64 gets 128 -> 112 (-12.5%) for 16 -> 28.  Area falls slowly on the K axis
# because the diagonal pair is the only one a narrower k actually trims.  Untested rather than
# refuted -- but the accuracy is on the line and the modelled saving is not there.
_SDPA_CAUSAL_MAX_RESCALE = 4


def _causal_cost(seq_q, seq_k, qc, kc):
    """(pairs, score tiles, rescale depth) for a causal flash pass at this (q, k) chunk pair.

    Flash skips a (q chunk, k chunk) pair only when the WHOLE pair is above the diagonal, so q chunk
    i still visits every k chunk overlapping rows 0..(i+1)*qc -- including the diagonal pair whose
    upper triangle the mask throws away.  Score area therefore falls as EITHER chunk narrows, but
    the two axes buy it at different prices: narrowing q multiplies the number of independent work
    units (rounds of the grid), narrowing k multiplies the rescale depth (accuracy).
    """
    pairs = 0
    area = 0
    for i in range(-(-int(seq_q) // int(qc))):
        visited = -(-min((i + 1) * int(qc), int(seq_q)) // int(kc))
        pairs += visited
        area += visited * int(qc) * int(kc)
    return pairs, area / float(TILE * TILE), -(-int(seq_k) // int(kc))


def sdpa_config(device, q, k, wide_k=False, causal=False):
    """Full-grid SDPAProgramConfig with tile-power-of-two flash chunks, sized from q/k.

    ttnn's SDPA falls back to q_chunk_size = k_chunk_size = 32 -- ONE TILE -- whenever no
    program_config is passed (sdpa_program_factory.cpp reads `program_config ? ... : 32`).
    That turns the audio tower's 1504-long sequence into 47 x 47 = 2209 flash chunk-pair
    iterations of a single tile each, so the kernel spends its time on loop and circular-buffer
    overhead rather than on the 11.6 GFLOP it is there to do.  Chunk sizing is the single most
    important SDPA knob (GUIDELINES/04 section 3); 256 is the catalogued winner for long
    encoder sequences, and it still leaves enough flash work units (batch * heads * q_chunks)
    to fill the grid.

    Chunks are capped at the largest power of two that does not exceed the sequence, so a short
    sequence keeps small chunks instead of padding itself up into wasted work.

    exp_approx_mode is left exact, but NOT for the reason this comment used to give.  It claimed
    the approximate exp "costs PCC without being faster"; that was asserted, never measured.
    Measured 2026-09-05, scoped per-caller to the six audio-tower bodies (the same split that
    already scopes this tower's SDPA to LoFi, so the LM's attention kept the exact exp): PCC went
    0.9580 -> 0.9636, i.e. BETTER than baseline, and encode went 21.3734 -> 21.4023 ms, i.e. did
    not move.  So the PCC half of the old claim is false and only the speed half holds.  What that
    rules out is the theory behind trying it: flash re-exponentiates its score block on the SFPU
    once per k chunk, and this kernel runs at 48 TFLOP/s -- 6% of block-float peak -- so the exp
    looked like the thing it was bound on.  It is not.  The overhead is the chunk loop itself, so
    the lever that would pay here is one that removes ITERATIONS, not one that cheapens the
    exponential.  Do not spend another round on exp_approx_mode.

    BUT THE Q CHUNK IS ALSO THE UNIT OF WORK, AND THE WIDEST ONE NEED NOT FILL THE GRID.  Flash
    hands out batch * heads * ceil(seq_q / q_chunk) independent work units and runs them in
    ceil(units / cores) rounds, so a chunk that is too WIDE leaves the LAST round mostly idle: the
    audio tower at 20 heads and 1504 positions produces 20 * 6 = 120 units against 110 cores, so it
    takes two rounds to do 1.09 rounds of work and the second round is 10 busy cores and 100 idle
    ones.  Halving the chunk doubles the units and halves each one, which is why the fix is not
    "smallest chunk wins": per-unit loop and circular-buffer overhead is what made the stock 32
    disastrous here in the first place.  So keep the widest chunk and halve it ONLY while the rounds
    are badly under-filled -- occupancy at or above _SDPA_MIN_OCCUPANCY -- which trades a chunk that
    is still wide for a last round that is nearly full.  k_chunk is untouched: it is the inner loop,
    not a work unit.
    """
    seq_q = int(q.shape[-2])
    seq_k = int(k.shape[-2])
    dims = [int(d) for d in q.shape]
    units = 1
    for d in dims[:-2]:
        units *= d

    def _chunk(s):
        return max(32, min(_SDPA_MAX_CHUNK, 1 << (max(int(s), 1).bit_length() - 1)))

    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y

    def _q_chunk():
        c = _chunk(seq_q)
        while c > 32:
            n = units * -(-seq_q // c)
            if n / (-(-n // cores) * cores) >= _SDPA_MIN_OCCUPANCY:
                break
            c //= 2
        return c

    qc, kc = _q_chunk(), _chunk(seq_k)
    if causal:
        # A CAUSAL PASS PAYS FOR AREA THE MASK THROWS AWAY, SO OCCUPANCY IS NOT THE ONLY AXIS.
        # `_q_chunk` above sizes the chunk so the LAST round of work units is nearly full, which is
        # the right rule for the audio tower -- that attention is NOT masked, so every candidate
        # computes the same total score area and only the round filling differs.  The LM prefill is
        # masked, and there the chunk decides how much score the kernel computes at all: at 416
        # positions a 256 chunk evaluates 3 pairs of 256x256 = 196 k score elements against a true
        # causal area of 86 k, i.e. 2.27x the necessary work, because the diagonal pair is half
        # wasted and the second q chunk is 96 rows of padding.  Halving to 128 makes it 10 pairs of
        # 128x128 = 164 k (1.89x) and still leaves occupancy at 0.93.
        #
        # Both terms move, so it is a minimisation rather than "smaller is better": the score tiles
        # fall toward the true triangle while the pair count -- and _SDPA_CAUSAL_PAIR_OVERHEAD with
        # it -- grows as n^2.  Search only chunks the occupancy rule would also accept, so this can
        # narrow the chunk but never un-fill the grid.
        #
        # AND THE TWO CHUNKS ARE SEPARATE AXES, WHICH A SINGLE SQUARE CHUNK CANNOT EXPRESS.  Area
        # falls as either one narrows, but they are paid for out of different budgets: narrowing Q
        # multiplies the WORK UNITS (so it is the occupancy rule's business), narrowing K multiplies
        # the RESCALE DEPTH (so it is the PCC gate's business -- see _SDPA_CAUSAL_MAX_RESCALE).  At
        # 416 positions the square pick 128x128 costs 160 score tiles at depth 4, while 64x128 costs
        # 128 -- a further 20% off the area for NO extra depth, bought only with 6 more pairs. So the
        # search runs over both axes with the depth held at what has already been paid for.
        # The cap is never allowed to reject the plan we already have: a long enough context puts the
        # occupancy pick's own depth past the constant, and the point of the constant is "do not spend
        # MORE accuracy than is already being spent", not "refuse to run".
        max_depth = max(_SDPA_CAUSAL_MAX_RESCALE, -(-seq_k // kc))
        best = None
        for c_q in [qc >> i for i in range(16) if (qc >> i) >= 32]:
            n = units * -(-seq_q // c_q)
            if n / (-(-n // cores) * cores) < _SDPA_MIN_OCCUPANCY:
                continue
            for c_k in [kc >> i for i in range(16) if (kc >> i) >= 32]:
                pairs, tiles, depth = _causal_cost(seq_q, seq_k, c_q, c_k)
                if depth > max_depth:
                    continue
                cost = pairs * _SDPA_CAUSAL_PAIR_OVERHEAD + tiles
                if best is None or cost < best[0]:
                    best = (cost, c_q, c_k)
        if best is not None:
            qc, kc = best[1], best[2]
    if wide_k:
        # TRADE Q WIDTH FOR K WIDTH AT A CONSTANT PRODUCT.  The k loop is where this kernel's
        # overhead lives, so a wider k block is the shape worth having; it just cannot be bought
        # with extra L1, because the next step up in product does not fit.  Widening k and paying
        # for it out of q keeps the circular buffers inside the budget that is known to fit.
        kc = max(32, min(_SDPA_MAX_K_CHUNK, 1 << (max(int(seq_k), 1).bit_length() - 1)))
        while qc > 32 and qc * kc > _SDPA_CHUNK_PRODUCT:
            qc //= 2
        while kc > 32 and qc * kc > _SDPA_CHUNK_PRODUCT:
            kc //= 2
    key = (grid.x, grid.y, qc, kc)
    cfg = _SDPA_CFGS.get(key)
    if cfg is None:
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=key[2],
            k_chunk_size=key[3],
            exp_approx_mode=False,
        )
        _SDPA_CFGS[key] = cfg
    return cfg


def sdpa_decode_out_config():
    """Where sdpa_decode should leave its output at the decode shape: INTERLEAVED L1.

    The attention output is [1, B, padded_nh, hd] -- at B=8, 32 heads, 128 head_dim that is 65 kB,
    three orders of magnitude smaller than the KV cache the op just streamed.  Leaving it in DRAM
    costs a full controller round trip for a tensor that fits in a handful of L1 banks, and it is
    read TWICE more before it becomes anything: once by the head merge and once by o_proj's
    activation reshard.

    It must be INTERLEAVED, not sharded: sdpa_decode raises `Sharded output not supported for GQA`
    the moment a sharded memory_config is asked for on a grouped-query model (32 q heads over 8 kv
    heads).  L1 placement is the part of that lever GQA does NOT block -- it changes only which
    banks hold the result, so the values are bit-identical.

    THAT DOES NOT MAKE `nlp_concat_heads_decode` UNREACHABLE, which an earlier note here claimed.
    The GQA rule binds the PRODUCER only.  The concat op's own contract is just that ITS input be
    height-sharded one user per core with shard shape (padded_heads, head_dim), and an explicit
    to_memory_config builds exactly that from the interleaved L1 tensor sdpa_decode is allowed to
    write -- measured on device, it accepted all 1860 decode calls and returned [1,8,32,128] ->
    [1,1,32,4096].  What actually stops it is its OUTPUT: it pads the user dim up to a full tile
    (8 users -> 32 rows), so the result no longer carries the layer's logical [1, B, ...] batch and
    converting back costs a sub-tile batch slice that eats the ~6 us the swap saves.  To claim it,
    decode has to carry the PADDED 32-row batch as its logical shape end to end -- it is already 32
    rows physically -- rather than slicing back per layer.  See merge_heads_decode below.
    """
    return ttnn.L1_MEMORY_CONFIG


def _row_cores(device, n):
    """The first `n` cores of the compute grid as a CoreRangeSet, x fastest."""
    g = device.compute_with_storage_grid_size()
    full, rem = divmod(int(n), int(g.x))
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(int(g.x) - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


def _square_cores(device, n):
    """`n` cores as the SQUAREST single rectangle anchored at (0, 0), or None if n is prime-ish.

    For a gather whose every consumer reads from every one of these cores, the shape of the source
    rectangle is the NOC hop distance: `n` cores in one row put the far end `n-1` columns away,
    where the squarest factorisation halves the worst case.  Anchored at (0, 0) and returned as ONE
    range on purpose -- prim::nlp_concat_heads_decode flips to `on_subcoregrids` (a different
    program factory, which then REQUIRES an explicit sub_core_grids) the moment the input's first
    range does not start at the origin or there is more than one of them.
    """
    g = device.compute_with_storage_grid_size()
    n = int(n)
    best = None
    for y in range(1, int(g.y) + 1):
        if n % y:
            continue
        x = n // y
        if x > int(g.x):
            continue
        if best is None or max(x, y) < max(best):
            best = (x, y)
    if best is None:
        return None
    x, y = best
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def _slice_into(x, end, mirror):
    """Slice `x` down to `end`, landing directly in `mirror`'s activation shard when it will take it.

    Falls back to interleaved L1 -- the placement this slice has always used -- on any build or
    shape whose slice will not pack a width shard, so the caller never loses the dedicated head
    merge just because the placement was refused.
    """
    wanted = mirror.act_config(math.ceil(int(end[-2]) / TILE)) if mirror is not None else None
    if wanted is not None:
        try:
            return ttnn.slice(x, (0,) * len(end), end, memory_config=wanted)
        except (RuntimeError, TypeError, ValueError):
            pass
    return ttnn.slice(x, (0,) * len(end), end, memory_config=ttnn.L1_MEMORY_CONFIG)


def _concat_heads_decode(attn_out, batch, num_heads, head_dim, mirror=None):
    """The DEDICATED head merge, or None when this build/shape will not take it.

    THE GQA RULE BINDS THE PRODUCER, NOT THIS OP.  sdpa_decode refuses a sharded output on a
    grouped-query model, so its result arrives interleaved in L1 -- but nlp_concat_heads_decode
    only asks that ITS input be height-sharded one user per core with shard (padded_heads,
    head_dim), and an explicit to_memory_config builds exactly that.

    WHAT USED TO STOP IT IS THE OUTPUT, AND THAT IS A SUB-TILE CUT, NOT A WALL.  The op pads the
    user dim to a full tile (8 users -> 32 rows), so the result no longer carries the layer's
    logical [1, B, ...] batch and the residual add next door cannot broadcast against it.  Cutting
    the real users back out is one slice of a SINGLE tile row -- 32 x 4096, ~260 kB -- against a
    reshape that re-tilizes the whole tensor, so the question is which of the two costs more, and
    only a measurement answers it.

    Returns None on any refusal so the caller keeps the reshape; correctness never depends on this.
    """
    try:
        dev = attn_out.device()
        dims = [int(d) for d in attn_out.shape]
        if len(dims) != 4 or dims[0] != 1 or dims[1] != int(batch):
            return None
        padded_heads, hd = dims[2], dims[3]
        width = int(num_heads) * int(head_dim)
        # EVERY OUTPUT CORE READS EVERY ONE OF THESE CORES, so their rectangle is the hop distance.
        # The factory hands each of the `num_heads` output cores the whole noc_x/noc_y cross product
        # of this shard's bounding box and the reader gathers one sub-tile LINE (16 elements) per
        # user, so the gather is `num_heads x batch` tiny reads and latency, not bytes, is what it
        # costs.  A single 8-wide row puts the far end 7 columns away; the squarest rectangle for
        # the same core count halves that.  Falls back to the row when the count will not factor.
        shard = ttnn.create_sharded_memory_config(
            (padded_heads, hd),
            _square_cores(dev, batch) or _row_cores(dev, batch),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        xs = attn_out if attn_out.memory_config() == shard else ttnn.to_memory_config(attn_out, shard)
        # PREALLOCATE THE OUTPUT AND THE PADDING SLICE DISAPPEARS WITH IT.  This op ignores
        # `memory_config` outright (nlp_concat_heads_decode.cpp forwards it and prim:: drops it) and
        # builds its own spec: batch padded up to a full tile, width-sharded (padded_batch, head_dim)
        # over `num_cores_to_corerangeset(num_heads, compute_grid, row_wise)`.  That is why the merge
        # used to need a slice afterwards -- the padded batch is not the layer's logical one.  But
        # `output_tensor` short-circuits the whole spec: compute_output_specs returns the
        # preallocated tensor's spec verbatim, and the factory reads its core placement back out of
        # `output.shard_spec()` (nlp_concat_heads_decode_program_factory.cpp: `q_cores =
        # output.shard_spec().grid`, then `grid_to_cores(num_cores, bbox.x, bbox.y, true)`), so the
        # tensor we hand it decides BOTH the grid and the logical shape.  Hand it o_proj's own
        # activation shard with the LOGICAL batch: the op writes exactly the `batch` rows it always
        # wrote, the padded rows stay untouched as before, and the result already carries both the
        # shape the residual add needs and the placement the projection borrows -- so the sub-tile
        # slice AND the InterleavedToSharded that used to follow it are both gone.
        #
        # The specs coincide only when the projection's in0 rectangle happens to be `num_heads`
        # cores of `head_dim` columns each; check it rather than assume it, and fall through to the
        # slice when it does not hold, so this stays a placement lever and not a shape assumption.
        # The op pads the batch to a full tile (compute_output_specs: `batch = max(batch, 32)`), and
        # that padded height is what a one-tile-row activation shard already has -- but they are two
        # different quantities, so compare against the padding rule rather than against the input's
        # padded head count.
        padded_batch = max(int(batch), TILE)
        want = mirror.act_config(1) if mirror is not None else None
        prealloc = None
        if want is not None:
            spec = want.shard_spec
            if (
                want.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                and spec is not None
                and spec.grid.num_cores() == int(num_heads)
                and tuple(int(v) for v in spec.shape) == (padded_batch, int(head_dim))
            ):
                prealloc = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, int(batch), width]), xs.dtype, ttnn.TILE_LAYOUT, dev, want
                )
        joined = ttnn.experimental.nlp_concat_heads_decode(
            xs, num_heads=int(num_heads), output_tensor=prealloc
        )
        if xs is not attn_out:
            ttnn.deallocate(xs)
        jd = [int(d) for d in joined.shape]
        if jd[-2] != int(batch):
            # AND THE UNPADDING SLICE IS THE PLACE TO BUILD o_proj's ACTIVATION SHARD, not merely a
            # place to land in L1.  The slice has to write a new tensor whatever config it is given,
            # and the profile shows what interleaved costs: this slice lands L1-interleaved and
            # o_proj's very next act is an InterleavedToSharded on 32 cores, 0.80 us/call, once per
            # layer per token.  Naming the projection's own activation config makes its borrow check
            # compare equal and the reshard disappears into a write that was happening anyway.
            joined = _slice_into(joined, (1, 1, int(batch), width), mirror)
        return ttnn.reshape(joined, (1, int(batch), width))
    except (RuntimeError, TypeError, AttributeError, ValueError):
        return None


def merge_heads_decode(attn_out, batch, num_heads, head_dim, l1=True, mirror=None):
    """[1, B, padded_nh, hd] -> [1, B, nh*hd] for the o_proj, kept in L1.

    This reshape collapses the last TWO dims, so on TILE layout it genuinely re-tilizes rather than
    returning a view -- it is the single most expensive datamove left in the decode step (measured
    ~20 us/call, once per layer per token).  ttnn has a dedicated op for exactly this shape; see
    _concat_heads_decode for why it is reachable and what it costs to convert its padded batch back.
    Shared by every attention body so the swap reaches all of them rather than only the
    individually-routed layers.
    """
    shape = (1, int(batch), int(num_heads) * int(head_dim))
    if l1:
        swapped = _concat_heads_decode(attn_out, batch, num_heads, head_dim, mirror)
        if swapped is not None:
            return swapped
    if not l1:
        return ttnn.reshape(attn_out, shape)
    try:
        return ttnn.reshape(attn_out, shape, memory_config=ttnn.L1_MEMORY_CONFIG)
    except (RuntimeError, TypeError, AttributeError):
        # A width this board cannot hold interleaved in L1 -- fall back rather than fail the merge.
        return ttnn.reshape(attn_out, shape)


def qkv_split_decode(qkv, batch, num_heads, num_kv_heads, head_dim):
    """Split a fused DECODE-shape [1, B, (nh + 2*nkv)*hd] projection into three per-head tensors.

    Decode cannot use nlp_create_qkv_heads (that op wants a sequence axis, and decode has one
    tile row holding the batch), so the fused projection is cut with three last-dim slices at
    multiples of head_dim -- all tile-width multiples, so each is a straight tile copy of a few
    tens of KB.  That is far cheaper than the two extra matmul launches it replaces, and it is
    what makes fusing worthwhile even though the split itself is not free.

    WHY FUSING PAYS MOST AT DECODE: k_proj and v_proj are 3072x1024, i.e. 32 output tiles, and
    _plan cannot find a bank-worker count that divides 32 exactly -- so each of them FAILS to get
    a DRAM-sharded mirror at all and falls back to a plain ttnn.linear that measured 125 GB/s,
    under a third of peak.  The fused width is 6144 = 192 tiles, which divides across the bank
    workers exactly, so all three projections inherit the sharded path in one weight read.
    """
    q_w = num_heads * head_dim
    kv_w = num_kv_heads * head_dim
    dims = [int(d) for d in qkv.shape]
    m = 1
    for d in dims[:-1]:
        m *= d
    # DO NOT RESHAPE A TENSOR THAT IS ALREADY THE RIGHT SHAPE.  When the projection handed back its
    # L1 shard (mm(..., keep_sharded=True)) it is already (1, 1, m, width), and a reshape to the
    # shape it already has is not free -- it is a ReshapeView launch, and on a sharded tensor it
    # would also risk dropping the very shard spec that selects the fast program factory.
    flat = qkv if dims == [1, 1, m, dims[-1]] else ttnn.reshape(qkv, (1, 1, m, dims[-1]))

    # THE DEDICATED DECODE HEAD-SPLIT.  ttnn has one op for exactly this shape --
    # nlp_create_qkv_heads_decode takes [1, 1, batch, (nh + 2*nkv)*hd] TILE and returns the three
    # [1, batch, heads, hd] tensors directly.  It replaces the three slices below AND, more
    # importantly, the three reshapes that follow them: those split the LAST dim (4096 -> 32x128),
    # which re-tilizes, and they profiled at 10.2 us each, 3 per layer -- the largest per-layer
    # overhead left in the decode step.  Its outputs are also height-sharded over batch, which is
    # the layout paged_update_cache wants anyway.  Preconditions it asserts and we satisfy:
    # rank-4 with dims 0 and 1 == 1, batch <= 32, width a multiple of TILE_WIDTH, TILE layout,
    # bf16/fp32.  Kept behind a fallback so correctness never depends on the op accepting them.
    if m <= 32:
        try:
            return ttnn.experimental.nlp_create_qkv_heads_decode(
                flat,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                # k's CORE RANGE WRAPS THE GRID ROW, AND THAT COSTS NOTHING -- MEASURED.  The op
                # lays q and k out in ROW-MAJOR order inside the core grid of the output_mem_config
                # it is handed, starting k where q's (batch+1)'th core would be
                # (nlp_create_qkv_heads_decode_device_operation.cpp: k_start_core_coord is the end
                # of the batch+1 range).  Defaulted, that grid is the whole 11 x 10 compute grid, so
                # q takes (0,0)-(7,0) and k takes eight cores from (8,0) -- off the end of row 0 and
                # on to row 1.  k is therefore TWO CoreRanges with a BOUNDING BOX of 22 cores, and
                # the profile duly tags both ops that inherit its shard spec at 22: the K rope, and
                # paged_fused_update_cache (whose core set is input1.grid MERGED with input2.grid,
                # then bounding-boxed, with a dummy kernel on every core in the box that owns
                # nothing).  Passing a `batch` x 2 output_mem_config grid instead gives q row 0 and
                # k row 1 -- one contiguous range each, and a 16-core box with nothing unused.
                # MEASURED 2026-09-07: paged_fused_update_cache 5.8926 -> 5.8756 ms of roofline gap,
                # K rope 2.0674 -> 2.0809, head split 3.3658 -> 3.3660, datamove bucket 40.018 ->
                # 39.974 -- all inside the run-to-run band.  So the core count the PROFILER reports
                # for these ops is the bounding box, not work, and a dummy kernel on an idle core is
                # free once the program is traced.  Do not spend another round on core-range
                # compaction; if this op is to get cheaper it has to be the 8 scattered partial-tile
                # cache writes each core makes, not the shape of the range.
                #
                # NON-OVERLAPPING q/k CORE GRIDS.  v always shares q's grid (the op hard-codes
                # v_shard_grid = q_shard_grid), so this is what makes k and v DISJOINT -- the
                # precondition paged_fused_update_cache needs to write both caches in one launch
                # instead of two per layer.  Costs nothing here: the split writes the same shards,
                # just placing k's on the next `batch` cores instead of on top of v's.
                overlap_qk_coregrid=False,
            )
        except (RuntimeError, TypeError, AttributeError):
            pass

    # THE FALLBACK MUST NOT INHERIT THE SHARD.  `flat` may be the projection's live L1 shard now, and
    # ttnn.slice does not cut a width-sharded tensor at an arbitrary column; interleave it first so
    # this path stays a genuine fallback rather than a second way to fail.
    if flat.is_sharded():
        flat = ttnn.to_memory_config(flat, ttnn.DRAM_MEMORY_CONFIG)

    q = ttnn.slice(flat, (0, 0, 0, 0), (1, 1, m, q_w))
    k = ttnn.slice(flat, (0, 0, 0, q_w), (1, 1, m, q_w + kv_w))
    v = ttnn.slice(flat, (0, 0, 0, q_w + kv_w), (1, 1, m, q_w + 2 * kv_w))
    return (
        ttnn.reshape(q, (1, batch, num_heads, head_dim)),
        ttnn.reshape(k, (1, batch, num_kv_heads, head_dim)),
        ttnn.reshape(v, (1, batch, num_kv_heads, head_dim)),
    )


# --------------------------------------------------------------- decode-shape RMSNorm
# INTERLEAVED NORM PARALLELISES OVER ROWS.  A decode activation is ONE tile row, so the whole
# reduction lands on ONE Tensix core (measured: 64 us/call on 1 core over 3072 elements).
# Width-sharding splits the EMBEDDING dim across cores instead -- the decode-side variant
# (GUIDELINES/02 section 2).  Prefill keeps the interleaved path: there the rows already
# parallelise and the sharded activation would not fit.
_NORM_PLANS = {}


def _norm_plan(device, h):
    key = (id(device), h)
    plan = _NORM_PLANS.get(key)
    if plan is None:
        g = device.compute_with_storage_grid_size()
        h_tiles = h // TILE
        best = None
        for y in range(1, g.y + 1):
            for x in range(1, g.x + 1):
                c = x * y
                if c > 64 or h_tiles % c:
                    continue
                if best is None or c > best[0]:
                    best = (c, x, y)
        if best is None:
            _NORM_PLANS[key] = False
            return False
        cores, gx, gy = best
        block_w = h_tiles // cores
        subblock_w = next(s for s in range(min(4, block_w), 0, -1) if block_w % s == 0)
        plan = (
            ttnn.create_sharded_memory_config(
                (TILE, h // cores),
                ttnn.CoreGrid(y=gy, x=gx),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
            ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[gx, gy],
                subblock_w=subblock_w,
                block_h=1,
                block_w=block_w,
                inplace=False,
            ),
        )
        _NORM_PLANS[key] = plan
    return plan


def residual_add(device, residual, delta):
    """`residual + delta`, written straight into the layout the RMSNorm that consumes it wants.

    The add already reads and writes the full [1, B, hidden] activation; naming the norm's own width
    shard as its output costs it nothing and deletes the InterleavedToSharded the norm would
    otherwise run on the result.  Falls back to a plain add whenever this hidden size has no shard
    plan (prefill heights, or a hidden size no rectangle divides).

    AND THIS LAUNCH CANNOT BE FUSED AWAY, WHICH IS WORTH SAYING BECAUSE THE CATALOG KEEPS OFFERING
    IT.  `ttnn.rms_norm` really does take `residual_input_tensor` (GUIDELINES/02 section 6 and 06
    section 1 both name add+Norm as the most reusable fusion, and the kwarg is there on this build),
    so `norm(x + delta)` looks like one op instead of two -- worth ~2 us a layer at decode and ~78 us
    a layer at prefill.  It does not apply to a PRE-norm block: the fused op returns ONE tensor
    (layernorm_device_operation.cpp's compute_output_specs returns a single TensorSpec, not a
    vector), so the SUM is consumed internally and thrown away -- and in a pre-norm residual stream
    that sum is precisely the next residual.  Recomputing it costs the add back.  The catalog's
    lever is written for a POST-norm block, where `LN(x + r)` has no other consumer.
    What DOES transfer is the placement half, and this function already takes it: the add writes the
    norm's own shard, so the pair costs one launch plus a borrow rather than one launch plus a
    reshard.
    """
    # ONLY AT THE DECODE HEIGHT.  _norm_plan keys on the hidden size alone, but the norm itself only
    # takes the sharded path when the activation is ONE tile row; a prefill [1, 512, hidden] forced
    # into a (TILE, hidden/cores) shard is simply wrong, so gate on the row count here too.
    dims = [int(x) for x in residual.shape]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    plan = _norm_plan(device, dims[-1]) if rows <= TILE else False
    if not plan:
        # THE RESIDUAL STREAM IS THE LAST bf16 TENSOR IN PREFILL, AND IT IS THE WIDEST.  Every delta
        # added into it already arrives as bf8_b (the projections are asked for that output) and
        # every consumer is a bf8_b-weighted matmul or the RMSNorm feeding one -- but ttnn.add takes
        # the WIDER of its two inputs, so the sum came back bf16 and the whole layer paid for it
        # twice over: the add wrote 25 MB instead of 12.6, and `ttnn.rms_norm` has no output-dtype
        # argument (its output dtype MATCHES its input), so the norm then read AND wrote 25 MB and
        # handed gate/up/qkv a bf16 activation.  That last part is not just bytes: on identical
        # shapes the profile shows `BFP8 x BFP8` down_proj at 414 TFLOP/s against `BF16 x BFP8`
        # gate_proj at 284, because a bf16 in0 against a block-float in1 costs the unpacker extra
        # passes at LoFi.  Narrowing the accumulator here therefore moves three ops at once.
        #
        # bf8_b IS THE FLOOR (GUIDELINES/01 section 13 names normalization activations as a tensor
        # that must never go below it), and the increments are already quantised at exactly this
        # granularity, so this rounds the running sum rather than introducing a new format.
        #
        # DO NOT PASS memory_config=stream_config(...) HERE.  It looks like the same placement lever
        # that moved the audio tower's norms and adds, but at the prefill height stream_config's size
        # gate returns DRAM, and NAMING DRAM is not the same as leaving the output alone: ttnn.add
        # otherwise inherits its input's config, so the explicit argument forces a round trip the
        # default never made.  Measured 2026-09-05 with the same change on rms_norm below: prefill
        # 142.87 -> 144.48 ms (+1.13%).  Leave the output placement implicit.
        #
        # AND NAMING L1 IS A DIFFERENT ARGUMENT THAT ALSO LOSES -- BUT ONLY AT DEPTH, WHICH IS THE
        # PART WORTH RECORDING.  Naming DRAM forces a round trip; L1 is the other direction and it
        # PROPAGATES for free, because ttnn.add inherits operand A's placement and A is the running
        # residual, so one named output carries the whole chain and no other call site changes.  On
        # the rates it looks like the best lever left in prefill: these two adds profile at 401 and
        # 422 GB/s reading DRAM interleaved, against ~757 GB/s for the ops that already have an L1
        # hand-off (nlp_create_heads, concatenate_heads) and 886 for the SwiGLU multiply on two L1
        # operands.  And on the 3-LAYER CAPTURE it delivers exactly that: the adds go 99.6 / 81.2
        # -> 47.6 us each and the LayerNorm that reads the sum goes 99.1 -> 62.9, i.e. 158 us a
        # LAYER, with PCC bit-identical at 0.9576212 because nothing but placement changed.
        #
        # It reaches none of that at the 30-layer trace stage, and the reason is CONTENTION, which
        # takes the whole 2x2 to see.  Measured 2026-09-08, prefill stage ms, PCC bit-identical in
        # every cell:
        #                          residual DRAM   residual L1
        #     SwiGLU L1 (HEAD)         106.49         106.94
        #     SwiGLU DRAM              113.85         111.13
        # Read down the columns: this chain is worth 2.72 ms on its own (113.85 -> 111.13) and
        # -0.45 ms once the SwiGLU hand-off holds the same L1 (106.49 -> 106.94).  Read across the
        # rows: that hand-off is worth 7.36 ms alone and 4.19 ms beside the chain.  The pair is
        # strictly SUB-ADDITIVE -- 2.72 + 7.36 = 10.1 ms of separate wins, 7.36 ms available
        # together -- because the residual pair is 2 x 10.9 MB = 197 kB/core and has to stay
        # resident ACROSS the SwiGLU's own 527 kB/core of gate/up, beside the down projection's
        # ~700 kB of circular buffers.  The SwiGLU wins outright, so this stays implicit; do not
        # re-derive the chain in isolation and conclude it is free.
        #
        # AND THE THREE-CELL VERSION OF THIS TABLE READ THE OPPOSITE WAY.  With only the top row
        # and 111.13, the natural conclusion was that the chain fails at depth on its own merits --
        # it takes the fourth cell to see that 111.13 is 2.72 ms BETTER than its true baseline, not
        # a loss.  An A/B on two interacting levers needs all four cells; three of them will
        # confidently support the wrong lever.
        #
        # SCOPING IT TO THE LINK THAT DOES NOT OVERLAP THE FFN WAS THE OBVIOUS ESCAPE, AND IT IS
        # ALSO MEASURED.  The chain has two links a layer and only one overlaps the SwiGLU: the
        # attention sum is the residual the MLP add reads, so it is resident through gate/up, while
        # the MLP sum is only resident across the NEXT layer's attention block.  Tried 2026-09-08
        # with a `place=` argument -- "l1" at the three MLP call sites, "dram" at the three
        # attention ones (an eviction, named only when the incoming residual really is in L1, so a
        # refused hand-off stays implicit).  Per-op it did exactly what it was designed to do:
        #     attention add  99.6 -> 72.67 us   (its A operand now reads L1)
        #     MLP add        81.2 -> 56.13 us
        #     LayerNorm on the MLP sum  99.1 -> 62.7 us
        # 88.4 us a layer, PCC bit-identical, which predicts 2.65 ms of prefill.  Delivered 0.10 ms:
        # 106.47 -> 106.37, two samples each, within-config spread 0.02.  Reverted.
        #
        # SO THE CONCLUSION IS NOT ABOUT THE SwiGLU, IT IS ABOUT THE BUDGET: PREFILL L1 IS
        # SATURATED.  Twice now, on two different links, an incremental tenant has produced a large
        # per-op win and ~nothing at depth -- because every hand-off it displaces is worth about
        # what it gains.  On the attention side those are qkv_out_config, the head split and
        # attn_out_config; on the FFN side, gate/up.  Do not add another L1 tenant to prefill
        # expecting the per-op delta to survive; the only way to win here is to make an EXISTING
        # hand-off cheaper, not to add one.
        #
        # GENERALISE THE OTHER HALF TOO: TT_PERF_LAYERS caps the profile at 3 layers, so the 158
        # us/layer above is measured against a per-core budget the 30-layer forward does not have.
        # Judge a placement lever on the STAGE time, never on the per-op delta the capture shows.
        return ttnn.add(residual, delta, dtype=_ACT_DTYPE if rows >= _GRID_REQUEST_MIN_ROWS else None)
    # THE ACCUMULATOR'S WIDTH IS THIS MODEL'S PCC EXCHANGE RATE, AND IT IS MEASURED BOTH WAYS.
    # The four projections that feed this add were narrowed to _ACT_DTYPE at the decode height and
    # the accumulator followed them; narrowing the SUM is the part that costs accuracy, because the
    # FFN intermediates at bf8_b are what prefill already runs while the residual sum is a new
    # rounding, applied once per block per layer.  Measured 2026-09-07 with everything else held:
    #   dtype=_ACT_DTYPE here   decode 10.0775 ms/token, e2e PCC 0.9576212
    #   dtype left implicit     decode 10.1086 ms/token, e2e PCC 0.9610618
    # i.e. 0.31% of decode against 0.0034 of PCC, in EITHER direction -- and the bf16 form is
    # actually ABOVE the 0.9600964 this model held before the projections were narrowed at all.
    # Keep the narrow form while the gate has room; this is the cheapest place in the model to buy
    # PCC BACK if a later lever needs it, and the only one whose price is already known.
    try:
        return ttnn.add(residual, delta, memory_config=plan[0], dtype=_ACT_DTYPE)
    except (RuntimeError, TypeError, AttributeError):
        return ttnn.add(residual, delta)


def rms_norm(device, x, weight, epsilon, compute_kernel_config, keep_sharded=False):
    """RMSNorm that width-shards the embedding dim when the activation is one tile row.

    `keep_sharded` returns the L1 shard the norm just produced instead of pushing it to DRAM.  It is
    worth asking for whenever the consumer is a decode projection: _norm_plan and in0_grid both pick
    the widest core rectangle dividing the SAME hidden size in tiles, so they land on the identical
    (TILE, hidden/cores) width shard -- meaning the DRAM round-trip was writing a layout out only for
    the next op to build it again.  The caller keeps ownership; nothing here deallocates it, so the
    one shard can feed both halves of a SwiGLU.
    """
    dims = [int(x.shape[i]) for i in range(len(x.shape))]
    m = 1
    for d in dims[:-1]:
        m *= d
    h = dims[-1]
    plan = _norm_plan(device, h) if m <= TILE else False
    if not plan:
        # Same non-lever as the add above: naming stream_config(x) here resolves to DRAM at the
        # prefill height and costs a round trip the implicit output placement never made.
        #
        # AND DO NOT BOTHER FOLDING THE LEADING BATCH HERE.  The matmuls next door gained 4% from
        # exactly that fold (see _block_linear), and the argument transfers on paper: the norm is
        # over the LAST dim and ttnn splits its work over ROWS, so handing it [8, 448, 3072] looks
        # like offering 448 rows where [1, 1, 3584, 3072] -- the identical buffer, since 448 is
        # tile-aligned and both reshapes merge leading dims only -- offers 3584.  MEASURED
        # 2026-09-05: prefill 136.49 -> 136.44 ms, i.e. nothing.  ttnn's layernorm already flattens
        # the leading dims internally, so unlike the matmul factory it never saw a batch to walk.
        # The fold is a matmul lever specifically, not a general one.
        return ttnn.rms_norm(x, weight=weight, epsilon=epsilon, compute_kernel_config=compute_kernel_config)
    shard_cfg, program_config = plan
    # BORROW AN INPUT ALREADY IN THIS LAYOUT.  The residual add that produces it can be asked to
    # write the norm's own shard (see residual_add), which removes this InterleavedToSharded
    # entirely; comparing by value is safe because MemoryConfig compares by value in ttnn.
    borrowed_in = x.memory_config() == shard_cfg
    xs = x if borrowed_in else ttnn.to_memory_config(ttnn.reshape(x, (1, 1, m, h)), shard_cfg)
    ys = ttnn.rms_norm(
        xs,
        weight=weight,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
        memory_config=shard_cfg,
    )
    if not borrowed_in:
        ttnn.deallocate(xs)
    if keep_sharded:
        # KEEP THE CALLER'S RANK.  The norm works in (1, 1, m, h) but its caller holds a residual at
        # the original rank and adds the block's output to it, so handing back the rank-4 view makes
        # every downstream shape rank-4 and the residual add stops matching.  The projection's
        # borrowed-activation check compares the memory config and the flattened dims, not the rank,
        # so restoring it here costs nothing on that path.
        return ys if len(dims) == 4 else ttnn.reshape(ys, tuple(dims))
    out = ttnn.to_memory_config(ys, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ys)
    return ttnn.reshape(out, tuple(dims))
