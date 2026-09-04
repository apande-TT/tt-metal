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
# Weight tiles one in0 mcast block must still stream for widening the activation shard to pay for
# the extra block's synchronisation -- see in0_grid for the four measurements that set it.
_MIN_WEIGHT_TILES_PER_IN0_BLOCK = 384
# Rows below which `mm` does NOT name the core grid (see the note at its call to ttnn.linear).
_GRID_REQUEST_MIN_ROWS = 32 * TILE


def _core_count(device, k_tiles, n_tiles):
    """Largest core count (as a grid that fits the device) dividing BOTH k_tiles and n_tiles.

    The DRAM-sharded matmul gives every core an exact slice of K and of N and has NO padding
    support, so a count that does not divide both is invalid rather than merely slow.
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


def in0_grid(device, k_tiles, weight_tiles):
    """Widest in0 (activation) shard whose per-core slice still amortises its mcast block.

    The activation is width-sharded one mcast block per core, so the core count IS the number of
    K blocks the matmul iterates.  Widening it narrows every in0/in1 circular buffer and measurably
    raises the weight stream's DRAM utilisation -- but each block also costs a fixed multicast and
    synchronisation, and that only pays if the block still streams a substantial slice of in1.
    MEASURED on this model, per call: gate/up (24576 weight tiles) 32 -> 48 cores took 71.0 -> 58.3
    us and 69% -> 84% of DRAM peak; down (24576) 32 -> 64 cores took 65.9 -> 56.6 us and 75% -> 87%.
    o_proj is the counter-example that sets the floor: it has HALF the weight (12288 tiles), so at 64
    cores each block carries only 192 tiles and the block overhead stops being hidden -- 34.7 -> 51.3
    us and 71% -> 48%.  So the bound is weight tiles PER BLOCK, not a core count.
    """
    g = device.compute_with_storage_grid_size()
    best = None
    for y in range(1, g.y + 1):
        for x in range(1, g.x + 1):
            c = x * y
            if c > _MAX_COMPUTE_CORES or k_tiles % c:
                continue
            if weight_tiles // c < _MIN_WEIGHT_TILES_PER_IN0_BLOCK:
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

    def __init__(self, device, weight, max_m_tiles=1):
        self.device = device
        self.ok = False
        self.max_m_tiles = max_m_tiles
        try:
            self._build(device, weight, max_m_tiles)
        except Exception:  # noqa: BLE001 - any shape this cannot serve exactly falls back
            self.ok = False

    def _build(self, device, weight, max_m_tiles):
        shape = tuple(weight.shape)
        self.k, self.n = int(shape[-2]), int(shape[-1])
        k_tiles, n_tiles = self.k // TILE, self.n // TILE
        if self.k % TILE or self.n % TILE:
            return

        dram_grid = device.dram_grid_size()
        dram_cores = dram_grid.x
        plan = self._plan(device, k_tiles, n_tiles, dram_cores, max_m_tiles)
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
        in0_plan = in0_grid(device, k_tiles, k_tiles * n_tiles) or (cores, gx, gy)
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

    def _plan(self, device, k_tiles, n_tiles, dram_cores, max_m_tiles):
        """Fewest power-of-two chunks that divide the compute grid AND the bank workers exactly."""
        for splits in (1 << i for i in range(int(math.log2(n_tiles)) + 1)):
            chunk_tiles = n_tiles // splits
            if chunk_tiles * splits != n_tiles:
                continue
            picked = _core_count(device, k_tiles, chunk_tiles)
            if picked is None:
                continue
            for wpb in (2, 1):
                workers = dram_cores * wpb
                if chunk_tiles % workers:
                    continue
                if max_m_tiles * (chunk_tiles // workers) > _MAX_OUT_TILES_PER_WORKER:
                    continue
                return splits, wpb, picked[0], picked[1], picked[2]
        return None

    def serves(self, m_tiles):
        return self.ok and 0 < m_tiles <= self.max_m_tiles

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

    def __call__(self, x, compute_kernel_config=None, keep_sharded=False):
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


def attach(device, weight, max_m_tiles=1):
    """Build a mirror for `weight`, or return None when the shape cannot be served exactly."""
    mirror = DramShardedLinear(device, weight, max_m_tiles=max_m_tiles)
    return mirror if mirror.ok else None


def linear(x, weight, mirror, compute_kernel_config=None, core_grid=None, activation=None, keep_sharded=False):
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
            return _apply(activation, mirror(x, compute_kernel_config=compute_kernel_config, keep_sharded=keep_sharded))
    # The PLAIN path is the opposite: prefill hands ttnn.linear hundreds of tile rows, so fusing
    # removes a full-width read-modify-write of the [rows, intermediate] tensor.  Same measurement:
    # prefill 210.82 -> 196.35 ms (-6.9%).
    rows = 1
    for d in tuple(x.shape)[:-1]:
        rows *= d
    return ttnn.linear(
        x,
        weight,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        activation=_linear_activation(activation),
        # Same regime boundary `mm` uses: narrow the output only where the tensor is big enough for
        # the bytes to matter, so a decode shape that misses its mirror still lands in bf16.
        dtype=_ACT_DTYPE if rows >= _GRID_REQUEST_MIN_ROWS else None,
    )


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
    """
    rank = len([int(d) for d in x.shape])
    gate = linear(x, gate_w, gate_ds, compute_kernel_config, core_grid, activation="silu", keep_sharded=True)
    up = linear(x, up_w, up_ds, compute_kernel_config, core_grid, keep_sharded=True)
    if gate.is_sharded():
        # LEAVE THE SHARD, BUT NOT ALL THE WAY TO DRAM.  The multiply has to produce something
        # unsharded (down's in0 rectangle is wider than gate/up's output rectangle, so there is no
        # shard-to-shard conversion to inherit), and that is what lets it replace both
        # ShardedToInterleaved ops.  But "unsharded" does not have to mean DRAM: at the decode shape
        # this is 131 kB, and down_proj's very next act is to reshard it back into L1.
        n = 1
        for d in tuple(gate.shape):
            n *= int(d)
        h = ttnn.multiply(gate, up, memory_config=_handoff_config(n, gate.dtype))
        # FREE THE HALVES.  multiply builds a new DRAM tensor rather than viewing its inputs, so this
        # is safe, and it matters: the down projection sizes its circular buffers against whatever L1
        # is still free on these cores.
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        dims = [int(d) for d in h.shape]
        if len(dims) > rank:
            h = ttnn.reshape(h, tuple(dims[-rank:]))
    else:
        h = ttnn.multiply(gate, up)
    return linear(h, down_w, down_ds, compute_kernel_config, core_grid)


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
_COST_TIE = 0.01
# Bytes of circular buffer one core may hold for a 2-D mcast matmul.  Blackhole has 1,572,864 B of
# L1 per core; this is deliberately well under half of it because the estimate below counts only
# in0/in1/out/interm and the factory also reserves space for the reader/writer and the semaphores.
_CB_BUDGET_BYTES = 700 * 1024
# Tile-matmul-equivalents one K block costs in multicast plus semaphore synchronisation.  Used only
# to RANK candidates against each other, so its exactness matters far less than its sign: it is what
# stops the search from picking a one-tile K block, which is the shape of the config ttnn was
# choosing on its own.
_KBLOCK_OVERHEAD = 60.0
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


def block_config(device, m_tiles, k_tiles, n_tiles, tile_bytes=1088, interm_bytes=2048, activation=None):
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

    Returns None when nothing fits, so the caller keeps its own path.
    """
    key = (int(m_tiles), int(k_tiles), int(n_tiles), int(tile_bytes), int(interm_bytes), str(activation))
    if key in _BLOCK_CFG_CACHE:
        return _BLOCK_CFG_CACHE[key]
    g = device.compute_with_storage_grid_size()
    k_divs = _divisors(k_tiles)
    best = None
    for gy in range(1, int(g.y) + 1):
        per_core_m = -(-m_tiles // gy)
        if gy > 1 and per_core_m * (gy - 1) >= m_tiles:
            continue
        for gx in range(1, int(g.x) + 1):
            per_core_n = -(-n_tiles // gx)
            if gx > 1 and per_core_n * (gx - 1) >= n_tiles:
                continue
            for block_h in _divisors(per_core_m):
                for block_w in _divisors(per_core_n):
                    h, w = _subblock(block_h, block_w)
                    reuse = min(1.0, (h * w) / float(h + w))
                    blocks = (per_core_m // block_h) * (per_core_n // block_w)
                    for in0_block_w in k_divs:
                        if _cb_bytes(block_h, block_w, in0_block_w, tile_bytes, interm_bytes) > _CB_BUDGET_BYTES:
                            continue
                        cost = per_core_m * per_core_n * k_tiles / reuse + blocks * (
                            k_tiles / in0_block_w
                        ) * _KBLOCK_OVERHEAD
                        cand = (cost, gx, gy, per_core_m, per_core_n, block_h, block_w, in0_block_w, h, w)
                        if (
                            best is None
                            or cost < best[0] * (1.0 - _COST_TIE)
                            or (cost < best[0] * (1.0 + _COST_TIE) and gx * gy > best[1] * best[2])
                        ):
                            best = cand
                        break
    cfg = None
    if best is not None:
        _, gx, gy, per_core_m, per_core_n, block_h, block_w, in0_block_w, h, w = best
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=h,
            out_subblock_w=w,
            out_block_h=block_h,
            out_block_w=block_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=_fused_activation(activation),
        )
    _BLOCK_CFG_CACHE[key] = cfg
    return cfg


# Shapes whose 2-D config ttnn refused at runtime; they fall back for the rest of the run so a
# rejected shape costs one exception, not one per call.
_BLOCK_CFG_REFUSED = set()


def mm(device, x, weight, compute_kernel_config=None, bias=None, mirror=None, keep_sharded=False, activation=None):
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
            activation, mirror(x, compute_kernel_config=compute_kernel_config, keep_sharded=keep_sharded)
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
    # to the grid request below and is remembered so the exception is paid once, not per call.
    wshape = tuple(weight.shape)
    k_tiles, n_tiles = int(wshape[-2]) // TILE, int(wshape[-1]) // TILE
    m_tiles = math.ceil(m / TILE)
    batch = 1
    for d in tuple(x.shape)[:-2]:
        batch *= int(d)
    key = (m_tiles, k_tiles, n_tiles, batch, str(activation))
    # M IS THE PADDED HEIGHT, NOT THE LOGICAL ONE.  The audio tower's activation is 1500 rows, which
    # the matmul pads to 47 tiles; requiring m % TILE == 0 here silently excluded every encoder
    # projection -- the exact shapes this config exists for.
    # ONE ACTIVATION, NOT A STACK OF THEM.  With a leading batch the 2-D factory runs the whole
    # grid once PER BATCH ENTRY, so the block sizes below are chosen against a height the kernel
    # never sees in one pass; measured on this model's 8-stream prefill it cost 0.8% where the
    # single-stream audio tower gained 7%.  Batched shapes keep the grid request underneath.
    if (
        batch == 1
        and key not in _BLOCK_CFG_REFUSED
        and int(wshape[-2]) % TILE == 0
        and int(wshape[-1]) % TILE == 0
    ):
        cfg = block_config(device, m_tiles, k_tiles, n_tiles, activation=activation)
        if cfg is not None:
            try:
                return ttnn.linear(
                    x,
                    weight,
                    bias=bias,
                    compute_kernel_config=compute_kernel_config,
                    program_config=cfg,
                    dtype=_ACT_DTYPE,
                )
            except (RuntimeError, TypeError, ValueError):
                _BLOCK_CFG_REFUSED.add(key)
    return ttnn.linear(
        x,
        weight,
        bias=bias,
        compute_kernel_config=compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=g.y, x=g.x),
        activation=_linear_activation(activation),
        dtype=_ACT_DTYPE,
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
    mem = ttnn.L1_MEMORY_CONFIG if b * s * w * 2 <= _SDPA_L1_MAX_BYTES else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.experimental.nlp_create_qkv_heads(
        ttnn.reshape(qkv, (b, 1, s, w)),
        num_heads=num_heads,
        num_kv_heads=num_heads if num_kv_heads is None else num_kv_heads,
        transpose_k_heads=False,
        memory_config=mem,
    )


_SDPA_CFGS = {}
_SDPA_MAX_CHUNK = 256
# Fraction of the grid the flash work units must keep busy before sdpa_config stops halving the q
# chunk.  2/3 is the point where one more halving stops paying: it doubles the per-unit loop and CB
# overhead to recover less than half a round.
_SDPA_MIN_OCCUPANCY = 2.0 / 3.0


def sdpa_config(device, q, k):
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
    sequence keeps small chunks instead of padding itself up into wasted work.  exp_approx_mode
    is left exact: these sequences accumulate over many chunks, and the approximate exp costs
    PCC there without being faster on this arch.

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

    key = (grid.x, grid.y, _q_chunk(), _chunk(seq_k))
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


def merge_heads_decode(attn_out, batch, num_heads, head_dim, l1=True):
    """[1, B, padded_nh, hd] -> [1, B, nh*hd] for the o_proj, kept in L1.

    This reshape collapses the last TWO dims, so on TILE layout it genuinely re-tilizes rather than
    returning a view -- it is the single most expensive datamove left in the decode step (measured
    ~20 us/call, once per layer per token).  ttnn's dedicated replacement wants a height-sharded
    input that GQA will not let sdpa_decode produce, so the op itself cannot be swapped; what CAN
    change is that it no longer reads and writes DRAM for a 65 kB tensor.  Shared by every attention
    body so the placement reaches all of them rather than only the individually-routed layers.
    """
    shape = (1, int(batch), int(num_heads) * int(head_dim))
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
        return ttnn.add(residual, delta, dtype=_ACT_DTYPE if rows >= _GRID_REQUEST_MIN_ROWS else None)
    try:
        return ttnn.add(residual, delta, memory_config=plan[0])
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
