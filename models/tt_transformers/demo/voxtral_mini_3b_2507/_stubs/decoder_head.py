# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for decoder_head (lm_head).

Maps to: lm_head on VoxtralForConditionalGeneration
Simple linear projection: hidden_size -> vocab_size, no bias.
"""
from __future__ import annotations

import math

import ttnn


def _dram_sharded():
    """Load the shared DRAM-bank-sharded projection helper that sits next to this stub.

    The stubs are imported standalone BY PATH (tt/pipeline._load_stub_module), so they have no
    package context and a relative import is not available to them.
    """
    import importlib.util
    import pathlib
    import sys

    key = "_voxtral_stub__dram_sharded"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, pathlib.Path(__file__).with_name("_dram_sharded.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


_DS = _dram_sharded()

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)

# MATCHED TO bf8_b WEIGHTS.  Running 8-bit operands through a HiFi4 kernel makes the math engine
# take 4 passes over data that only has one pass worth of precision, which is what cancelled the
# bandwidth saving when the width was dropped on its own.  LoFi is the pairing for bf8_b; the
# matmul preference for fp32_dest_acc_en is False (it also unlocks wider subblocks).
_LOFI_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

_TILE = 32
# Output tiles one DRAM-bank worker may own before its circular buffers stop fitting L1.
# THIS NUMBER ALSO PICKS THE SPLIT COUNT, so it looks like a perf knob: at 64 the only feasible split
# of a 4096-tile vocab is FOUR, and 128 admits TWO -- halving both the matmul launches and the concat
# width per token.  MEASURED 2026-09-05: decode 10.9826 -> 10.9827 ms/token, i.e. exactly nothing.
# The projection is already at 91% of its DRAM roofline and the split count costs launches, not
# bytes, so trace has already absorbed them.  Left at 64, the value with the smaller L1 footprint.
_MAX_TILES_PER_WORKER = 64
# in0 is multicast to every compute core, so the activation shard has to stay small.
_MAX_COMPUTE_CORES = 32


class TtLMHead:
    """Vocab projection as ONE batch-folded, DRAM-SHARDED matmul over column chunks.

    THE BATCH DIMENSION WAS THE COST.  Both call sites hand this head `[B, 1, hidden]` -- the
    stream index is the LEADING dim and the tile-height dim holds a single position.  `ttnn.linear`
    reads a leading dim as BATCH, so it ran B independent `[1, H] x [H, V]` matmuls and re-streamed
    the whole 3072x131072 weight ONCE PER STREAM.  At B=8 that is eight full passes over ~428 MB of
    bf8_b weights for eight rows of output.  Folding the streams into M first makes it a single
    `[B, H] x [H, V]` matmul that reads the weight ONCE; the math is identical because every stream
    multiplies by the same weight.

    THE SECOND COST IS HOW THE WEIGHT IS LAID OUT.  DRAM-INTERLEAVED, every core pulls tiles from
    every bank over the NoC and a one-tile-tall matmul never reaches the DRAM roofline.  Width-
    sharding the weight ACROSS THE DRAM BANKS and using the DRAM-sharded program config pins each
    worker to the bank slice it consumes -- the decode-regime layout this matmul variant exists for.

    THE VOCAB IS SPLIT so the per-worker circular buffers stay inside the 1.5 MB L1 budget: at the
    full 131072 width each worker owns hundreds of output tiles and the in1 + intermediate buffers
    do not fit.  Splits are powers of two chosen so every chunk divides evenly across both the
    compute grid and the bank workers -- the DRAM-sharded matmul has NO padding support, so a ragged
    chunk is invalid rather than merely slow.  Chunking partitions output columns only, so the
    contract to argmax downstream is unchanged.
    """

    def __init__(self, device, torch_module):
        self.device = device
        weight = torch_module.weight.T.contiguous().float()
        self.k, self.n = int(weight.shape[0]), int(weight.shape[1])

        grid = device.compute_with_storage_grid_size()
        dram_grid = device.dram_grid_size()
        self.dram_cores = dram_grid.x
        gx = min(grid.x, 8)
        k_tiles = self.k // _TILE
        rows = [r for r in range(1, min(grid.y, _MAX_COMPUTE_CORES // gx) + 1) if k_tiles % (r * gx) == 0]
        self.num_cores = (max(rows) if rows else 1) * gx
        self.core_grid = ttnn.CoreGrid(y=self.num_cores // gx, x=gx)

        self.workers_per_bank, splits = self._pick_split()
        self.split_size = self.n // splits
        # THE ACTIVATION SHARD IS A SEPARATE AXIS FROM THE OUTPUT SHARD.  `num_cores` above has to
        # divide the vocab chunk as well as K, which a 2^17 vocab pins to 32; in0 only has to satisfy
        # `Kt % in0_block_w == 0`, so it takes the widest rectangle dividing k_tiles instead (48 here,
        # one mcast block of 2 tiles per core rather than 32 blocks of 3).  Same lever, same shared
        # planner, as the LM projections in _dram_sharded.py: measured there at 69% -> 86% of DRAM
        # peak on gate/up and 75% -> 88% on down.
        in0_plan = _DS.in0_grid(device, k_tiles, k_tiles * (self.split_size // _TILE))
        if in0_plan is None:
            in0_plan = (self.num_cores, gx, self.num_cores // gx)
        self.in0_cores, in0_gx, in0_gy = in0_plan
        self.in0_grid = ttnn.CoreGrid(y=in0_gy, x=in0_gx)
        self.in0_block_w = self.k // (_TILE * self.in0_cores)
        self._configs = {}

        dram_range = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.dram_cores - 1, dram_grid.y - 1))}
        )
        padded_split = math.ceil(self.split_size / (_TILE * self.dram_cores)) * (_TILE * self.dram_cores)
        weight_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_range, (self.k, padded_split // self.dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.weights = [
            ttnn.from_torch(
                weight[:, i * self.split_size : (i + 1) * self.split_size].contiguous(),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=weight_mem_cfg,
            )
            for i in range(splits)
        ]

    def _pick_split(self):
        """Fewest power-of-two chunks that fit both the compute grid and the bank workers."""
        n_tiles = self.n // _TILE
        candidates = [1 << i for i in range(int(math.log2(n_tiles)) + 1)]
        for wpb in (2, 1):
            workers = self.dram_cores * wpb
            for s in candidates:
                split, split_tiles = self.n // s, n_tiles // s
                if split * s != self.n or split % (_TILE * self.num_cores):
                    continue
                if split_tiles % workers or split_tiles // workers > _MAX_TILES_PER_WORKER:
                    continue
                return wpb, s
        # No exact fit against the bank count (a board whose bank count shares no factor with the
        # vocab): keep the compute-grid split exact and let the weight shard pad across the banks.
        for s in candidates:
            split, split_tiles = self.n // s, n_tiles // s
            if split * s != self.n or split % (_TILE * self.num_cores):
                continue
            if math.ceil(split_tiles / self.dram_cores) <= _MAX_TILES_PER_WORKER:
                return 1, s
        raise RuntimeError(f"no usable vocab split for k={self.k} n={self.n} cores={self.num_cores}")

    def _config_for(self, m_tiles):
        cfg = self._configs.get(m_tiles)
        if cfg is None:
            cfg = (
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=self.in0_block_w,
                    per_core_M=m_tiles,
                    per_core_N=self.split_size // (_TILE * self.num_cores),
                    fused_activation=None,
                    num_workers_per_dram_bank=self.workers_per_bank,
                ),
                ttnn.create_sharded_memory_config(
                    (m_tiles * _TILE, self.k // self.in0_cores),
                    self.in0_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
                # NAME THE OUTPUT SHARD.  `L1_WIDTH_SHARDED_MEMORY_CONFIG` carries no shard spec, so
                # ttnn spreads the output over the WHOLE compute grid while the program config sizes
                # per_core_N for THIS grid -- the two disagree and the circular buffers get sized for
                # far more cores than the matmul has work for, overflowing L1 by a hair.
                ttnn.create_sharded_memory_config(
                    (m_tiles * _TILE, self.split_size // self.num_cores),
                    self.core_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
            self._configs[m_tiles] = cfg
        return cfg

    def __call__(self, x, **kwargs):
        dims = [int(x.shape[i]) for i in range(len(x.shape))]
        m = 1
        for d in dims[:-1]:
            m *= d
        program_config, act_mem_cfg, out_mem_cfg = self._config_for(math.ceil(m / _TILE))
        flat = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, m, self.k)), act_mem_cfg)
        # CONCAT THE SHARDS DIRECTLY.  Re-interleaving each split before the concat is a second full
        # pass over the logits for no reason -- concat reads the L1 shards and writes DRAM in one op.
        # NARROW THE LOGITS THEMSELVES, NOT JUST THE WEIGHT.  The chunks are 32 x 32768 each, so at
        # bf16 the four of them are 8.4 MB that get written by the matmul, read and rewritten by the
        # concat, and read again by the untilize -- three full passes over a tensor whose only
        # consumer is an argmax.  bf8_b halves every one of those, and costs the scan nothing: the
        # untilize converts a BFLOAT8_B input back to BFLOAT16 on output (it is documented to), so
        # the sampler still sees exactly the bf16 row-major block its resident buffer expects.
        # This is the one tensor in the model where block-float rounding is checked directly by the
        # gate rather than indirectly -- the sampled token is an argmax over these very values -- so
        # it lives or dies on the e2e PCC number.
        parts = [
            ttnn.linear(
                flat,
                w,
                program_config=program_config,
                memory_config=out_mem_cfg,
                compute_kernel_config=_LOFI_CFG,
                dtype=ttnn.bfloat8_b,
            )
            for w in self.weights
        ]
        ttnn.deallocate(flat)
        # THE JOINED LOGITS MUST STAY IN DRAM.  Landing them in interleaved L1 instead looks free --
        # 2.1 MB against 110 x 1.5 MB, consumed immediately by the sampler's untilize, which already
        # asks for L1 itself -- but it BREAKS BOTH TRACED STAGES: prefill and decode stopped
        # reporting entirely (only encode survived the measurement) because the trace's own L1 is
        # allocated around these buffers.  Measured 2026-09-05; do not retry without a trace-region
        # budget to match.
        rows = 1
        for d in dims[:-1]:
            rows *= d
        # UNTILIZE PER CHUNK, THEN JOIN -- see _join_unpadded.  A build whose untilize refuses an L1
        # width shard falls back to the original tiled concat below, so the sampler's contract (a
        # bf16 ROW_MAJOR [.., rows, V] block) never depends on this path holding.
        try:
            return ttnn.reshape(self._join_unpadded(parts, rows), tuple(dims[:-1]) + (self.n,))
        except (RuntimeError, TypeError, ValueError):
            pass
        out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if out is not parts[0]:
            # RELEASE THE CHUNKS ONCE THEY ARE JOINED.  Each is an L1 width shard (~1.1 MB over the
            # compute grid) and Python keeps the list alive to the end of this frame, which used not
            # to matter because the frame ended one reshape later; the untilize below is another op
            # inside the same frame and the sampler's programs have to place circular buffers around
            # whatever is still allocated.
            for p in parts:
                ttnn.deallocate(p)
        # UNTILIZE BEFORE UNFOLDING THE STREAMS, NOT AFTER.  `out` is [1, 1, B, V] in TILE layout:
        # ONE tile row carrying B=8 real rows in a 32-row pad.  Reshaping THAT to [B, 1, V] while it
        # is still tiled does not rearrange 8 rows, it builds EIGHT tile rows -- the last two dims of
        # each stream become (1, V), which tile layout pads straight back up to (32, V).  At
        # V = 131072 that turns an 8.4 MB tensor into a 67 MB one, and the sampler's untilize then
        # has to read all 67 MB to recover the same 2 MB of logits.
        #
        # Unpadding first collapses both: one untilize reads the single 8.4 MB tile row and writes
        # the 2 MB of real values, and the [1, 1, B, V] -> [B, 1, V] reshape that follows is a
        # metadata view, because a ROW_MAJOR tensor pages by its last dim and both shapes are the
        # same B pages of V.
        #
        # IT STAYS IN DRAM, and that is not the timid choice -- it is the one that costs nothing.
        # The sampler's scan needs these values in L1, but it already copies them into a RESIDENT L1
        # buffer of its own (cpp_argmax builds its program descriptors against fixed addresses, so it
        # cannot read a freshly-allocated tensor), and that copy takes a DRAM source just as happily
        # as an L1 one.  Asking for L1 here instead only adds a second 2 MB L1 tenant next to that
        # buffer, and measured 2026-09-05 it made the sampler's copy program unplaceable outright:
        # "statically allocated circular buffers in program 200 clash with L1 buffers on core range
        # [0-0 - 0-7]".  A ROW_MAJOR [B, 1, V] tensor is B pages of 256 kB, so both tenants land on
        # the same eight banks whatever the grid size suggests.
        rows = 1
        for d in dims[:-1]:
            rows *= d
        try:
            rm = ttnn.untilize_with_unpadding(out, [0, 0, rows - 1, self.n - 1])
        except (RuntimeError, TypeError, ValueError):
            return ttnn.reshape(out, tuple(dims[:-1]) + (self.n,))
        return ttnn.reshape(rm, tuple(dims[:-1]) + (self.n,))

    def _join_unpadded(self, parts, rows):
        """Untilize each vocab chunk FIRST, then join the row-major pieces.

        UNPADDING IS A SHRINK, SO IT BELONGS UPSTREAM OF THE JOIN.  Each chunk is one TILE row
        carrying `rows` real values in a 32-row pad, so the tiled form is four times the bytes of
        the values in it.  Concatenating tiled and untilizing after moves that padding twice --
        the concat reads and rewrites all four padded chunks, and the untilize then reads the
        joined padded block to recover the same values.  Untilizing per chunk pays the padding
        exactly once, on the read the matmul's output has to be read for anyway, and the join
        that follows moves only the real row-major values.
        """
        # ...AND ASK THE UNTILIZE TO LAND THE PIECE WHERE THE JOIN READS IT.  Left to itself the op
        # inherits its INPUT's placement, and the input is the matmul's L1 WIDTH shard -- so the
        # unpadded piece came back sharded too and every one of the four then needed a separate
        # ShardedToInterleaved before the concat could touch it.  Those four launches profiled at
        # 2.45 us each on 32 cores and did nothing but move 512 kB onto itself in a different
        # layout; untilize_with_unpadding takes the memory_config directly, so naming it here folds
        # all four into the untilize the chunk was being read for anyway.
        #
        # DRAM RATHER THAN L1, AND THAT IS MEASURED.  L1 looked like the obvious destination -- the
        # four pieces are ~2 MB together, they die inside this function, and the join is the op that
        # cannot spread (a row-major tensor pages by its LAST dim, so [.., rows, split] is `rows`
        # pages and the concat runs on that many cores whatever the grid size), which ought to make
        # its read the one worth putting in Tensix banks.  It is not: measured 2026-09-06 the concat
        # went 24.4 -> 29.2 us reading L1, because 8 cores pulling 8 pages scattered over a 32-core
        # shard is more NoC hops than the same 8 cores streaming them out of the DRAM controller.
        # DRAM keeps the concat where it was and still folds the four ShardedToInterleaved away.
        try:
            pieces = [
                ttnn.untilize_with_unpadding(
                    p, [0, 0, rows - 1, self.split_size - 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                for p in parts
            ]
        except (RuntimeError, TypeError, ValueError):
            pieces = [ttnn.untilize_with_unpadding(p, [0, 0, rows - 1, self.split_size - 1]) for p in parts]
        joined = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # NOTHING IS RELEASED UNTIL THE WHOLE CHAIN HAS SUCCEEDED.  The caller keeps the tiled chunks
        # as its fallback, so freeing them before the concat returns would leave that path holding
        # deallocated buffers on exactly the builds the fallback exists for.
        for p in parts:
            ttnn.deallocate(p)
        if joined is not pieces[0]:
            for p in pieces:
                ttnn.deallocate(p)
        return joined


def build(device, torch_module=None):
    return TtLMHead(device, torch_module)
