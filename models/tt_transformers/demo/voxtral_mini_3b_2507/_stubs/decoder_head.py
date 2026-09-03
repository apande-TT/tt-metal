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
        parts = [
            ttnn.linear(
                flat,
                w,
                program_config=program_config,
                memory_config=out_mem_cfg,
                compute_kernel_config=_LOFI_CFG,
            )
            for w in self.weights
        ]
        ttnn.deallocate(flat)
        out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, tuple(dims[:-1]) + (self.n,))


def build(device, torch_module=None):
    return TtLMHead(device, torch_module)
