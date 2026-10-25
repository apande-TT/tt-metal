# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_attention` (Ministral3Attention,
`language_model.layers.0.self_attn`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    q, k, v = x Wq, x Wk, x Wv                       (32 q heads, 8 kv heads, head_dim 128 -> GQA 4:1)
    q, k    = rope(q), rope(k)
    q       = q * (1 + beta * log(1 + floor(pos / orig_max)))      (llama-4 attention scaling)
    out     = sdpa(q, k, v, mask, scale=head_dim**-0.5) Wo

The canonical tt_transformers Attention is not used: it has no llama-4 query scaling and its
ModelArgs path does not load this multimodal checkpoint.

Two modes, one weight set:
  * prefill  x (B, 1, S, D): causal SDPA over the prompt. With `kv_cache=(k, v)` the prompt's K/V
             (B, kv_heads/TP, S, head_dim) are written into the resident cache rows 0..S-1.
  * decode   x (1, 1, B, D): one token per batch row. K/V are appended at `cur_pos` (int32 [B]) with
             paged_update_cache and attention reads the resident cache with SDPA-decode.

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * q/k/v  column-parallel by HEAD: device d holds q heads [d*Hq/TP, (d+1)*Hq/TP) and the kv heads
           those q heads attend with ([d*Hkv/TP, (d+1)*Hkv/TP)), so every GQA group stays local.
           Fused per device as [q | k | v] into one wqkv.
  * o_proj: decode row-parallel (input features = that device's q heads), then all_reduce; prefill
           column-parallel over the all_gathered heads (see self.wo).
  * rotary tables, the mask and the llama-4 scale are replicated; the KV cache holds the local heads.
"""

from __future__ import annotations

import math

import torch

import ttnn


def tp_ccl_args():
    """Topology / links for the text decoder's TP collectives: Ring over 2 links when the ring fabric is active
    (QB2, see mesh.py), else Linear on the default link count."""
    try:
        if ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING:
            return {"topology": ttnn.Topology.Ring, "num_links": 2}
    except Exception:  # noqa: BLE001
        pass
    return {"topology": ttnn.Topology.Linear}


def fold_input_scale(w, in_scale):
    """Weight w (in, out) with a per-input-feature scale folded in: (x * s) @ w == x @ (s[:, None] * w).

    The decoder folds each RMSNorm's gamma into the projections that read the norm, so the norm's output feeds the
    matmul directly instead of through an elementwise gamma multiply. The product is formed in fp32 and rounded to
    the weight dtype once."""
    if in_scale is None:
        return w
    return w.float() * in_scale.detach().float()[:, None]


def _mesh_size(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


# L1 cores the decode QKV activation is width-sharded over for its DRAM-sharded matmul: 16 divides both K (160 tiles)
# and the per-device [q | k | v] N (48 tiles).
QKV_DECODE_CORES = ttnn.CoreGrid(y=2, x=8)
# dtype of the decode-only QKV / o_proj weight copies.
DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b


class TtMinistral3Attention:
    # dtype of o_proj's output and its all_reduce: fp32, added unrounded into the decoder's fp32 residual stream.
    OUT_DTYPE = ttnn.float32
    # Prefill's many-row o_proj partials are produced and reduce-scattered in bf16 instead (half the reduce-scatter
    # bytes, no fp32 -> bf16 cast before the gather).
    PREFILL_REDUCE_DTYPE = ttnn.bfloat16
    # Decode's single-row o_proj partials likewise (None: fp32 partials, cast to bf16 after the reduce-scatter).
    DECODE_REDUCE_DTYPE = ttnn.bfloat16
    # The decode QKV takes the input norm's width-sharded fp32 output as it is (no unshard + cast, no reshard onto
    # QKV_DECODE_CORES), in K blocks of DECODE_SHARDED_K_BLOCK tiles (whole shards).
    DECODE_SHARDED_INPUT = True
    DECODE_SHARDED_K_BLOCK = 20
    DECODE_SHARDED_READERS_PER_BANK = 3  # (the op allows 1-3)
    # nlp_create_qkv_heads_decode reads the QKV's width shards in place (its sharded-input path) instead of an
    # unsharded copy of the row.
    DECODE_SHARDED_HEADS = True
    # Decode splits the heads with k on its own cores (overlap_qk_coregrid=False: q and v on the head grid's first B
    # cores, k on the next B), so q's and k's RoPE run as ONE rotary_embedding_llama_fused_qk and K / V are appended
    # by ONE paged_fused_update_cache (each needs its two inputs on disjoint cores).
    DECODE_FUSED_QK = True
    # That K / V append as tt/kv_append.py's program (cpp try 1, run 8): the new rows packed to bfloat8_b and written
    # straight into the cache tiles (exponent groups + mantissa face rows), not untilized / patched / re-tilized.
    DECODE_KV_APPEND = True
    # Decode o_proj as the MLP down's C++ streaming program (tt/down_decode.py: the concatenated heads multicast once,
    # a bank-major copy streamed one column per data-movement RISC, ~9 KB pushes, fp32 DEST over all K), its bf16
    # partials left in L1 for the reduce-scatter.
    DECODE_OPROJ_STREAM = True
    # K rows a push of that o_proj stream (its K is only 32 tiles: 8 pushes a column)
    DECODE_OPROJ_PUSH_ROWS = 4
    # Decode SDPA key chunk (0: sized per step by the kernel)
    DECODE_SDPA_K_CHUNK = 128
    # Rows of the decode SDPA grid: its 32 users x 2 kv heads (TP=4) take one core each, so 8 x 8 runs the same
    # kernel as 8 x 10 without launching 16 idle cores (trace decode ~8 us a token less).
    DECODE_SDPA_ROWS = 8
    # False: prefill returns o_proj's reduce-scattered 1/TP hidden shard (the decoder keeps a hidden-sharded
    # residual in prefill and gathers only the normed activation).
    prefill_gather = True
    # Widest SDPA key chunk prefill may use (see __call__).
    PREFILL_MAX_K_CHUNK = 512
    # Prefill SDPA query chunk beside a 512-row key chunk: every q chunk streams the whole key/value range from DRAM
    # again (64-row chunks re-read ~540 MB of K/V per layer and device), and 128 is the widest whose score block fits
    # L1 there. Beside a 257..448-row key chunk the queries take PREFILL_WIDE_Q_CHUNK rows (see __call__).
    PREFILL_Q_CHUNK = 128
    PREFILL_WIDE_Q_CHUNK = 160
    # Columns of the decode head grid (one batch row per core): 32 rows fill an 8 x 4 rectangle, the shape the fused
    # decode RoPE needs (it runs on its input shard grid's bounding box).
    HEAD_GRID_X = 8
    # Batch rows the resident decode rotation tiles cover (the pipeline's maximum batch).
    MAX_DECODE_BATCH = 32
    # Prefill RoPE runs on a (batch, heads) regrouping of the heads that balances its grid split (see _rope_regroup).
    REGROUP_PREFILL_ROPE = True
    # ... and reads cos / sin and the pair-swap tile from per-core L1 shards (prefill_rope_tables) when that moves
    # fewer DRAM bytes than its per-row reload of the interleaved tables.
    L1_PREFILL_ROPE = True
    # Prefill q / k / v are split out of the fused QKV into L1 (interleaved over the grid) when they take at most this
    # many bytes per core: the RoPEs read q / k from L1 and write DRAM, and v is copied to DRAM for the cache fill and
    # SDPA (none stays resident in L1 across the larger prefill ops, whose static buffers leave no room). 0 disables.
    PREFILL_HEADS_L1_BYTES = 512 * 1024
    # structural (run 8): the prefill QKV written in the head layouts by its own output writer (minimal_mm heads_out)
    PREFILL_QKV_HEADS_OUT = True
    # grid try (run 8): the prompt K / V cast to the cache dtype on the fill's slab grid (one (user, kv head) slab of S
    # rows a core, in L1) after the reshard, instead of on the interleaved split DRAM -> DRAM before it
    PREFILL_KV_CAST_ON_SLABS = True
    # ... and the fused QKV output feeding that split lands in L1 too while it takes at most this many bytes per core
    # (114 KB at 32 x 128 rows, 372 KB at 32 x 416 rows beside the matmul's ~0.9 MB of static buffers).
    PREFILL_QKV_L1_BYTES = 400 * 1024
    # The o_proj's hidden-shard output (TP prefill, read once by the decoder's residual add) lands in L1 while it takes
    # at most this many bytes per core (95 KB at 32 x 128 rows; the caller frees it right after the add).
    PREFILL_OPROJ_OUT_L1_BYTES = 128 * 1024
    # With TP, SDPA's output (read once by nlp_concat_heads, which writes L1 for the all_gather) lands in L1 too while
    # it takes at most this many bytes per core (76 KB at 32 x 128 rows; 248 KB at 416 rows does not fit beside SDPA's
    # static buffers).
    PREFILL_SDPA_OUT_L1_BYTES = 256 * 1024
    # The prefill RoPEs write q / k into L1 (interleaved) instead of DRAM while q takes at most this many bytes per
    # core (shard try, run 8: the 32 x 128-row prompt's ~76 KB); SDPA and the cache fill read them there, and both
    # are freed right after SDPA.
    PREFILL_ROPE_OUT_L1_BYTES = 96 * 1024
    # ... and the L1 heads are concatenated by one C++ program (tt/concat_heads.py: every core pushes the tiles of its
    # own L1 bank to their slots) instead of nlp_concat_heads.
    CPP_PREFILL_CONCAT = True
    # Decode's SDPA output is concatenated by one C++ program too (tt/concat_heads.py ConcatHeadsDecode: each core
    # gathers one output tile's rows straight out of the L1-interleaved SDPA tiles) instead of a reshard onto the head
    # shards, nlp_concat_heads_decode and an unshard.
    CPP_DECODE_CONCAT = True
    # Set by the decoder layer: the width-sharded layout its decode residual add runs in. The decode all_gather of
    # the reduced o_proj shard writes straight into it, so the add reads two shards (no interleaved row).
    decode_gather_mem = None
    # Set by the decoder layer: decode's o_proj all_reduce as one collective (tt/decode_all_reduce.py).
    decode_all_reduce = None
    # Set by the decoder layer (per call): decode returns o_proj's partials un-reduced, in decode_all_reduce's width
    # shards -- the layer's gate / up program reduces them (TtMinistral3MLP.decode_ar).
    defer_decode_reduce = False

    def __init__(self, device, torch_module, in_scale=None, meta_rope=False) -> None:
        """in_scale: optional per-input-feature scale (the input RMSNorm's gamma) folded into q/k/v.

        meta_rope: q and k carry each head's dims in interleaved-pair order (2j, 2j+1 <- HF's j, j + head_dim/2): the
        q/k weight columns are permuted per head, the rotary tables must come in the same order
        (TtMinistral3RotaryEmbedding(interleaved=True)), and rotate_half becomes a pair swap inside every 32-wide tile,
        so decode runs q's and k's RoPE as ONE fused rotary_embedding_llama each on the height-sharded heads. q . k is
        unchanged by a permutation applied to both, and V is not permuted, so attention is the same math."""
        self.device = device
        self.meta_rope = bool(meta_rope)
        self.tp = tp = _mesh_size(device)
        cfg = torch_module.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = hd = torch_module.head_dim
        self.scale = float(torch_module.scaling)
        rope = getattr(cfg, "rope_parameters", None) or {}
        self.l4_beta = rope.get("llama_4_scaling_beta")
        self.l4_max = rope.get("original_max_position_embeddings")
        assert (
            self.num_heads % tp == 0 and self.num_kv_heads % tp == 0
        ), f"heads {self.num_heads}/{self.num_kv_heads} not divisible by TP={tp}"
        self.local_heads = lh = self.num_heads // tp
        self.local_kv_heads = lkv = self.num_kv_heads // tp
        self.concat_heads = None
        if self.CPP_PREFILL_CONCAT:
            from models.demos.devstral_small_2_24b_instruct_2512.tt.concat_heads import ConcatHeads

            self.concat_heads = ConcatHeads(device)
        self.concat_heads_decode = None
        if self.CPP_DECODE_CONCAT:
            from models.demos.devstral_small_2_24b_instruct_2512.tt.concat_heads import ConcatHeadsDecode

            self.concat_heads_decode = ConcatHeadsDecode(device)

        is_mesh = tp > 1
        shard = lambda dim: ttnn.ShardTensorToMesh(device, dim=dim) if is_mesh else None
        self._repl = repl = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

        def _put(t, mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                t.detach().to(torch.bfloat16).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=memory_config,
            )

        wq = fold_input_scale(torch_module.q_proj.weight.T, in_scale)  # (in, Hq*hd)
        wk = fold_input_scale(torch_module.k_proj.weight.T, in_scale)  # (in, Hkv*hd)
        wv = fold_input_scale(torch_module.v_proj.weight.T, in_scale)
        if self.meta_rope:
            perm = torch.arange(hd).reshape(2, hd // 2).T.reshape(-1)  # [0, hd/2, 1, hd/2 + 1, ...]
            per_head = lambda w: w.reshape(w.shape[0], -1, hd)[:, :, perm].reshape(w.shape[0], -1)
            wq, wk = per_head(wq), per_head(wk)
        chunks = []
        for d in range(tp):
            q_sl = slice(d * lh * hd, (d + 1) * lh * hd)
            kv_sl = slice(d * lkv * hd, (d + 1) * lkv * hd)
            chunks += [wq[:, q_sl], wk[:, kv_sl], wv[:, kv_sl]]
        wqkv = torch.concat(chunks, dim=-1)
        # The prefill QKV multiplies a block-float8 copy (half the weight bytes; HiFi2, fp32 accumulation).
        self.wqkv = _put(wqkv, shard(-1), dtype=ttnn.bfloat8_b)
        # Decode reads its own copy of [q | k | v], DRAM width-sharded over the banks so each bank's reader streams
        # only its column slice, with the single-row activation width-sharded over QKV_DECODE_CORES: the interleaved
        # matmul has only N / 32 = 48 output tiles to spread, one per core, and streams the weight at ~190 GB/s.
        k_in, n_dev = wqkv.shape[0], wqkv.shape[1] // tp
        banks, cores = device.dram_grid_size().x, QKV_DECODE_CORES.num_cores
        self.wqkv_decode = _put(
            wqkv,
            shard(-1),
            dtype=DECODE_WEIGHT_DTYPE,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                    (k_in, -(-n_dev // (ttnn.TILE_SIZE * banks)) * ttnn.TILE_SIZE),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        self.qkv_act_cfg = ttnn.create_sharded_memory_config(
            (ttnn.TILE_SIZE, k_in // cores),
            core_grid=QKV_DECODE_CORES,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # One activation shard per multicast block, two readers per DRAM bank (3 weight columns each).
        self.qkv_per_core_n = n_dev // ttnn.TILE_SIZE // cores
        self.qkv_decode_pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k_in // ttnn.TILE_SIZE // cores,
            per_core_M=1,
            per_core_N=self.qkv_per_core_n,
            num_workers_per_dram_bank=2,
        )
        # Prefill o_proj is COLUMN-parallel: the heads' outputs are all_gathered (13312 x 1024 -> 4096 per device) and
        # each device multiplies them by its 1/TP column slice of W_o, which yields exactly its 1/TP hidden slice -- the
        # shard the decoder's sharded prefill residual adds -- with no reduce-scatter of 5120-wide partials. Device d's
        # heads are the contiguous block d of the 4096 inputs, so the gather restores W_o's input order.
        # The prefill o_proj weight is block-float8, like the prefill QKV weight (HiFi2 matmuls).
        self.wo = _put(torch_module.o_proj.weight.T, shard(-1), dtype=ttnn.bfloat8_b)
        # Decode reads block-float8 copies of its projection weights (DRAM-bound single-row matmuls).
        # DECODE_OPROJ_STREAM: the decode o_proj is the MLP down's C++ streaming program over its own bank-major copy
        # (tt/down_decode.py, bit-identical to the 1D-multicast matmul); then no interleaved decode copy is made.
        self.stream_o = None
        if self.DECODE_OPROJ_STREAM and tp > 1 and self.DECODE_REDUCE_DTYPE == ttnn.bfloat16:
            from models.demos.devstral_small_2_24b_instruct_2512.tt.down_decode import StreamDownDecode

            w_o = torch_module.o_proj.weight.T  # (Hq * hd, D): device d multiplies the rows of its own heads
            if StreamDownDecode.supports(device, int(w_o.shape[0]) // tp, int(w_o.shape[1]), DECODE_WEIGHT_DTYPE):
                put = lambda t, dtype: _put(t, shard(-2), dtype=dtype)
                self.stream_o = StreamDownDecode(
                    device, w_o, DECODE_WEIGHT_DTYPE, ttnn.MathFidelity.HiFi4, put, tp, rb=self.DECODE_OPROJ_PUSH_ROWS
                )
        self.wo_decode = None
        if self.stream_o is None:
            self.wo_decode = _put(torch_module.o_proj.weight.T, shard(-2), dtype=DECODE_WEIGHT_DTYPE)
        # Decode o_proj (K = 32 tiles, N = 160 tiles): 8-tile K multicast blocks (the auto config used 2), 2 N tiles
        # per core.
        grid = device.compute_with_storage_grid_size()
        self.wo_decode_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        # rotate_half(x) == x @ R  (HF convention: cat(-x2, x1); meta_rope: each pair (x_2j, x_2j+1) -> (-x_2j+1, x_2j)).
        if self.meta_rope:
            rot = self._pair_swap(hd)
            # Decode's fused RoPE reads one 32 x 32 pair-swap tile per batch row from its own core; prefill's reads
            # the single tile from DRAM.
            self.trans_mat = _put(self._pair_swap(ttnn.TILE_SIZE).repeat(1, 1, self.MAX_DECODE_BATCH, 1), repl)
            # ... and with DECODE_FUSED_QK one per q core and one per k core (q's and k's heads on disjoint cores)
            self.trans_mat_qk = None
            if self.DECODE_FUSED_QK:
                pair = self._pair_swap(ttnn.TILE_SIZE).repeat(1, 1, 2 * self.MAX_DECODE_BATCH, 1)
                self.trans_mat_qk = _put(pair, repl)
            self.prefill_trans_mat = _put(self._pair_swap(ttnn.TILE_SIZE).reshape(1, 1, ttnn.TILE_SIZE, -1), repl)
            # One pair-swap tile per core, height-sharded into L1 around each prefill RoPE (see _prefill_rope).
            n_cores = grid.x * grid.y
            self.prefill_trans_rows = _put(self._pair_swap(ttnn.TILE_SIZE).repeat(1, 1, n_cores, 1), repl)
            core_shard = lambda w: ttnn.create_sharded_memory_config(
                (ttnn.TILE_SIZE, w),
                core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.prefill_rope_l1_cfg, self.prefill_trans_l1_cfg = core_shard(hd), core_shard(ttnn.TILE_SIZE)
        else:
            half = hd // 2
            rot = torch.zeros(hd, hd)
            rot[torch.arange(half) + half, torch.arange(half)] = -1.0
            rot[torch.arange(half), torch.arange(half) + half] = 1.0
        self.rot = _put(rot.reshape(1, 1, hd, hd), repl)

        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )
        # The decode QKV (fp32 norm shards x block-float8 weight, 320 tile products a core on 24 cores) at HiFi2.
        self.qkv_decode_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Prefill SDPA is bound by its per-tile work (QK^T, softmax, PV per q chunk): HiFi4 matmuls, fp32 DEST kept.
        # HiFi2 (optimize run 2, ~12 ms of trace prefill) cost text logits PCC 0.998839 -> 0.997511 and, with the rest
        # of the reduced-precision prefill, test_e2e_image_text_to_text's prefill hidden PCC (floor 0.995): its image-
        # token rows amplify every layer's error ~10x more than text rows (0.995511 at HiFi4 vs 0.992736 at HiFi2 with
        # the HiFi4 down, HiFi2 gate/up and HiFi2 vision gate/up; see ministral3_m_l_p PREFILL_GATE_UP_LOFI_LAYERS).
        self.prefill_sdpa_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # The many-row prefill QKV and o_proj projections are compute-bound: HiFi2 (fp32 accumulation kept).
        self.prefill_proj_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # SDPA decode reads q straight from its head shards (batch row b on core (b % HEAD_GRID_X, b // HEAD_GRID_X)):
        # with sharded q, batch b's output core is core b of SDPA's grid in row-major order, so that grid is
        # HEAD_GRID_X wide (8 x 10: 2 cores per batch row = 1 per kv head, as on the full grid).
        # Decode SDPA multiplies q by the block-float8 K / V cache at HiFi2 (fp32 accumulation kept): 42-44 -> 40 us.
        self.sdpa_decode_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        grid = device.compute_with_storage_grid_size()
        # A fixed 4-tile key chunk: the chunk the kernel's dynamic sizing picks at every position >= 64 anyway (the
        # next power of two of the cached tiles, capped at fp32 DEST's 4), with compile-time subblocks (40 -> 21-34 us).
        # Its last chunk takes the causal mask as a separate add over the packed scores (the dynamic path adds it in
        # DEST): -0.00018 gate PCC, paid for by two fewer block-float4 down layers (ministral3_m_l_p).
        self.sdpa_decode_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(
                min(self.HEAD_GRID_X, grid.x), min(self.DECODE_SDPA_ROWS, grid.y)
            ),
            q_chunk_size=0,
            k_chunk_size=self.DECODE_SDPA_K_CHUNK,
            exp_approx_mode=False,
        )
        self._head_cfgs = {}

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    @staticmethod
    def _pair_swap(n):
        """(n, n) R with x @ R = (-x_1, x_0, -x_3, x_2, ...): rotate_half in interleaved-pair order."""
        rot = torch.zeros(n, n)
        rot[torch.arange(0, n, 2), torch.arange(1, n, 2)] = 1.0
        rot[torch.arange(1, n, 2), torch.arange(0, n, 2)] = -1.0
        return rot

    def _to_tt(self, t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self._repl,
        )

    def _head_cfg(self, batch):
        """HEIGHT-sharded L1 config holding one batch row's [heads (tile-padded), head_dim] per core."""
        if batch not in self._head_cfgs:
            self._head_cfgs[batch] = self._batch_shard_cfg(batch, self.head_dim)
        return self._head_cfgs[batch]

    def _batch_cores(self, batch):
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.num_cores_to_corerangeset(
            batch, ttnn.CoreCoord(min(self.HEAD_GRID_X, grid.x), grid.y), row_wise=True
        )

    def _batch_shard_cfg(self, batch, width):
        """HEIGHT-sharded L1 config with one (TILE_SIZE, width) shard per batch row on the decode head grid."""
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width),
            core_grid=self._batch_cores(batch),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _fused_decode_rope(self, batch):
        # The fused op runs on its input shard grid's bounding box, so the batch rows must fill a rectangle.
        w = min(self.HEAD_GRID_X, self.device.compute_with_storage_grid_size().x)
        return self.meta_rope and batch <= self.MAX_DECODE_BATCH and (batch <= w or batch % w == 0)

    def _fused_qk_rope(self, batch):
        # q on the head grid's first `batch` cores and k on the next `batch`: 2 * batch whole rows of the 8-wide grid
        w = min(self.HEAD_GRID_X, self.device.compute_with_storage_grid_size().x)
        return (
            self.DECODE_FUSED_QK
            and self.trans_mat_qk is not None
            and self._fused_decode_rope(batch)
            and (2 * batch) % w == 0
            and 2 * batch <= w * self.device.compute_with_storage_grid_size().y
        )

    def head_grid_x(self):
        return min(self.HEAD_GRID_X, self.device.compute_with_storage_grid_size().x)

    def query_shares_rope(self, max_position):
        """Whether q uses k's rotary tables at every position below max_position (query_rope_tables: the llama-4
        query scale is exactly 1 there)."""
        if self.l4_beta is None or not self.l4_max:
            return True
        return max_position is not None and int(max_position) <= int(self.l4_max)

    def decode_rope_gathered(self, gather, cur_pos, cos_tab, sin_tab):
        """decode_rope_tables' DECODE_FUSED_QK result (cos, sin, trans) with the step's cos / sin rows gathered from
        the resident tables of every position (tt/rope_gather.py) instead of derived, concatenated and resharded."""
        batch = gather.batch
        cfg = self._batch_shard_cfg(2 * batch, self.head_dim)
        cos, sin = gather(cur_pos, cos_tab, sin_tab, cfg)
        trans = self.trans_mat_qk
        if batch < self.MAX_DECODE_BATCH:
            trans = ttnn.slice(trans, (0, 0, 0, 0), (1, 1, 2 * batch * ttnn.TILE_SIZE, ttnn.TILE_SIZE))
        return cos, sin, ttnn.to_memory_config(trans, self._batch_shard_cfg(2 * batch, ttnn.TILE_SIZE))

    def decode_rope_tables(self, rope, batch):
        """Decode rope tables (cos, sin, q_cos, q_sin), each (1, B, 1, head_dim), height-sharded onto the decode head
        grid plus the pair-swap tiles, ONCE per forward for every layer: the fused RoPE reads them from the cores
        that hold the q / k heads. Unchanged when the fused path does not apply.

        With DECODE_FUSED_QK: (cos, sin, trans), each table (1, 2B, 1, head_dim) = q's rows then k's, height-sharded
        one row per core over the 2B cores the q and k heads occupy, for rotary_embedding_llama_fused_qk."""
        if self._fused_qk_rope(batch):
            q_cos, q_sin = rope[2:4] if len(rope) >= 4 else rope[:2]
            cfg = self._batch_shard_cfg(2 * batch, self.head_dim)
            both = lambda q_t, k_t: ttnn.to_memory_config(ttnn.concat([q_t, k_t], dim=1), cfg)
            trans = self.trans_mat_qk
            if batch < self.MAX_DECODE_BATCH:
                trans = ttnn.slice(trans, (0, 0, 0, 0), (1, 1, 2 * batch * ttnn.TILE_SIZE, ttnn.TILE_SIZE))
            trans = ttnn.to_memory_config(trans, self._batch_shard_cfg(2 * batch, ttnn.TILE_SIZE))
            return both(q_cos, rope[0]), both(q_sin, rope[1]), trans
        if not self._fused_decode_rope(batch):
            return rope
        cfg = self._head_cfg(batch)
        trans = self.trans_mat
        if batch < self.MAX_DECODE_BATCH:
            trans = ttnn.slice(trans, (0, 0, 0, 0), (1, 1, batch * ttnn.TILE_SIZE, ttnn.TILE_SIZE))
        trans = ttnn.to_memory_config(trans, self._batch_shard_cfg(batch, ttnn.TILE_SIZE))
        cos, sin = (ttnn.to_memory_config(t, cfg) for t in rope[:2])
        if len(rope) < 4 or (rope[2] is rope[0] and rope[3] is rope[1]):
            return cos, sin, cos, sin, trans  # q shares k's tables
        return (cos, sin) + tuple(ttnn.to_memory_config(t, cfg) for t in rope[2:4]) + (trans,)

    def _sharded_qkv_pc(self):
        if getattr(self, "_sharded_qkv_pc_cache", None) is None:
            # DECODE_SHARDED_READERS_PER_BANK readers per DRAM bank, each streaming 6 / readers of its bank's 6 columns
            readers = self.DECODE_SHARDED_READERS_PER_BANK
            banks = self.device.dram_grid_size().x
            self._sharded_qkv_pc_cache = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=self.DECODE_SHARDED_K_BLOCK,
                per_core_M=1,
                per_core_N=self.qkv_per_core_n * QKV_DECODE_CORES.num_cores // (readers * banks),
                num_workers_per_dram_bank=readers,
            )
        return self._sharded_qkv_pc_cache

    def _reduce(self, t, gather=True):
        """all_reduce of o_proj's fp32 partials: reduce-scatter in fp32 (the sum is exact to fp32), round the reduced
        1/TP shard to bf16 once and all_gather that -- half the gather bytes. The decoder adds the bf16 result into
        its fp32 residual unrounded (mixed-dtype ADD runs on the SFPU in fp32). gather=False: the reduced shard."""
        if self.tp == 1:
            return t
        ccl = tp_ccl_args()
        t = ttnn.reduce_scatter(t, dim=3, **ccl)
        if not gather:
            return t
        if t.dtype != ttnn.bfloat16:
            t = ttnn.typecast(t, ttnn.bfloat16)
        if self.decode_gather_mem is not None and int(t.shape[-2]) == ttnn.TILE_SIZE:
            return ttnn.all_gather(t, dim=3, memory_config=self.decode_gather_mem, **ccl)
        return ttnn.all_gather(t, dim=3, **ccl)

    def _rope(self, x, cos, sin):
        return ttnn.add(
            ttnn.multiply(x, cos),
            ttnn.multiply(ttnn.matmul(x, self.rot, compute_kernel_config=self.compute_cfg), sin),
        )

    def _prefill_mm_cfg(self, mt, nt):
        """minimal_matmul config of the prefill QKV / o_proj: _one_block_cfg's single block per core for short prompts;
        otherwise 8-tile M blocks with N blocks of exactly each core's (narrow) N range -- the 416-row QKV's 5-tile
        range ran the default 8-tile block, computing 6 tiles for 5 -- in 4 x 1 subblocks for an odd width, 1 x 4 for a
        multiple of 4 (QKV 994 -> 869 us). Blocking leaves each output tile's K accumulation order unchanged."""
        cfg = self._one_block_cfg(mt, nt)
        if cfg is not None:
            return cfg
        grid = self.device.compute_with_storage_grid_size()
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        n = -(-nt // (grid.y if minimal_mm.transposed(mt, nt, grid.x, grid.y) else grid.x))  # the split minimal_mm runs
        if n > 8:
            return None
        sh, sw = (1, 4) if n % 4 == 0 else (4, 1) if n % 2 else (2, 2)
        return ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=n,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=grid,
        )

    def _one_block_cfg(self, mt, nt):
        """minimal_matmul config giving every core ONE output block -- its whole (M, N) tile range -- when that range
        is small (short prompts): the default 8 x 8 blocks split e.g. the 128-row QKV's 12 x 5-tile range into an 8-
        and a 4-row block, streaming the weight columns twice. None (the default config) otherwise."""
        grid = self.device.compute_with_storage_grid_size()
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        gm, gn = (grid.x, grid.y) if minimal_mm.transposed(mt, nt, grid.x, grid.y) else (grid.y, grid.x)
        m, n = -(-mt // gm), -(-nt // gn)
        if m * n > 64 or m <= 8:
            return None
        sh, sw = (2, 2) if m % 2 == 0 and n % 2 == 0 else ((4, 1) if m % 4 == 0 else (2, 1) if m % 2 == 0 else (1, 1))
        return ttnn.MinimalMatmulConfig(
            M_block_size=m,
            K_block_size=8,
            N_block_size=n,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=grid,
        )

    def prefill_rope_tables(self, rope, s):
        """Prefill rope tables (cos, sin[, q_cos, q_sin]), each (1, 1, s, hd), extended ONCE per forward for every
        layer with their per-core layouts: (cos, sin, q_cos | None, q_sin | None, rows), rows = (cos, sin, q_cos,
        q_sin) as (1, 1, cores * 32, hd) DRAM tables whose tile i is sequence tile i % S_t -- the tile the prefill
        split hands core i when every core takes one sequence tile. _prefill_rope height-shards them into L1 around
        the op (one tile per core), which then reads them in place instead of from DRAM for every row; they stay out
        of L1 between layers (forward-lifetime L1 tables clash with the static buffers of the larger prefill ops).
        Unchanged when the fused RoPE does not apply."""
        grid = self.device.compute_with_storage_grid_size()
        cores, st = grid.x * grid.y, -(-int(s) // ttnn.TILE_SIZE)
        if not (self.meta_rope and self.L1_PREFILL_ROPE) or st > cores:
            return rope
        reps = -(-cores // st)

        def per_core(t):
            if reps > 1:
                t = ttnn.repeat(t, ttnn.Shape([1, 1, reps, 1]))
            if reps * st != cores:
                t = ttnn.slice(t, (0, 0, 0, 0), (1, 1, cores * ttnn.TILE_SIZE, self.head_dim))
            return t

        q = tuple(rope[2:4]) if len(rope) >= 4 else (None, None)
        cos, sin = per_core(rope[0]), per_core(rope[1])
        shared = q[0] is None or (q[0] is rope[0] and q[1] is rope[1])
        q_cos, q_sin = (cos, sin) if shared else (per_core(q[0]), per_core(q[1]))
        return tuple(rope[:2]) + q + ((cos, sin, q_cos, q_sin),)

    def _prefill_rope(self, t, cos, sin, rows=None, out_mem=None):
        """Fused prefill RoPE of the (B, H, S, hd) heads t (meta_rope pair order), run on _rope_regroup's view.
        rows: (cos, sin) per-core tables from prefill_rope_tables, moved into L1 for the op when _rope_regroup
        picks them (with the pair-swap tile) and freed after it."""
        shape = tuple(int(d) for d in t.shape)
        g, use_l1 = self._rope_regroup(shape, rows is not None)
        trans, l1 = self.prefill_trans_mat, ()
        if use_l1:
            l1 = (
                ttnn.to_memory_config(rows[0], self.prefill_rope_l1_cfg),
                ttnn.to_memory_config(rows[1], self.prefill_rope_l1_cfg),
                ttnn.to_memory_config(self.prefill_trans_rows, self.prefill_trans_l1_cfg),
            )
            cos, sin, trans = l1
        if g != shape[0]:
            t = ttnn.reshape(t, (g, shape[0] * shape[1] // g) + shape[2:])
        t = ttnn.experimental.rotary_embedding_llama(
            t,
            cos,
            sin,
            trans,
            is_decode_mode=False,
            compute_kernel_config=self.prefill_proj_cfg,
            memory_config=out_mem or ttnn.DRAM_MEMORY_CONFIG,
        )
        for x in l1:
            ttnn.deallocate(x)
        return t if g == shape[0] else ttnn.reshape(t, shape)

    def _rope_regroup(self, shape, l1=False):
        """(B', use_l1): the (B', B * H / B') regrouping of (B, H, S, hd) heads for rotary_embedding_llama, and whether
        it reads prefill_rope_tables' L1 copies. Its prefill split gives min(B', cores) batch slices the remaining
        cores over sequence tiles, each core every head of its slice. With DRAM tables the regrouping leaves the
        busiest core the least work: 32 users x 8 q heads x 13 sequence tiles leaves 96 cores with 5/5/3 tiles of 8
        heads (160 head tiles on the busiest), while 8 x 32 puts one sequence tile of 32 heads on each of 104 cores
        (128). The op reads cos and sin from DRAM again for every (sequence tile, head) row above 8 rows per core
        (four tile transfers per head tile instead of two), else once per batch slice. The L1 tables need one
        sequence tile per core (at most cores // S_t batch slices) and cost two transfers per head tile; they are
        used when that is less DRAM traffic on the busiest core (416 rows: q 8 x 32 at 64 transfers vs 128).
        The rows' positions (and so cos/sin) are unchanged, and the outer-dim reshape is a view."""
        b, h, s = shape[:3]
        grid = self.device.compute_with_storage_grid_size()
        cores, st, n = grid.x * grid.y, -(-s // ttnn.TILE_SIZE), b * h
        divisors = [g for g in range(1, n + 1) if n % g == 0]

        def split(g):  # (head tiles, DRAM tile transfers) on the busiest core with DRAM tables
            bp = min(g, cores)
            sp = min(cores // bp, st)
            seq = -(-st // sp)
            rows, slices = seq * (n // g), -(-g // bp)
            return slices * rows, slices * (4 * rows if rows > 8 else 2 * (rows + seq))

        g = min(divisors, key=lambda g: (split(g)[0], g != b)) if self.REGROUP_PREFILL_ROPE else b
        if l1 and st <= cores:
            g_l1 = max(d for d in divisors if d <= cores // st)
            if 2 * (n // g_l1) <= split(g)[1]:
                return g_l1, True
        return g, False

    def _llama4_scale(self, position_ids, shape):
        """Per-position query scale 1 + beta * log(1 + floor(pos / orig_max)), on device, in `shape`."""
        pos = position_ids
        if isinstance(pos, torch.Tensor):
            pos = self._to_tt(pos.reshape(shape).to(torch.float32), dtype=ttnn.float32)
        t = ttnn.floor(ttnn.multiply(pos, 1.0 / float(self.l4_max)))
        t = ttnn.log(ttnn.add(t, 1.0))
        scale = ttnn.add(ttnn.multiply(t, float(self.l4_beta)), 1.0)
        return ttnn.typecast(ttnn.reshape(scale, shape), ttnn.bfloat16)

    @classmethod
    def sdpa_len(cls, s):
        """Rows prefill SDPA runs on for an s-row prompt: one power-of-two key chunk (32..PREFILL_MAX_K_CHUNK) that
        holds it, else s rounded up to a multiple of PREFILL_MAX_K_CHUNK."""
        if s > cls.PREFILL_MAX_K_CHUNK:
            return int(math.ceil(s / cls.PREFILL_MAX_K_CHUNK) * cls.PREFILL_MAX_K_CHUNK)
        n = 32
        while n < s:
            n *= 2
        return n

    def query_rope_tables(self, cos, sin, position_ids, shape, max_position=None):
        """(cos, sin) for q with the llama-4 query scale folded in: rope(q) * s == q * (cos * s) + rot(q) * (sin * s).

        The scale depends only on the position, so the model builds these ONCE per forward and every layer shares
        them, instead of each layer re-deriving the scale (floor/log/...) and multiplying its q by it. Below
        original_max_position_embeddings the scale is exactly 1, so the shared tables are bit-identical; when every
        position is known to be below it (max_position: an exclusive bound, e.g. the KV capacity) q simply shares
        k's tables."""
        if self.l4_beta is None or not self.l4_max or position_ids is None:
            return cos, sin
        if max_position is not None and int(max_position) <= int(self.l4_max):
            return cos, sin
        scale = self._llama4_scale(position_ids, shape)
        return ttnn.multiply(cos, scale), ttnn.multiply(sin, scale)

    def _fill_cache(self, cache, kv):
        """Write prompt K or V (B, kv_heads, S, hd) into rows 0..S-1 of the resident (B, kv_heads, C, hd) cache.

        Only the S prompt rows are cast and written (rows past them are not read: decode SDPA is causal on
        cur_pos, and decode writes each position before attending to it). fill_cache fills one batch row's heads,
        so batch is folded into heads -- the same memory order in both tensors -- and one op writes every user. Its
        interleaved path assumes no core's share of the work spans two heads, so the prompt goes in height-sharded
        with one (user, kv head) slab of S rows per core."""
        b, h, s, d = kv.shape
        slots = b * h
        grid = self.device.compute_with_storage_grid_size()
        cast_on_slabs = self.PREFILL_KV_CAST_ON_SLABS and slots <= grid.x * grid.y
        if kv.dtype != cache.dtype and not cast_on_slabs:  # e.g. a block-float8 cache
            kv = ttnn.typecast(kv, cache.dtype)
        if slots > grid.x * grid.y:
            s_cap = cache.shape[2]
            if s < s_cap:
                kv = ttnn.pad(kv, padding=((0, 0), (0, 0), (0, s_cap - s), (0, 0)), value=0.0)
            ttnn.copy(kv, cache)
            return
        slab_cfg = ttnn.create_sharded_memory_config(
            (s, d),
            core_grid=ttnn.num_cores_to_corerangeset(slots, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        kv = ttnn.to_memory_config(ttnn.reshape(kv, (1, slots, s, d)), slab_cfg)
        if kv.dtype != cache.dtype:  # (cast_on_slabs) each core casts its own slab in L1
            kv_in, kv = kv, ttnn.typecast(kv, cache.dtype, memory_config=slab_cfg)
            ttnn.deallocate(kv_in)
        ttnn.fill_cache(ttnn.reshape(cache, (1, slots, cache.shape[2], d)), kv, 0)

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        kv_cache=None,
        cur_pos=None,
        mode="prefill",
        **kwargs,
    ):
        if mode == "decode":
            return self._decode(hidden_states, position_embeddings, position_ids, kv_cache, cur_pos)
        x = hidden_states
        in_rank = len(x.shape)
        if in_rank == 3:
            x = ttnn.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        b, _, s, _ = x.shape
        hd = self.head_dim

        cos, sin = position_embeddings[:2]
        q_tables = None
        if len(position_embeddings) >= 4 and position_embeddings[2] is not None:
            q_tables = tuple(position_embeddings[2:4])
        rows = position_embeddings[4] if len(position_embeddings) > 4 else None  # prefill_rope_tables' per-core tables
        if isinstance(cos, torch.Tensor):
            cos = self._to_tt(cos.to(torch.bfloat16).reshape(1, 1, s, hd))
        if isinstance(sin, torch.Tensor):
            sin = self._to_tt(sin.to(torch.bfloat16).reshape(1, 1, s, hd))
        mask = attention_mask
        if isinstance(mask, torch.Tensor):
            mask = self._to_tt(mask.reshape(1, 1, s, s).to(torch.bfloat16))

        grid = self.device.compute_with_storage_grid_size()
        qkv_bytes = 2 * int(b) * int(s) * int(self.wqkv.shape[-1])  # bf16
        heads_l1 = self.meta_rope and 0 < qkv_bytes // (grid.x * grid.y) <= self.PREFILL_HEADS_L1_BYTES
        qkv_l1 = heads_l1 and qkv_bytes // (grid.x * grid.y) <= self.PREFILL_QKV_L1_BYTES
        # Prefill QKV on the full grid (ttnn.linear's auto config left this many-row projection on a partial one).
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        qkv_cfg = self._prefill_mm_cfg(int(b) * int(s) // ttnn.TILE_SIZE, int(self.wqkv.shape[-1]) // ttnn.TILE_SIZE)
        if (
            self.PREFILL_QKV_HEADS_OUT
            and qkv_cfg is not None
            and minimal_mm.supported(x, self.wqkv, self.prefill_proj_cfg)
        ):
            # the QKV written straight into nlp_create_qkv_heads' q / k / v layouts (minimal_mm heads_out): q / k in L1
            # (PREFILL_HEADS_L1_BYTES) for the RoPEs, v in DRAM -- no split program, no v copy out of L1
            qk_mem = ttnn.L1_MEMORY_CONFIG if heads_l1 else ttnn.DRAM_MEMORY_CONFIG
            q, k, v = minimal_mm.matmul(
                x,
                self.wqkv,
                qkv_cfg,
                self.prefill_proj_cfg,
                dtype=ttnn.bfloat16,
                heads_out=(self.local_heads, self.local_kv_heads, qk_mem, ttnn.DRAM_MEMORY_CONFIG),
            )
            if heads_l1:
                heads_in = (q, k)
        else:
            qkv = self._prefill_mm(
                minimal_mm,
                x,
                self.wqkv,
                config=qkv_cfg,
                compute_kernel_config=self.prefill_proj_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG if qkv_l1 else ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.local_heads,
                num_kv_heads=self.local_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG if heads_l1 else None,
            )
            if qkv_l1:
                ttnn.deallocate(qkv)
            if heads_l1:  # see PREFILL_HEADS_L1_BYTES; the RoPEs below write q / k to DRAM
                heads_in = (q, k)
                v_l1, v = v, ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(v_l1)
        rope_k = rope_q = self._rope
        if self.meta_rope:
            # One fused RoPE per tensor over every user's heads (the prompt positions are shared by all rows): a
            # 32 x 32 pair-swap per head-dim tile instead of a 128 x 128 rotate-half matmul plus mul / mul / add.
            k_rows = None if rows is None else rows[:2]
            q_rows = None if rows is None else rows[2:4]
            q_core_bytes = (
                2 * int(q.shape[0]) * int(q.shape[1]) * int(q.shape[2]) * int(q.shape[3]) // (grid.x * grid.y)
            )
            rope_l1 = heads_l1 and q_core_bytes <= self.PREFILL_ROPE_OUT_L1_BYTES
            rope_mem = ttnn.L1_MEMORY_CONFIG if rope_l1 else None
            rope_k = lambda t, c, si: self._prefill_rope(t, c, si, k_rows, rope_mem)
            rope_q = lambda t, c, si: self._prefill_rope(t, c, si, q_rows, rope_mem)
        k = rope_k(k, cos, sin)
        if q_tables is not None:
            q = rope_q(q, *q_tables)  # llama-4 scale already folded into the shared q tables
        else:
            q = rope_q(q, cos, sin)
            if self.l4_beta is not None and self.l4_max and position_ids is not None:
                q = ttnn.multiply(q, self._llama4_scale(position_ids, (1, 1, s, 1)))
        if heads_l1:
            for t in heads_in:
                ttnn.deallocate(t)
        if kv_cache is not None:
            self._fill_cache(kv_cache[0], k)
            self._fill_cache(kv_cache[1], v)

        # SDPA's online softmax rounds its running state once per KEY chunk, which biases the output level
        # upward (layer-8 Q/K/V of the image prompt: +0.23% at k_chunk 32, +0.10% at 128, +0.02% at 512).
        # Image-token rows amplify that bias over 40 layers (prefill hidden PCC 0.95 on those rows at k_chunk
        # 32), so prefill attends over the widest key chunk: one chunk when the prompt fits in one, else chunks of
        # the widest. SDPA pads the sequence to whole chunks itself (no pad / slice ops); the pad keys sit after
        # every real row, so causal attention leaves the real rows unchanged. The q chunk does not affect the level.
        if s <= self.PREFILL_MAX_K_CHUNK:
            # One key chunk of s rounded up to an even tile count. Up to 256 rows the queries are one chunk too
            # (K/V stream once per head). Longer prompts take 160-row q chunks: the time follows (rounds of work
            # units over the grid) x (q tiles per unit), and 416 rows x 32 users x 8 heads in 3 q chunks is 768 units
            # = 7 even rounds on 110 cores (q224: 1041 us, q160: 881 us, q128: 1003 us). A 512-row key chunk keeps
            # 128-row q chunks (a 256 x 512 score block overflows L1).
            k_chunk = -(-s // 64) * 64
            if k_chunk <= 256:
                q_chunk = k_chunk
            elif k_chunk <= 448:
                q_chunk = self.PREFILL_WIDE_Q_CHUNK
            else:
                q_chunk = self.PREFILL_Q_CHUNK
        else:
            k_chunk, q_chunk = self.PREFILL_MAX_K_CHUNK, self.PREFILL_Q_CHUNK
        # With TP the concatenated heads only feed the all_gather: they go through L1 when they fit (the same budget
        # as the head split), so the concat writes and the gather reads no DRAM for them.
        attn_core_bytes = (qkv_bytes * 2 // 3) // (grid.x * grid.y)
        concat_l1 = self.tp > 1 and 0 < attn_core_bytes <= self.PREFILL_HEADS_L1_BYTES
        sdpa_l1 = concat_l1 and attn_core_bytes <= self.PREFILL_SDPA_OUT_L1_BYTES
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        sdpa_mem = ttnn.L1_MEMORY_CONFIG if sdpa_l1 else ttnn.DRAM_MEMORY_CONFIG
        from models.demos.devstral_small_2_24b_instruct_2512.tt import sdpa_prefill

        if sdpa_prefill.supported(q, k, v, mask is None, mask, self.prefill_sdpa_cfg):
            # the same program with the softmax's scale + exp as one SFPU pass (tt/sdpa_prefill.py), bit-identical
            attn = sdpa_prefill.sdpa(q, k, v, True, self.scale, sdpa_cfg, self.prefill_sdpa_cfg, sdpa_mem)
        else:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=mask is None,
                attn_mask=mask,
                scale=self.scale,
                program_config=sdpa_cfg,
                compute_kernel_config=self.prefill_sdpa_cfg,
                memory_config=sdpa_mem,
            )
        if (
            self.meta_rope and rope_l1
        ):  # the L1 RoPE outputs (PREFILL_ROPE_OUT_L1_BYTES), read by SDPA and the cache fill
            ttnn.deallocate(q)
            ttnn.deallocate(k)
        heads = attn
        concat_mem = ttnn.L1_MEMORY_CONFIG if concat_l1 else None
        if self.concat_heads is not None and self.concat_heads.supports(attn):
            attn = self.concat_heads(attn, concat_mem or ttnn.DRAM_MEMORY_CONFIG)
        else:
            attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=concat_mem)
        if sdpa_l1:
            ttnn.deallocate(heads)
        if self.tp > 1:
            # every head's output, for the column-parallel o_proj
            concat = attn
            attn = ttnn.all_gather(attn, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tp_ccl_args())
            if concat_l1:
                ttnn.deallocate(concat)
        # Prefill o_proj on the full grid too (ttnn.linear's auto 1D config ran it on 80 cores at ~30 TFLOP/s).
        # Its hidden shard (TP, no gather) goes to L1 for short prompts: the decoder's residual add reads it there and
        # frees it (see PREFILL_OPROJ_OUT_L1_BYTES).
        out_l1 = (
            self.tp > 1
            and not self.prefill_gather
            and 2 * int(b) * int(s) * int(self.wo.shape[-1]) // (grid.x * grid.y) <= self.PREFILL_OPROJ_OUT_L1_BYTES
        )
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        out = self._prefill_mm(
            minimal_mm,
            attn,
            self.wo,
            config=self._prefill_mm_cfg(int(b) * int(s) // ttnn.TILE_SIZE, int(self.wo.shape[-1]) // ttnn.TILE_SIZE),
            compute_kernel_config=self.prefill_proj_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.PREFILL_REDUCE_DTYPE if self.tp > 1 else self.OUT_DTYPE,
        )
        if self.tp > 1 and self.prefill_gather:
            out = ttnn.all_gather(out, dim=3, **tp_ccl_args())  # the full-width output of the unsharded path
        if in_rank == 3:
            return ttnn.reshape(out, (b, s, out.shape[-1]))
        return out

    @staticmethod
    def _prefill_mm(minimal_mm, x, w, config, compute_kernel_config, **kw):
        """ttnn.experimental.minimal_matmul, as tt/minimal_mm.py's program when it covers the call: the same arithmetic,
        the core ranges' padding skipped (SKIP_PADDING), which the power-capped prefill turns into speed."""
        if config is not None and minimal_mm.supported(x, w, compute_kernel_config):
            return minimal_mm.matmul(x, w, config, compute_kernel_config, **kw)
        return ttnn.experimental.minimal_matmul(x, w, config=config, compute_kernel_config=compute_kernel_config, **kw)

    def _decode(self, x, position_embeddings, position_ids, kv_cache, cur_pos):
        """x (1, 1, B, D) -> (1, 1, B, D); cos/sin (1, B, 1, hd); cur_pos int32 [B]; position_ids (B, 1, 1, 1) fp32."""
        b = x.shape[-2]
        head_cfg = self._head_cfg(b)
        if x.is_sharded():  # the input norm's own width shards (DECODE_SHARDED_INPUT)
            qkv = ttnn.linear(
                x,
                self.wqkv_decode,
                program_config=self._sharded_qkv_pc(),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.qkv_decode_cfg,
            )
            ttnn.deallocate(x)
        else:
            qkv = ttnn.linear(
                ttnn.to_memory_config(x, self.qkv_act_cfg),
                self.wqkv_decode,
                program_config=self.qkv_decode_pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.compute_cfg,
            )
        k_cache, v_cache = kv_cache
        if len(position_embeddings) == 3 and self.DECODE_SHARDED_HEADS:
            # DECODE_FUSED_QK (tables from decode_rope_tables): k's heads on the B cores after q's, so one fused RoPE
            # rotates both and one fused cache update appends K (k's cores) and V (q's cores).
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=self.local_heads,
                num_kv_heads=self.local_kv_heads,
                overlap_qk_coregrid=False,
                memory_config=self._batch_shard_cfg(2 * b, self.head_dim),
            )
            cos, sin, trans = position_embeddings
            q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q, k, cos, sin, trans, compute_kernel_config=self.compute_cfg
            )
            from models.demos.devstral_small_2_24b_instruct_2512.tt import kv_append

            if self.DECODE_KV_APPEND and kv_append.supports(k_cache, k, v_cache, v, cur_pos):
                kv_append.append(k_cache, k, v_cache, v, cur_pos)
            else:
                ttnn.experimental.paged_fused_update_cache(k_cache, k, v_cache, v, update_idxs_tensor=cur_pos)
            ttnn.deallocate(k)  # free the L1 head shards before SDPA's circular buffers are placed
            ttnn.deallocate(v)
            return self._decode_attend(q, k_cache, v_cache, cur_pos, head_cfg)
        if not self.DECODE_SHARDED_HEADS:
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=self.local_heads, num_kv_heads=self.local_kv_heads, memory_config=head_cfg
        )
        cos, sin = position_embeddings[:2]
        if len(position_embeddings) == 5:
            # Sharded tables from decode_rope_tables: q's and k's RoPE run fused in place on the head shards.
            q_cos, q_sin, trans = position_embeddings[2:5]
            rope = lambda t, c, s: ttnn.experimental.rotary_embedding_llama(
                t, c, s, trans, is_decode_mode=True, compute_kernel_config=self.compute_cfg
            )
            k, q = rope(k, cos, sin), rope(q, q_cos, q_sin)
            # SDPA reads q from these head shards (one (heads, head_dim) tile row per core, see sdpa_decode_cfg).
        else:
            k = self._rope(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
            if len(position_embeddings) == 4:
                q = self._rope(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), *position_embeddings[2:4])
            else:
                q = self._rope(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
                if self.l4_beta is not None and self.l4_max and position_ids is not None:
                    q = ttnn.multiply(q, self._llama4_scale(position_ids, (1, b, 1, 1)))
            k = ttnn.to_memory_config(k, head_cfg)
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos)
        ttnn.deallocate(k)  # free the L1 head shards before SDPA's circular buffers are placed
        ttnn.deallocate(v)
        return self._decode_attend(q, k_cache, v_cache, cur_pos, head_cfg)

    def _decode_attend(self, q, k_cache, v_cache, cur_pos, head_cfg):
        """SDPA-decode of the post-RoPE q head shards over the resident cache, head concat, o_proj and its reduce."""
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=self.sdpa_decode_cfg,
            compute_kernel_config=self.sdpa_decode_compute_cfg,
            # L1-interleaved: GQA allows no sharded SDPA output, and the head-shard reshard below reads it on-chip
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if self.concat_heads_decode is not None and self.concat_heads_decode.supports(attn):
            # (1, 1, B, heads * head_dim) L1-interleaved in one copy program, read straight out of SDPA's tiles
            heads, attn = attn, self.concat_heads_decode(attn)
            ttnn.deallocate(heads)
        else:
            # Concat heads per row with ttnn's fused decode op on the head shards (one batch row's tile-padded heads
            # per core) instead of a reshape that un-pads the 8 heads out of their 32-row tiles through DRAM.
            attn = ttnn.experimental.nlp_concat_heads_decode(
                ttnn.to_memory_config(attn, head_cfg), num_heads=self.local_heads
            )
            attn = ttnn.sharded_to_interleaved(attn, ttnn.L1_MEMORY_CONFIG)  # (1, 1, B, heads * head_dim)
        if self.stream_o is not None:  # one program streaming the bank-major o_proj copy (tt/down_decode.py)
            if self.decode_all_reduce is not None:  # partials in its width shards, one collective
                p = self.stream_o(attn, memory_config=self.decode_all_reduce.mem)
                return p if self.defer_decode_reduce else self.decode_all_reduce(p)
            return self._reduce(self.stream_o(attn))
        return self._reduce(
            ttnn.linear(
                attn,
                self.wo_decode,
                program_config=self.wo_decode_pc,
                dtype=self.DECODE_REDUCE_DTYPE if (self.DECODE_REDUCE_DTYPE and self.tp > 1) else self.OUT_DTYPE,
                compute_kernel_config=self.compute_cfg,
            )
        )


def build(device, torch_module=None):
    return TtMinistral3Attention.build(device, torch_module)


def ministral3_attention(device, torch_module=None):
    return TtMinistral3Attention.build(device, torch_module)
