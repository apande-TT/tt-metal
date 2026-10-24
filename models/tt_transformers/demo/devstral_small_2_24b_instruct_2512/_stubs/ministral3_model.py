# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_model` (Ministral3Model, `language_model`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    h   = embed_tokens(input_ids)            (or the caller's inputs_embeds)
    h   = layer_i(h, causal mask, rope(pos), pos)     for i in 0..n_layers-1
    out = norm(h)

Composed from the graduated child ports:
  * embed_tokens  `token_embed.py` (replicated table, ttnn.embedding; prefill gathers from its hidden-sharded copy)
  * rotary_emb    `ministral3_rotary_embedding.py` (yarn cos/sin evaluated on device from positions)
  * layers        `ministral3_decoder_layer.py` (the TP decoder layer; a plain list of one type)
  * norm          `ministral3_r_m_s_norm.py`

Modes:
  * prefill  input_ids (B, S) / inputs_embeds (B, 1, S, D); positions (1, 1, S, 1) fp32 (built from
             position_ids, default 0..S-1). With `kv_caches` (one (k, v) pair per layer) the prompt K/V
             are written into the resident cache. Returns (B, S, D).
  * decode   input_ids (1, B) uint32 / inputs_embeds (1, 1, B, D); position_ids (B, 1, 1, 1) fp32 and
             cur_pos int32 [B] give each row's write position. Returns (1, 1, B, D).

Tensor-parallel over the mesh (TP = number of devices): decoder layers are column/row parallel with an
all_reduce per block; embed_tokens, the final norm and the rotary tables are replicated.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_attention import tp_ccl_args
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_decoder_layer import (
    TtMinistral3DecoderLayer,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_r_m_s_norm import (
    TtMinistral3RMSNorm,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_rotary_embedding import (
    TtMinistral3RotaryEmbedding,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.token_embed import TtTokenEmbed


class TtMinistral3Model:
    # structural try (run 8): the prefill final norm on the hidden shards (see _sharded_final_norm)
    SHARDED_FINAL_NORM = True

    def __init__(self, device, torch_module, max_seq_len: int = 8192, n_layers=None) -> None:
        self.device = device
        cfg = torch_module.config
        depth = cfg.num_hidden_layers if n_layers is None else max(1, min(int(n_layers), cfg.num_hidden_layers))
        self.embed_tokens = TtTokenEmbed(device, torch_module.embed_tokens)
        # q / k in interleaved-pair head_dim order with tables to match, so decode's RoPE runs fused on the head shards
        # (TtMinistral3Attention meta_rope); attention itself is unchanged by the shared permutation.
        self.rotary_emb = TtMinistral3RotaryEmbedding(device, torch_module.rotary_emb, interleaved=True)
        self.layers = [
            TtMinistral3DecoderLayer(device, layer, meta_rope=True, layer_idx=i)
            for i, layer in enumerate(torch_module.layers[:depth])
        ]
        self.norm = TtMinistral3RMSNorm.build(device, torch_module.norm)
        self.tp = self.layers[0].tp
        # structural try (run 8, SHARDED_FINAL_NORM): the final norm's checkpoint gamma split over the TP mesh, each
        # device multiplying only its own hidden columns before the gather
        self._gamma_shard = None
        if self.SHARDED_FINAL_NORM and self.tp > 1 and self.norm.gamma is not None:
            self._gamma_shard = ttnn.from_torch(
                torch_module.norm.weight.detach().to(torch.bfloat16).reshape(1, 1, 1, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
            )
        # the last layer's prefill output only feeds the bf16 all_gather below (see bf16_prefill_out)
        self.layers[-1].bf16_prefill_out = True
        self.head_dim = self.layers[0].self_attn.head_dim
        self.max_seq_len = max_seq_len
        self._repl = ttnn.ReplicateTensorToMesh(device) if self.tp > 1 else None
        self.compute_cfg = self.layers[0].compute_cfg
        # (cos, sin) tables of every decode position and the program that gathers a step's rows (prepare_decode_rope)
        self._decode_rope = None
        self._rope_gather = None
        # Decode's token embedding as one program writing the fp32 residual's width shards (tt/embed_decode.py)
        self._embed_decode = None
        L = TtMinistral3DecoderLayer
        if self.DECODE_EMBED_SHARDED and self.tp > 1 and L.SHARDED_DECODE_RESIDUAL and L.RESIDUAL_DTYPE == ttnn.float32:
            from models.demos.devstral_small_2_24b_instruct_2512.tt.embed_decode import EmbedDecode

            d = self.layers[0].hidden
            mem = self.layers[0].input_layernorm.decode_mem(d)
            if EmbedDecode.supports(mem, d):
                self._embed_decode = EmbedDecode(device, mem, d)

    # Decode reads its RoPE rows from tables of every position below the KV capacity, built once (rope_gather.py),
    # instead of deriving them on device each step.
    DECODE_ROPE_TABLES = True
    # Decode's token rows gathered, widened to fp32 and tilized straight into the first layer's width-sharded residual
    # layout by one program (tt/embed_decode.py) instead of a one-core tiled embedding, a typecast and a reshard.
    DECODE_EMBED_SHARDED = True

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def prepare_decode_rope(self, capacity, batch):
        """Build the decode RoPE tables for positions 0 .. capacity - 1 (the KV capacity bounds every decode position)
        when decode runs the fused q / k RoPE with q sharing k's tables; otherwise decode derives them per step."""
        attn0 = self.layers[0].self_attn
        if not (self.DECODE_ROPE_TABLES and attn0._fused_qk_rope(batch) and attn0.query_shares_rope(capacity)):
            return
        from models.demos.devstral_small_2_24b_instruct_2512.tt.rope_gather import RopeGather

        self._decode_rope = self.rotary_emb.table(int(capacity))
        self._rope_gather = RopeGather(self.device, batch, self.head_dim, attn0.head_grid_x())

    def _positions(self, position_ids, s):
        """(1, 1, S, 1) fp32 prefill positions on device (a device tensor is used as given)."""
        if isinstance(position_ids, ttnn.Tensor):
            return position_ids
        if position_ids is None:
            pos = torch.arange(s, dtype=torch.float32)
        else:
            pos = position_ids.reshape(-1)[-s:].to(torch.float32)
        return ttnn.from_torch(
            pos.reshape(1, 1, s, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self._repl,
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        kv_caches=None,
        cur_pos=None,
        mode="prefill",
        **kwargs,
    ):
        if isinstance(attention_mask, torch.Tensor):
            assert bool((attention_mask != 0).all()), "padded prompts are not supported by this prefill port"
        if mode == "decode":
            return self._decode(input_ids, inputs_embeds, position_ids, kv_caches, cur_pos)

        if inputs_embeds is None:
            x = self.prefill_embeds(input_ids)
        else:
            x = inputs_embeds
            if len(x.shape) == 3:
                x = ttnn.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        b, _, s, _ = x.shape
        d = self.layers[0].hidden
        assert s <= self.max_seq_len and s % 32 == 0, f"seq {s} must be a tile multiple <= {self.max_seq_len}"

        pos = self._positions(position_ids, s)
        # bf16 tables (rotary_emb casts them to its x's dtype; the prompt embeddings may already be the fp32 residual)
        cos, sin = self.rotary_emb(None, position_ids=pos)  # (1, S, hd)
        cos = ttnn.reshape(cos, (1, 1, s, self.head_dim))
        sin = ttnn.reshape(sin, (1, 1, s, self.head_dim))
        # q's rope tables with the llama-4 query scale folded in, built once and shared by every layer.
        rope = (cos, sin) + self.layers[0].self_attn.query_rope_tables(cos, sin, pos, (1, 1, s, 1))
        rope = self.layers[0].self_attn.prefill_rope_tables(rope, s)  # + per-core L1 copies for the fused RoPE
        shard = self.tp > 1 and TtMinistral3DecoderLayer.SHARD_PREFILL_RESIDUAL
        if shard and int(x.shape[-1]) == d:
            # The layers carry the prefill residual hidden-sharded (TtMinistral3DecoderLayer.SHARD_PREFILL_RESIDUAL).
            x = ttnn.mesh_partition(x, dim=3)
        for i, layer in enumerate(self.layers):
            # attention_mask=None -> causal SDPA, the mask HF builds for an unpadded prompt.
            x = layer(
                x,
                attention_mask=None,
                position_ids=pos,
                position_embeddings=rope,
                kv_cache=None if kv_caches is None else kv_caches[i],
                mode="prefill",
            )
        if shard and self._gamma_shard is not None and int(x.shape[-1]) * self.tp == d:
            out = self._sharded_final_norm(self._bf16(x))
            return ttnn.reshape(out, (b, s, d))
        if shard:
            x = ttnn.all_gather(self._bf16(x), dim=3, **tp_ccl_args())
        out = self._bf16(self.norm(self._bf16(x)))
        return ttnn.reshape(out, (b, s, d))

    def _sharded_final_norm(self, x):
        """The final RMSNorm on the bf16 hidden shard x, through the norm's own call (TtMinistral3RMSNorm
        _sharded_prefill): stats all_gather + shard norm * gamma columns + one gather of the bf16 result."""
        layer = self.layers[-1]
        ctx = dict(hidden=layer.hidden, stats_mask=layer._stats_mask, gamma_shard=self._gamma_shard, ccl=tp_ccl_args())
        return self.norm(x, hidden_shard=ctx)

    # The sharded gather's bf16 slice goes through L1 (interleaved, <= this many bytes per bank) on its way to the
    # residual dtype: the typecast reads it there instead of from DRAM, and it is freed right after.
    PREFILL_EMBED_L1_BYTES = 384 * 1024

    def prefill_embeds(self, input_ids):
        """(B, 1, S, D) prompt embeddings of input_ids (B, S), or, when the layers carry the prefill residual
        hidden-sharded, this device's (B, 1, S, D / TP) slice gathered straight from the sharded table (in the
        residual dtype, or kept bf16 with its per-token norm statistics handed to the first layer)."""
        shard = self.tp > 1 and TtMinistral3DecoderLayer.SHARD_PREFILL_RESIDUAL
        if not (shard and self.embed_tokens.weight_shard is not None):
            x = self.embed_tokens(input_ids)
            b, s, d = x.shape
            return ttnn.reshape(x, (b, 1, s, d))
        b, s = (int(v) for v in input_ids.shape)
        if self.embed_tokens.sq_table is not None and TtMinistral3DecoderLayer.accepts_bf16_input():
            # The bf16 slice is the first layer's residual as is (its fused residual add widens it to fp32) and its
            # input norm reads the gathered per-token sums of squares, so nothing casts or reduces the embeddings.
            x = ttnn.reshape(self.embed_tokens.shard(input_ids), (b, 1, s, self.embed_tokens.shard_dim))
            sq = ttnn.reshape(self.embed_tokens.shard_sq(input_ids), (b, 1, s, ttnn.TILE_SIZE))
            TtMinistral3DecoderLayer.carry_input_stats(x, sq)
            return x
        grid = self.device.compute_with_storage_grid_size()
        per_bank = b * s * self.embed_tokens.shard_dim * 2 / (grid.x * grid.y)
        rd = TtMinistral3DecoderLayer.RESIDUAL_DTYPE
        l1 = rd is not None and rd != ttnn.bfloat16 and per_bank <= self.PREFILL_EMBED_L1_BYTES
        x = self.embed_tokens.shard(input_ids, memory_config=ttnn.L1_MEMORY_CONFIG if l1 else None)
        if l1:
            y = ttnn.typecast(x, rd, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x)
            x = y
        return ttnn.reshape(x, (b, 1, s, x.shape[-1]))

    def _decode(self, input_ids, inputs_embeds, position_ids, kv_caches, cur_pos):
        if inputs_embeds is None and self._embed_decode is not None and self._embeds_rows(input_ids):
            # (1, 1, B, D) fp32 in the residual's width shards (the token_embed stub runs the program)
            x = self.embed_tokens(input_ids, decode_program=self._embed_decode)
        elif inputs_embeds is None:
            x = self.embed_tokens(input_ids)  # (1, B, D)
            x = ttnn.reshape(x, (1, 1, x.shape[-2], x.shape[-1]))
        else:
            x = inputs_embeds
        b = x.shape[-2]
        attn0 = self.layers[0].self_attn
        if self._rope_gather is not None and self._rope_gather.batch == b and cur_pos is not None:
            # this step's rows of the resident tables, straight into the fused RoPE's shards (prepare_decode_rope)
            rope = attn0.decode_rope_gathered(self._rope_gather, cur_pos, *self._decode_rope)
        else:
            cos, sin = self.rotary_emb(x, position_ids=position_ids)  # (B, 1, hd)
            cos = ttnn.reshape(cos, (1, b, 1, self.head_dim))
            sin = ttnn.reshape(sin, (1, b, 1, self.head_dim))
            # Every decode position is below the resident KV capacity (dim 2 of the cache).
            capacity = int(kv_caches[0][0].shape[2]) if kv_caches else None
            rope = (cos, sin) + attn0.query_rope_tables(cos, sin, position_ids, (1, b, 1, 1), max_position=capacity)
            rope = attn0.decode_rope_tables(rope, b)
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                position_ids=position_ids,
                position_embeddings=rope,
                kv_cache=kv_caches[i],
                cur_pos=cur_pos,
                mode="decode",
            )
        return self.norm.decode(x)

    @staticmethod
    def _embeds_rows(input_ids):
        """Whether input_ids is the decode state's (1, B <= 32) uint32 row-major id row (EmbedDecode's input)."""
        return (
            isinstance(input_ids, ttnn.Tensor)
            and input_ids.layout == ttnn.ROW_MAJOR_LAYOUT
            and input_ids.dtype == ttnn.uint32
            and len(input_ids.shape) == 2
            and int(input_ids.shape[0]) == 1
            and int(input_ids.shape[1]) <= ttnn.TILE_SIZE
        )

    @staticmethod
    def _bf16(t):
        # the residual stream may be fp32 (TtMinistral3DecoderLayer.RESIDUAL_DTYPE); lm_head reads bf16
        return t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)


def build(device, torch_module=None):
    return TtMinistral3Model.build(device, torch_module)


def ministral3_model(device, torch_module=None):
    return TtMinistral3Model.build(device, torch_module)
