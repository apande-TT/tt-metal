# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_model` (Ministral3Model, `language_model`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    h   = embed_tokens(input_ids)            (or the caller's inputs_embeds)
    h   = layer_i(h, causal mask, rope(pos), pos)     for i in 0..n_layers-1
    out = norm(h)

Composed from the graduated child ports:
  * embed_tokens  `token_embed.py` (replicated table, ttnn.embedding)
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
    def __init__(self, device, torch_module, max_seq_len: int = 8192, n_layers=None) -> None:
        self.device = device
        cfg = torch_module.config
        depth = cfg.num_hidden_layers if n_layers is None else max(1, min(int(n_layers), cfg.num_hidden_layers))
        self.embed_tokens = TtTokenEmbed(device, torch_module.embed_tokens)
        # q / k in interleaved-pair head_dim order with tables to match, so decode's RoPE runs fused on the head shards
        # (TtMinistral3Attention meta_rope); attention itself is unchanged by the shared permutation.
        self.rotary_emb = TtMinistral3RotaryEmbedding(device, torch_module.rotary_emb, interleaved=True)
        self.layers = [TtMinistral3DecoderLayer(device, layer, meta_rope=True) for layer in torch_module.layers[:depth]]
        self.norm = TtMinistral3RMSNorm.build(device, torch_module.norm)
        self.tp = self.layers[0].tp
        self.head_dim = self.layers[0].self_attn.head_dim
        self.max_seq_len = max_seq_len
        self._repl = ttnn.ReplicateTensorToMesh(device) if self.tp > 1 else None
        self.compute_cfg = self.layers[0].compute_cfg

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

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
            x = self.embed_tokens(input_ids)  # (B, S, D)
            b, s, d = x.shape
            x = ttnn.reshape(x, (b, 1, s, d))
        else:
            x = inputs_embeds
            if len(x.shape) == 3:
                x = ttnn.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
            b, _, s, d = x.shape
        assert s <= self.max_seq_len and s % 32 == 0, f"seq {s} must be a tile multiple <= {self.max_seq_len}"

        pos = self._positions(position_ids, s)
        cos, sin = self.rotary_emb(x, position_ids=pos)  # (1, S, hd)
        cos = ttnn.reshape(cos, (1, 1, s, self.head_dim))
        sin = ttnn.reshape(sin, (1, 1, s, self.head_dim))
        # q's rope tables with the llama-4 query scale folded in, built once and shared by every layer.
        rope = (cos, sin) + self.layers[0].self_attn.query_rope_tables(cos, sin, pos, (1, 1, s, 1))
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
        out = self._bf16(self.norm(self._bf16(x)))
        return ttnn.reshape(out, (b, s, d))

    def _decode(self, input_ids, inputs_embeds, position_ids, kv_caches, cur_pos):
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)  # (1, B, D)
            x = ttnn.reshape(x, (1, 1, x.shape[-2], x.shape[-1]))
        else:
            x = inputs_embeds
        b = x.shape[-2]
        cos, sin = self.rotary_emb(x, position_ids=position_ids)  # (B, 1, hd)
        cos = ttnn.reshape(cos, (1, b, 1, self.head_dim))
        sin = ttnn.reshape(sin, (1, b, 1, self.head_dim))
        attn0 = self.layers[0].self_attn
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
    def _bf16(t):
        # the residual stream may be fp32 (TtMinistral3DecoderLayer.RESIDUAL_DTYPE); lm_head reads bf16
        return t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16)


def build(device, torch_module=None):
    return TtMinistral3Model.build(device, torch_module)


def ministral3_model(device, torch_module=None):
    return TtMinistral3Model.build(device, torch_module)
