# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_decoder_layer` (Ministral3DecoderLayer,
`language_model.layers.0`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    h   = x + self_attn(input_layernorm(x))
    out = h + mlp(post_attention_layernorm(h))          mlp = down(silu(gate(n)) * up(n))

Composed from the graduated child ports, each holding its own weights:
  * input_layernorm / post_attention_layernorm  `ministral3_r_m_s_norm.py` (replicated weight)
  * self_attn  `ministral3_attention.py` (q/k/v column-parallel by head with GQA groups kept local,
               o_proj row-parallel + all_reduce; prefill fills / decode appends the resident KV cache)
  * mlp        `ministral3_m_l_p.py` (gate/up column-parallel, down row-parallel + all_reduce)
so the residual between layers is full-width and replicated on every device.

Layouts: prefill x (B, 1, S, D) [or (B, S, D)], decode x (1, 1, B, D). The output keeps the input's rank.
"""
from __future__ import annotations

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_attention import (
    TtMinistral3Attention,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_m_l_p import TtMinistral3MLP
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_r_m_s_norm import (
    TtMinistral3RMSNorm,
)


class TtMinistral3DecoderLayer:
    # The residual stream is carried in fp32: bf16 residual adds round with an upward bias (each layer's output
    # ran ~0.1% high in level on fp32 input), which compounded to a +0.8% level drift by layer 36 and cost the
    # image-token rows most. Norms read a bf16 copy; attention / MLP outputs arrive in fp32 (their OUT_DTYPE).
    RESIDUAL_DTYPE = ttnn.float32

    def __init__(self, device, torch_module) -> None:
        self.device = device
        # Each norm's gamma is folded into the projections that read it (q/k/v; gate/up), so the norms apply no
        # weight and their outputs feed the matmuls without a separate gamma multiply.
        ln_in, ln_post = torch_module.input_layernorm, torch_module.post_attention_layernorm
        self.self_attn = TtMinistral3Attention(device, torch_module.self_attn, in_scale=ln_in.weight)
        self.tp = self.self_attn.tp
        self.input_layernorm = TtMinistral3RMSNorm.build(device, ln_in, fold_gamma=True)
        self.post_attention_layernorm = TtMinistral3RMSNorm.build(device, ln_post, fold_gamma=True)
        self.mlp = TtMinistral3MLP(device, torch_module.mlp, in_scale=ln_post.weight)
        self.compute_cfg = self.self_attn.compute_cfg

    @property
    def attn(self):
        return self.self_attn

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        kv_cache=None,
        cur_pos=None,
        mode="prefill",
        **kwargs,
    ):
        x = hidden_states
        in_shape = tuple(x.shape)
        if len(in_shape) == 3:
            x = ttnn.reshape(x, (in_shape[0], 1, in_shape[1], in_shape[2]))

        if self.RESIDUAL_DTYPE is not None and x.dtype != self.RESIDUAL_DTYPE:
            x = ttnn.typecast(x, self.RESIDUAL_DTYPE)
        if mode == "decode":
            n = self.input_layernorm.decode(x)  # sharded norm reads the fp32 residual directly
        else:
            n = self._bf16(self.input_layernorm(self._bf16(x)))  # rms_norm reads bf16 (fp32 input overflows L1 at 5120)
        a = self.self_attn(
            n,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            cur_pos=cur_pos,
            mode=mode,
        )
        h = self._residual_add(x, a)
        if mode == "decode":
            n = self.post_attention_layernorm.decode(h)
        else:
            n = self._bf16(self.post_attention_layernorm(self._bf16(h)))
        m = self.mlp(n, mode=mode)
        out = self._residual_add(h, m)
        if len(in_shape) == 3:
            return ttnn.reshape(out, in_shape)
        return out

    @staticmethod
    def _residual_add(x, delta):
        # A bf16 attention / MLP output is added into the fp32 residual without a cast op: binary ADD runs on the
        # SFPU in fp32 for mixed dtypes, so the bf16 operand is widened exactly.
        return ttnn.add(x, delta) if delta.dtype == x.dtype else ttnn.add(x, delta, dtype=x.dtype)

    @staticmethod
    def _as(t, dtype):
        return t if t.dtype == dtype else ttnn.typecast(t, dtype)

    def _bf16(self, t):
        return self._as(t, ttnn.bfloat16)


def build(device, torch_module=None):
    return TtMinistral3DecoderLayer.build(device, torch_module)


def ministral3_decoder_layer(device, torch_module=None):
    return TtMinistral3DecoderLayer.build(device, torch_module)
