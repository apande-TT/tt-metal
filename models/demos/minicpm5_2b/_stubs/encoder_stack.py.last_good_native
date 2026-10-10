# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `encoder_stack` (model.layers) for `openbmb/MiniCPM5-2B`.

MiniCPM5-2B is a decoder-only Llama-architecture LM with no vision tower, so the planner's
llama_vision_encoder mapping does not apply: the model's block stack is `model.layers`, a
ModuleList of LlamaDecoderLayer. Each block is a pre-norm Llama block built from the graduated
native RMSNorm / MLP / attention ports; the stack chains them in order. The PCC harness unwraps
the ModuleList to its first block.

The causal mask is built once at load time and sliced on device per call, instead of being
generated on device with ttnn.full + ttnn.triu — the device-side mask path was the one op chain
here not exercised by the other graduated components and the previous run stalled the readback.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minicpm5_2b._stubs.attention import TtAttention
from models.demos.minicpm5_2b._stubs.decoder_layer import TtMLP, TtRMSNorm

_MAX_MASK_LEN = 1024


class TtStackAttention(TtAttention):
    def __init__(self, device, torch_module) -> None:
        super().__init__(device, torch_module)
        # One causal mask built at load time; forward slices the [past_len:past_len+seq, :total] window.
        mask = torch.full((1, 1, _MAX_MASK_LEN, _MAX_MASK_LEN), -1e9).triu(diagonal=1)
        self.mask = self._to_tt(mask)

    def _causal_mask(self, seq, past_len):
        total = past_len + seq
        if total > _MAX_MASK_LEN:
            mask = ttnn.full((1, 1, seq, total), -1e9, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            return ttnn.triu(mask, diagonal=past_len + 1)
        return ttnn.slice(self.mask, (0, 0, past_len, 0), (1, 1, past_len + seq, total))

    def __call__(self, hidden_states, position_embeddings=None, past_key_values=None, **kwargs):
        x = hidden_states
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[-2], x.shape[-1]))
        b, seq = x.shape[0], x.shape[-2]

        q = ttnn.linear(x, self.wq, compute_kernel_config=self.compute_cfg)
        k = ttnn.linear(x, self.wk, compute_kernel_config=self.compute_cfg)
        v = ttnn.linear(x, self.wv, compute_kernel_config=self.compute_cfg)
        q = self._heads(q, self.n_heads, seq)
        k = self._heads(k, self.n_kv_heads, seq)
        v = self._heads(v, self.n_kv_heads, seq)

        cos, sin = self._cos_sin(position_embeddings, b, seq)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)

        past = self._past_kv(past_key_values)
        past_len = 0
        if past is not None:
            past_k, past_v, past_len = past
            k = ttnn.concat([past_k, k], dim=2)
            v = ttnn.concat([past_v, v], dim=2)
        k = ttnn.repeat_interleave(k, self.kv_groups, dim=1)
        v = ttnn.repeat_interleave(v, self.kv_groups, dim=1)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_cfg)
        scores = ttnn.multiply(scores, self.scale)
        if seq > 1:
            scores = ttnn.add(scores, self._causal_mask(seq, past_len))
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg)
        attn = ttnn.matmul(probs, v, compute_kernel_config=self.compute_cfg)

        attn = ttnn.permute(attn, (0, 2, 1, 3))
        attn = ttnn.reshape(attn, (b, seq, self.n_heads * self.head_dim))
        out = ttnn.linear(attn, self.wo, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out, None


class TtStackBlock:
    def __init__(self, device, torch_module):
        self.device = device
        self.input_layernorm = TtRMSNorm(device, torch_module.input_layernorm)
        self.post_attention_layernorm = TtRMSNorm(device, torch_module.post_attention_layernorm)
        self.self_attn = TtStackAttention(device, torch_module.self_attn)
        self.mlp = TtMLP(device, torch_module.mlp)

    def __call__(self, x, position_embeddings=None, past_key_values=None):
        attn_out, _ = self.self_attn(
            self.input_layernorm(x), position_embeddings=position_embeddings, past_key_values=past_key_values
        )
        h = ttnn.add(x, attn_out)
        mlp_out = self.mlp(self.post_attention_layernorm(h))
        return ttnn.add(h, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TtDecoderStack:
    """Chains the blocks of `model.layers` in order.

    `blocks` lets a caller supply already-built per-layer blocks (any callables with the
    (x, position_embeddings=..., past_key_values=..., **kw) signature); otherwise one TtStackBlock is
    built per HF layer. `kv_caches` (one per block) is handed to block i as `kv_cache`; any other
    keyword argument goes to every block.
    """

    def __init__(self, device, torch_module, blocks=None):
        self.device = device
        if blocks is not None:
            self.layers = list(blocks)
            return
        if isinstance(torch_module, (torch.nn.ModuleList, torch.nn.Sequential)):
            hf_blocks = list(torch_module)
        else:
            hf_blocks = [torch_module]
        self.layers = [TtStackBlock(device, blk) for blk in hf_blocks]

    def __call__(self, hidden_states, position_embeddings=None, past_key_values=None, kv_caches=None, **kwargs):
        x = hidden_states
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[-2], x.shape[-1]))
        for i, layer in enumerate(self.layers):
            extra = dict(kwargs)
            if kv_caches is not None:
                extra["kv_cache"] = kv_caches[i]
            x = layer(x, position_embeddings=position_embeddings, past_key_values=past_key_values, **extra)
        return x


def build(device, torch_module=None, blocks=None):
    return TtDecoderStack(device, torch_module, blocks=blocks)


def encoder_stack(device, torch_module=None, blocks=None):
    return TtDecoderStack(device, torch_module, blocks=blocks)
