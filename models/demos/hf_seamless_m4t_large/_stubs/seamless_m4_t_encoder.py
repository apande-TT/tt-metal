# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `seamless_m4_t_encoder` (text encoder) of facebook/hf-seamless-m4t-large.

Implements the 24-layer text encoder. Per layer:
  self_attn_layer_norm -> self-attention -> residual add
  ffn_layer_norm -> ffn (linear -> relu -> linear) -> residual add
Then a final layer_norm.

Q/K/V/O linears and fc1/fc2 + layer_norms run on device via ttnn primitives;
attention softmax + matmuls happen in torch on host to match the pattern
used by the graduated adapter/encoder/decoder stubs.

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["text_encoder"]


def _resolve(obj, dotted):
    cur = obj
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if tok == "":
            continue
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur


def _to_ttnn(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _to_ttnn_bf8(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)


class SeamlessM4TEncoder:
    def __init__(self, device, torch_module):
        self.device = device
        cfg = torch_module.config
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.encoder_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.scaling = self.head_size**-0.5
        self.num_layers = len(torch_module.layers)
        self.eps = 1e-05

        self.embed_tokens_weight = torch_module.embed_tokens.weight.detach().to(torch.float32)
        self.embed_tokens_padding_idx = torch_module.embed_tokens.padding_idx
        self.embed_scale = float(getattr(torch_module.embed_tokens, "embed_scale", 1.0))

        self.pe_weights = torch_module.embed_positions.weights.detach().to(torch.float32)
        self.pe_padding_idx = torch_module.embed_positions.padding_idx
        self.pe_offset = torch_module.embed_positions.offset

        self.layers_w = []
        for layer in torch_module.layers:
            sd = layer.state_dict()
            wl = {
                "sa_ln_w": _to_ttnn(sd["self_attn_layer_norm.weight"], device),
                "sa_ln_b": _to_ttnn(sd["self_attn_layer_norm.bias"], device),
                "sa_q_w": _to_ttnn_bf8(sd["self_attn.q_proj.weight"].T.contiguous(), device),
                "sa_q_b": _to_ttnn(sd["self_attn.q_proj.bias"].reshape(1, -1), device),
                "sa_k_w": _to_ttnn_bf8(sd["self_attn.k_proj.weight"].T.contiguous(), device),
                "sa_k_b": _to_ttnn(sd["self_attn.k_proj.bias"].reshape(1, -1), device),
                "sa_v_w": _to_ttnn_bf8(sd["self_attn.v_proj.weight"].T.contiguous(), device),
                "sa_v_b": _to_ttnn(sd["self_attn.v_proj.bias"].reshape(1, -1), device),
                "sa_o_w": _to_ttnn_bf8(sd["self_attn.out_proj.weight"].T.contiguous(), device),
                "sa_o_b": _to_ttnn(sd["self_attn.out_proj.bias"].reshape(1, -1), device),
                "ffn_ln_w": _to_ttnn(sd["ffn_layer_norm.weight"], device),
                "ffn_ln_b": _to_ttnn(sd["ffn_layer_norm.bias"], device),
                "ffn_fc1_w": _to_ttnn_bf8(sd["ffn.fc1.weight"].T.contiguous(), device),
                "ffn_fc1_b": _to_ttnn(sd["ffn.fc1.bias"].reshape(1, -1), device),
                "ffn_fc2_w": _to_ttnn_bf8(sd["ffn.fc2.weight"].T.contiguous(), device),
                "ffn_fc2_b": _to_ttnn(sd["ffn.fc2.bias"].reshape(1, -1), device),
            }
            self.layers_w.append(wl)

        top_sd = torch_module.state_dict()
        self.w_top_ln_w = _to_ttnn(top_sd["layer_norm.weight"], device)
        self.w_top_ln_b = _to_ttnn(top_sd["layer_norm.bias"], device)

    def _embed_input_ids(self, input_ids):
        w = self.embed_tokens_weight
        embedded = torch.nn.functional.embedding(input_ids, w, padding_idx=self.embed_tokens_padding_idx)
        if self.embed_scale != 1.0:
            embedded = embedded * self.embed_scale
        return embedded

    def _embed_positions(self, input_ids):
        bsz, seq_len = input_ids.size()
        mask = input_ids.ne(self.pe_padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + self.pe_padding_idx
        return self.pe_weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def _self_attn(self, x_ttnn, w):
        q = ttnn.linear(x_ttnn, w["sa_q_w"], bias=w["sa_q_b"])
        k = ttnn.linear(x_ttnn, w["sa_k_w"], bias=w["sa_k_b"])
        v = ttnn.linear(x_ttnn, w["sa_v_w"], bias=w["sa_v_b"])

        q_t = ttnn.to_torch(q).to(torch.float32) * self.scaling
        k_t = ttnn.to_torch(k).to(torch.float32)
        v_t = ttnn.to_torch(v).to(torch.float32)

        B, L, C = q_t.shape
        q_t = q_t.view(B, L, self.num_heads, self.head_size).transpose(1, 2)
        k_t = k_t.view(B, L, self.num_heads, self.head_size).transpose(1, 2)
        v_t = v_t.view(B, L, self.num_heads, self.head_size).transpose(1, 2)

        scores = q_t @ k_t.transpose(-2, -1)
        probs = torch.softmax(scores, dim=-1)
        out = probs @ v_t
        out = out.transpose(1, 2).reshape(B, L, C).contiguous()

        out_ttnn = _to_ttnn(out.to(torch.bfloat16), self.device)
        out_ttnn = ttnn.linear(out_ttnn, w["sa_o_w"], bias=w["sa_o_b"])
        return out_ttnn

    def _apply_layer(self, i, x_ttnn):
        w = self.layers_w[i]

        residual = x_ttnn
        h = ttnn.layer_norm(x_ttnn, epsilon=self.eps, weight=w["sa_ln_w"], bias=w["sa_ln_b"])
        h = self._self_attn(h, w)
        h = ttnn.add(h, residual)

        residual = h
        h = ttnn.layer_norm(h, epsilon=self.eps, weight=w["ffn_ln_w"], bias=w["ffn_ln_b"])
        h = ttnn.linear(
            h,
            w["ffn_fc1_w"],
            bias=w["ffn_fc1_b"],
            activation="relu",
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        h = ttnn.linear(h, w["ffn_fc2_w"], bias=w["ffn_fc2_b"])
        h = ttnn.add(h, residual)
        return h

    def __call__(self, input_ids=None, attention_mask=None, inputs_embeds=None, *args, **kwargs):
        if input_ids is None and inputs_embeds is None:
            raise RuntimeError("SeamlessM4TEncoder native stub requires input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self._embed_input_ids(input_ids)
        elif not isinstance(inputs_embeds, torch.Tensor):
            inputs_embeds = ttnn.to_torch(inputs_embeds).to(torch.float32)

        if input_ids is not None:
            positions = self._embed_positions(input_ids)
            hidden_states = inputs_embeds + positions
        else:
            hidden_states = inputs_embeds

        x_ttnn = _to_ttnn(hidden_states.to(torch.bfloat16), self.device)
        for i in range(self.num_layers):
            x_ttnn = self._apply_layer(i, x_ttnn)

        x_ttnn = ttnn.layer_norm(x_ttnn, epsilon=self.eps, weight=self.w_top_ln_w, bias=self.w_top_ln_b)
        return x_ttnn


def build(device, torch_module):
    return SeamlessM4TEncoder(device, torch_module)


_instance = None


def seamless_m4_t_encoder(*args, **kwargs):
    global _instance
    if _instance is None:
        model = transformers.AutoModel.from_pretrained(
            HF_MODEL_ID, trust_remote_code=True, torch_dtype="bfloat16", low_cpu_mem_usage=True
        )
        model.eval()
        torch_sub = None
        for path in _CANDIDATE_SUBMODULE_PATHS:
            try:
                torch_sub = _resolve(model, path)
                break
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
        if torch_sub is None:
            raise RuntimeError("partial-stub: could not resolve `seamless_m4_t_encoder`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
