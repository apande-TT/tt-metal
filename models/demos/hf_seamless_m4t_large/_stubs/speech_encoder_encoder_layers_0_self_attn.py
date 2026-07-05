# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `speech_encoder.encoder.layers.0.self_attn` of facebook/hf-seamless-m4t-large.

SeamlessM4TConformerSelfAttention with position_embeddings_type == "relative"
(Transformer-XL style). Q/K/V/O and pos-projection all run as ttnn.linear;
the head-dim relative-embedding shift, softmax and QK V reduction run on host
(mirrors the pattern of the graduated seamless_m4_t_conformer_adapter stub —
ttnn matmul on 4D head tensors + the T-XL relative shift is not worth the
complexity vs. the host reduction).
"""
from __future__ import annotations

import math

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["speech_encoder.encoder.layers.0.self_attn"]


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


class SpeechEncoderEncoderLayers0SelfAttn:
    def __init__(self, device, torch_module):
        self.device = device
        self.num_heads = torch_module.num_heads
        self.head_size = torch_module.head_size
        self.position_embeddings_type = torch_module.position_embeddings_type

        sd = torch_module.state_dict()

        self.w_q_w = ttnn.from_torch(
            sd["linear_q.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_q_b = ttnn.from_torch(
            sd["linear_q.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_k_w = ttnn.from_torch(
            sd["linear_k.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_k_b = ttnn.from_torch(
            sd["linear_k.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_v_w = ttnn.from_torch(
            sd["linear_v.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_v_b = ttnn.from_torch(
            sd["linear_v.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_o_w = ttnn.from_torch(
            sd["linear_out.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_o_b = ttnn.from_torch(
            sd["linear_out.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # relative-position weights (used when position_embeddings_type == "relative")
        self.w_pos_w = ttnn.from_torch(
            sd["linear_pos.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.pos_bias_u = sd["pos_bias_u"].to(torch.float32)  # (num_heads, head_size)
        self.pos_bias_v = sd["pos_bias_v"].to(torch.float32)

    def _apply_relative_shift(self, scores_bd):
        """Transformer-XL relative-position shift on scores_bd of
        shape (B, H, T, 2T-1) -> (B, H, T, T)."""
        B, H, T, twoTm1 = scores_bd.shape
        zero_pad = torch.zeros((B, H, T, 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.concat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded = scores_bd_padded.view(B, H, twoTm1 + 1, T)
        scores_bd = scores_bd_padded[:, :, 1:].view(B, H, T, twoTm1)
        scores_bd = scores_bd[:, :, :, : twoTm1 // 2 + 1]
        return scores_bd

    def __call__(self, hidden_states, attention_mask=None, relative_position_embeddings=None, *args, **kwargs):
        if isinstance(hidden_states, torch.Tensor):
            x_ttnn = ttnn.from_torch(
                hidden_states.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        else:
            x_ttnn = hidden_states

        q = ttnn.linear(x_ttnn, self.w_q_w, bias=self.w_q_b)
        k = ttnn.linear(x_ttnn, self.w_k_w, bias=self.w_k_b)
        v = ttnn.linear(x_ttnn, self.w_v_w, bias=self.w_v_b)

        q_t = ttnn.to_torch(q).to(torch.float32)
        k_t = ttnn.to_torch(k).to(torch.float32)
        v_t = ttnn.to_torch(v).to(torch.float32)

        B, S, _ = q_t.shape
        # (B, S, H, D) then (B, H, S, D)
        q_t = q_t.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        k_t = k_t.view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        v_t = v_t.view(B, S, self.num_heads, self.head_size).transpose(1, 2)

        if self.position_embeddings_type == "relative" and relative_position_embeddings is not None:
            # Project rel-pos through the (no-bias) linear_pos via ttnn.linear.
            if isinstance(relative_position_embeddings, torch.Tensor):
                rp_torch = relative_position_embeddings
            else:
                rp_torch = ttnn.to_torch(relative_position_embeddings)
            rp_ttnn = ttnn.from_torch(
                rp_torch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            proj_rp = ttnn.linear(rp_ttnn, self.w_pos_w)
            proj_rp = ttnn.to_torch(proj_rp).to(torch.float32)
            # (B, 2T-1, hidden) -> (B, 2T-1, H, D) -> (B, H, D, 2T-1)
            proj_rp = proj_rp.view(proj_rp.shape[0], -1, self.num_heads, self.head_size)
            proj_rp = proj_rp.transpose(1, 2).transpose(2, 3)  # (B, H, D, 2T-1)

            # (B, H, T, D) + pos_bias_u/v broadcast: pos_bias is (H, D)
            q_t_swap = q_t.transpose(1, 2)  # (B, T, H, D)
            q_with_bias_u = (q_t_swap + self.pos_bias_u).transpose(1, 2)  # (B, H, T, D)
            q_with_bias_v = (q_t_swap + self.pos_bias_v).transpose(1, 2)

            scores_ac = torch.matmul(q_with_bias_u, k_t.transpose(-2, -1))  # (B, H, T, T)
            scores_bd = torch.matmul(q_with_bias_v, proj_rp)  # (B, H, T, 2T-1)
            scores_bd = self._apply_relative_shift(scores_bd)  # (B, H, T, T)
            scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)
        else:
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(self.head_size)

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = ttnn.to_torch(attention_mask)
            scores = scores + attention_mask.to(scores.dtype)

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_t)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_size).contiguous()

        out_ttnn = ttnn.from_torch(
            out.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        out_ttnn = ttnn.linear(out_ttnn, self.w_o_w, bias=self.w_o_b)
        return out_ttnn


def build(device, torch_module):
    return SpeechEncoderEncoderLayers0SelfAttn(device, torch_module)


_instance = None


def speech_encoder_encoder_layers_0_self_attn(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `speech_encoder_encoder_layers_0_self_attn`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
