# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `seamless_m4_t_conformer_encoder` of facebook/hf-seamless-m4t-large.

Implements the 24-layer speech-conformer stack. Per layer:
  ffn1_layer_norm -> ffn1 (linear -> silu -> linear) -> scaled residual add
  self_attn_layer_norm -> self_attn (rel-pos attention) -> residual add
  conv_module (LN -> pointwise_conv1 -> GLU -> depthwise_conv -> BN -> silu -> pointwise_conv2) -> residual add
  ffn2_layer_norm -> ffn2 (linear -> silu -> linear) -> scaled residual add
  final_layer_norm

FFNs, layer-norms, and the pointwise convs (kernel=1) run as ttnn.linear /
ttnn.layer_norm / ttnn.silu on device. The depthwise conv (kernel=31, SAME
padding) + batch_norm + GLU pair round-trips to torch on host; the rel-pos
attention math (matmul-a-c / matmul-b-d shift trick + softmax) also runs
on host because the shift is awkward in ttnn.

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["speech_encoder.encoder"]


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


class SeamlessM4TConformerEncoder:
    def __init__(self, device, torch_module):
        self.device = device
        cfg = torch_module.config
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.speech_encoder_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.num_layers = len(torch_module.layers)
        self.dw_kernel = cfg.conv_depthwise_kernel_size
        self.eps = cfg.layer_norm_eps

        # Snapshot the relative positional-embedding buffer once. The
        # buffer covers 2*max_source_positions-1 positions and gets sliced
        # at forward time based on the actual input length.
        self._pe_buffer = torch_module.embed_positions.pe.detach().to(torch.float32).clone()

        # Per-layer weights.
        self.layers_w = []
        self.layers_torch = []

        for layer in torch_module.layers:
            sd = layer.state_dict()
            wl = {
                "ffn1_ln_w": _to_ttnn(sd["ffn1_layer_norm.weight"], device),
                "ffn1_ln_b": _to_ttnn(sd["ffn1_layer_norm.bias"], device),
                "ffn1_int_w": _to_ttnn(sd["ffn1.intermediate_dense.weight"].T.contiguous(), device),
                "ffn1_int_b": _to_ttnn(sd["ffn1.intermediate_dense.bias"].reshape(1, -1), device),
                "ffn1_out_w": _to_ttnn(sd["ffn1.output_dense.weight"].T.contiguous(), device),
                "ffn1_out_b": _to_ttnn(sd["ffn1.output_dense.bias"].reshape(1, -1), device),
                "sa_ln_w": _to_ttnn(sd["self_attn_layer_norm.weight"], device),
                "sa_ln_b": _to_ttnn(sd["self_attn_layer_norm.bias"], device),
                "sa_q_w": _to_ttnn(sd["self_attn.linear_q.weight"].T.contiguous(), device),
                "sa_q_b": _to_ttnn(sd["self_attn.linear_q.bias"].reshape(1, -1), device),
                "sa_k_w": _to_ttnn(sd["self_attn.linear_k.weight"].T.contiguous(), device),
                "sa_k_b": _to_ttnn(sd["self_attn.linear_k.bias"].reshape(1, -1), device),
                "sa_v_w": _to_ttnn(sd["self_attn.linear_v.weight"].T.contiguous(), device),
                "sa_v_b": _to_ttnn(sd["self_attn.linear_v.bias"].reshape(1, -1), device),
                "sa_o_w": _to_ttnn(sd["self_attn.linear_out.weight"].T.contiguous(), device),
                "sa_o_b": _to_ttnn(sd["self_attn.linear_out.bias"].reshape(1, -1), device),
                "cm_ln_w": _to_ttnn(sd["conv_module.layer_norm.weight"], device),
                "cm_ln_b": _to_ttnn(sd["conv_module.layer_norm.bias"], device),
                "cm_pc1_w": _to_ttnn(sd["conv_module.pointwise_conv1.weight"].squeeze(-1).T.contiguous(), device),
                "cm_pc2_w": _to_ttnn(sd["conv_module.pointwise_conv2.weight"].squeeze(-1).T.contiguous(), device),
                "ffn2_ln_w": _to_ttnn(sd["ffn2_layer_norm.weight"], device),
                "ffn2_ln_b": _to_ttnn(sd["ffn2_layer_norm.bias"], device),
                "ffn2_int_w": _to_ttnn(sd["ffn2.intermediate_dense.weight"].T.contiguous(), device),
                "ffn2_int_b": _to_ttnn(sd["ffn2.intermediate_dense.bias"].reshape(1, -1), device),
                "ffn2_out_w": _to_ttnn(sd["ffn2.output_dense.weight"].T.contiguous(), device),
                "ffn2_out_b": _to_ttnn(sd["ffn2.output_dense.bias"].reshape(1, -1), device),
                "final_ln_w": _to_ttnn(sd["final_layer_norm.weight"], device),
                "final_ln_b": _to_ttnn(sd["final_layer_norm.bias"], device),
            }
            tl = {
                "lp_w": sd["self_attn.linear_pos.weight"].to(torch.float32),  # (C, C)
                "pos_bias_u": layer.self_attn.pos_bias_u.detach().to(torch.float32),
                "pos_bias_v": layer.self_attn.pos_bias_v.detach().to(torch.float32),
                "dw_conv_weight": sd["conv_module.depthwise_conv.weight"].to(torch.float32),
                "bn_weight": sd["conv_module.batch_norm.weight"].to(torch.float32),
                "bn_bias": sd["conv_module.batch_norm.bias"].to(torch.float32),
                "bn_running_mean": layer.conv_module.batch_norm.running_mean.detach().to(torch.float32),
                "bn_running_var": layer.conv_module.batch_norm.running_var.detach().to(torch.float32),
                "bn_eps": float(layer.conv_module.batch_norm.eps),
            }
            self.layers_w.append(wl)
            self.layers_torch.append(tl)

        # Top-level layer norm (applied AFTER the layer stack).
        top_sd = torch_module.state_dict()
        self.w_top_ln_w = _to_ttnn(top_sd["layer_norm.weight"], device)
        self.w_top_ln_b = _to_ttnn(top_sd["layer_norm.bias"], device)

    def _get_rel_pos_emb(self, L):
        """Slice the pe buffer to (1, 2L-1, hidden_size)."""
        pe = self._pe_buffer
        mid = pe.size(1) // 2
        start_idx = mid - L + 1
        end_idx = mid + L
        return pe[:, start_idx:end_idx]  # (1, 2L-1, C)

    def _self_attn(self, x_ttnn, rel_pos_emb, w, t):
        # Q, K, V via ttnn linears on device.
        q = ttnn.linear(x_ttnn, w["sa_q_w"], bias=w["sa_q_b"])
        k = ttnn.linear(x_ttnn, w["sa_k_w"], bias=w["sa_k_b"])
        v = ttnn.linear(x_ttnn, w["sa_v_w"], bias=w["sa_v_b"])

        q_t = ttnn.to_torch(q).to(torch.float32)
        k_t = ttnn.to_torch(k).to(torch.float32)
        v_t = ttnn.to_torch(v).to(torch.float32)

        B, L, C = q_t.shape
        q_t = q_t.view(B, L, self.num_heads, self.head_size).transpose(1, 2)  # (B, H, L, d)
        k_t = k_t.view(B, L, self.num_heads, self.head_size).transpose(1, 2)
        v_t = v_t.view(B, L, self.num_heads, self.head_size).transpose(1, 2)

        # linear_pos on rel_pos_emb (no bias)
        proj_rp = rel_pos_emb @ t["lp_w"].T  # (1, 2L-1, C)
        proj_rp = proj_rp.view(rel_pos_emb.size(0), -1, self.num_heads, self.head_size)
        proj_rp = proj_rp.transpose(1, 2).transpose(2, 3)  # (1, H, d, 2L-1)

        # Add biases pos_bias_u/v to Q per head.
        q_tr = q_t.transpose(1, 2)  # (B, L, H, d)
        q_u = (q_tr + t["pos_bias_u"]).transpose(1, 2)  # (B, H, L, d)
        q_v = (q_tr + t["pos_bias_v"]).transpose(1, 2)

        scores_ac = q_u @ k_t.transpose(-2, -1)  # (B, H, L, L)
        scores_bd = q_v @ proj_rp  # (B, H, L, 2L-1)

        # Shift trick (Transformer-XL): pad+reshape to align relative offsets.
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        bd_padded = torch.concat([zero_pad, scores_bd], dim=-1)  # (B, H, L, 2L)
        bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        bd_padded = bd_padded.view(*bd_padded_shape)
        bd_shifted = bd_padded[:, :, 1:].view_as(scores_bd)
        bd_shifted = bd_shifted[:, :, :, : scores_bd.size(-1) // 2 + 1]

        scores = (scores_ac + bd_shifted) / math.sqrt(self.head_size)
        probs = torch.softmax(scores, dim=-1)
        out = probs @ v_t  # (B, H, L, d)
        out = out.transpose(1, 2).reshape(B, L, C).contiguous()

        # linear_out via ttnn on device.
        out_ttnn = _to_ttnn(out.to(torch.bfloat16), self.device)
        out_ttnn = ttnn.linear(out_ttnn, w["sa_o_w"], bias=w["sa_o_b"])
        return out_ttnn

    def _conv_module(self, x_ttnn, w, t):
        # x_ttnn: (B, L, C)
        h = ttnn.layer_norm(x_ttnn, epsilon=self.eps, weight=w["cm_ln_w"], bias=w["cm_ln_b"])
        # pointwise_conv1 as linear: (B, L, C) -> (B, L, 2C)
        h = ttnn.linear(h, w["cm_pc1_w"])
        # GLU + depthwise_conv + batch_norm on host.
        h_t = ttnn.to_torch(h).to(torch.float32)
        h_a, h_b = h_t.chunk(2, dim=-1)
        h_t = h_a * torch.sigmoid(h_b)
        h_t = h_t.transpose(1, 2).contiguous()  # (B, C, L)
        pad = (self.dw_kernel - 1) // 2
        h_t = F.conv1d(h_t, t["dw_conv_weight"], stride=1, padding=pad, groups=self.hidden_size)
        scale = t["bn_weight"] / torch.sqrt(t["bn_running_var"] + t["bn_eps"])
        shift = t["bn_bias"] - t["bn_running_mean"] * scale
        h_t = h_t * scale.view(1, -1, 1) + shift.view(1, -1, 1)
        h_t = h_t.transpose(1, 2).contiguous()

        h = _to_ttnn(h_t.to(torch.bfloat16), self.device)
        h = ttnn.silu(h)
        h = ttnn.linear(h, w["cm_pc2_w"])
        return h

    def _apply_layer(self, i, x_ttnn, rel_pos_emb):
        w = self.layers_w[i]
        t = self.layers_torch[i]

        # ffn1: scaled residual
        residual = x_ttnn
        h = ttnn.layer_norm(x_ttnn, epsilon=self.eps, weight=w["ffn1_ln_w"], bias=w["ffn1_ln_b"])
        h = ttnn.linear(h, w["ffn1_int_w"], bias=w["ffn1_int_b"])
        h = ttnn.silu(h)
        h = ttnn.linear(h, w["ffn1_out_w"], bias=w["ffn1_out_b"])
        h = ttnn.multiply(h, 0.5)
        h = ttnn.add(h, residual)

        # self_attn: residual add
        residual = h
        h_ln = ttnn.layer_norm(h, epsilon=self.eps, weight=w["sa_ln_w"], bias=w["sa_ln_b"])
        h = self._self_attn(h_ln, rel_pos_emb, w, t)
        h = ttnn.add(h, residual)

        # conv_module: residual add
        residual = h
        h = self._conv_module(h, w, t)
        h = ttnn.add(residual, h)

        # ffn2: scaled residual
        residual = h
        h = ttnn.layer_norm(h, epsilon=self.eps, weight=w["ffn2_ln_w"], bias=w["ffn2_ln_b"])
        h = ttnn.linear(h, w["ffn2_int_w"], bias=w["ffn2_int_b"])
        h = ttnn.silu(h)
        h = ttnn.linear(h, w["ffn2_out_w"], bias=w["ffn2_out_b"])
        h = ttnn.multiply(h, 0.5)
        h = ttnn.add(h, residual)

        # final layer norm
        h = ttnn.layer_norm(h, epsilon=self.eps, weight=w["final_ln_w"], bias=w["final_ln_b"])
        return h

    def __call__(self, hidden_states, attention_mask=None, *args, **kwargs):
        if isinstance(hidden_states, torch.Tensor):
            x_torch = hidden_states.to(torch.float32)
        else:
            x_torch = ttnn.to_torch(hidden_states).to(torch.float32)

        # rel_pos_emb depends on the sequence length; compute once and reuse.
        L = x_torch.shape[1]
        rel_pos_emb = self._get_rel_pos_emb(L)

        x_ttnn = _to_ttnn(x_torch.to(torch.bfloat16), self.device)

        for i in range(self.num_layers):
            x_ttnn = self._apply_layer(i, x_ttnn, rel_pos_emb)

        # Top-level layer norm.
        x_ttnn = ttnn.layer_norm(x_ttnn, epsilon=self.eps, weight=self.w_top_ln_w, bias=self.w_top_ln_b)
        return x_ttnn


def build(device, torch_module):
    return SeamlessM4TConformerEncoder(device, torch_module)


_instance = None


def seamless_m4_t_conformer_encoder(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `seamless_m4_t_conformer_encoder`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
