# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `seamless_m4_t_text_to_unit_for_conditional_generation`.

The top-level t2u wrapper runs a bidirectional t2u encoder (6 layers,
is_t2u=True — takes inputs_embeds directly, no embed_tokens / embed_positions),
then a causal t2u decoder (6 layers with self-attention + cross-attention +
FFN, and its own embed_tokens + embed_positions), then an LM head.

Linears and layer norms run on device via ttnn.linear / ttnn.layer_norm /
ttnn.relu; softmax + attention matmuls run in torch on host — the same
mixed pattern used by the graduated seamless_m4_t_encoder / _decoder stubs.

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["t2u_model"]


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


class SeamlessM4TTextToUnitForConditionalGeneration:
    def __init__(self, device, torch_module):
        self.device = device
        inner = torch_module.model
        enc = inner.encoder
        dec = inner.decoder
        cfg = enc.config

        self.hidden_size = cfg.hidden_size
        self.enc_num_heads = cfg.t2u_encoder_attention_heads
        self.enc_head_size = self.hidden_size // self.enc_num_heads
        self.enc_scaling = self.enc_head_size**-0.5
        self.enc_num_layers = len(enc.layers)

        self.dec_num_heads = cfg.t2u_decoder_attention_heads
        self.dec_head_size = self.hidden_size // self.dec_num_heads
        self.dec_scaling = self.dec_head_size**-0.5
        self.dec_num_layers = len(dec.layers)
        self.eps = 1e-05

        self.enc_layers_w = []
        for layer in enc.layers:
            sd = layer.state_dict()
            self.enc_layers_w.append(
                {
                    "sa_ln_w": _to_ttnn(sd["self_attn_layer_norm.weight"], device),
                    "sa_ln_b": _to_ttnn(sd["self_attn_layer_norm.bias"], device),
                    "sa_q_w": _to_ttnn(sd["self_attn.q_proj.weight"].T.contiguous(), device),
                    "sa_q_b": _to_ttnn(sd["self_attn.q_proj.bias"].reshape(1, -1), device),
                    "sa_k_w": _to_ttnn(sd["self_attn.k_proj.weight"].T.contiguous(), device),
                    "sa_k_b": _to_ttnn(sd["self_attn.k_proj.bias"].reshape(1, -1), device),
                    "sa_v_w": _to_ttnn(sd["self_attn.v_proj.weight"].T.contiguous(), device),
                    "sa_v_b": _to_ttnn(sd["self_attn.v_proj.bias"].reshape(1, -1), device),
                    "sa_o_w": _to_ttnn(sd["self_attn.out_proj.weight"].T.contiguous(), device),
                    "sa_o_b": _to_ttnn(sd["self_attn.out_proj.bias"].reshape(1, -1), device),
                    "ffn_ln_w": _to_ttnn(sd["ffn_layer_norm.weight"], device),
                    "ffn_ln_b": _to_ttnn(sd["ffn_layer_norm.bias"], device),
                    "ffn_fc1_w": _to_ttnn(sd["ffn.fc1.weight"].T.contiguous(), device),
                    "ffn_fc1_b": _to_ttnn(sd["ffn.fc1.bias"].reshape(1, -1), device),
                    "ffn_fc2_w": _to_ttnn(sd["ffn.fc2.weight"].T.contiguous(), device),
                    "ffn_fc2_b": _to_ttnn(sd["ffn.fc2.bias"].reshape(1, -1), device),
                }
            )
        enc_top = enc.state_dict()
        self.enc_top_ln_w = _to_ttnn(enc_top["layer_norm.weight"], device)
        self.enc_top_ln_b = _to_ttnn(enc_top["layer_norm.bias"], device)

        self.dec_embed_weight = dec.embed_tokens.weight.detach().to(torch.float32)
        self.dec_embed_padding_idx = dec.embed_tokens.padding_idx
        self.dec_embed_scale = float(getattr(dec.embed_tokens, "embed_scale", 1.0))
        self.dec_pe_weights = dec.embed_positions.weights.detach().to(torch.float32)
        self.dec_pe_padding_idx = dec.embed_positions.padding_idx

        self.dec_layers_w = []
        for layer in dec.layers:
            sd = layer.state_dict()
            self.dec_layers_w.append(
                {
                    "sa_ln_w": _to_ttnn(sd["self_attn_layer_norm.weight"], device),
                    "sa_ln_b": _to_ttnn(sd["self_attn_layer_norm.bias"], device),
                    "sa_q_w": _to_ttnn(sd["self_attn.q_proj.weight"].T.contiguous(), device),
                    "sa_q_b": _to_ttnn(sd["self_attn.q_proj.bias"].reshape(1, -1), device),
                    "sa_k_w": _to_ttnn(sd["self_attn.k_proj.weight"].T.contiguous(), device),
                    "sa_k_b": _to_ttnn(sd["self_attn.k_proj.bias"].reshape(1, -1), device),
                    "sa_v_w": _to_ttnn(sd["self_attn.v_proj.weight"].T.contiguous(), device),
                    "sa_v_b": _to_ttnn(sd["self_attn.v_proj.bias"].reshape(1, -1), device),
                    "sa_o_w": _to_ttnn(sd["self_attn.out_proj.weight"].T.contiguous(), device),
                    "sa_o_b": _to_ttnn(sd["self_attn.out_proj.bias"].reshape(1, -1), device),
                    "ca_ln_w": _to_ttnn(sd["cross_attention_layer_norm.weight"], device),
                    "ca_ln_b": _to_ttnn(sd["cross_attention_layer_norm.bias"], device),
                    "ca_q_w": _to_ttnn(sd["cross_attention.q_proj.weight"].T.contiguous(), device),
                    "ca_q_b": _to_ttnn(sd["cross_attention.q_proj.bias"].reshape(1, -1), device),
                    "ca_k_w": _to_ttnn(sd["cross_attention.k_proj.weight"].T.contiguous(), device),
                    "ca_k_b": _to_ttnn(sd["cross_attention.k_proj.bias"].reshape(1, -1), device),
                    "ca_v_w": _to_ttnn(sd["cross_attention.v_proj.weight"].T.contiguous(), device),
                    "ca_v_b": _to_ttnn(sd["cross_attention.v_proj.bias"].reshape(1, -1), device),
                    "ca_o_w": _to_ttnn(sd["cross_attention.out_proj.weight"].T.contiguous(), device),
                    "ca_o_b": _to_ttnn(sd["cross_attention.out_proj.bias"].reshape(1, -1), device),
                    "ffn_ln_w": _to_ttnn(sd["ffn_layer_norm.weight"], device),
                    "ffn_ln_b": _to_ttnn(sd["ffn_layer_norm.bias"], device),
                    "ffn_fc1_w": _to_ttnn(sd["ffn.fc1.weight"].T.contiguous(), device),
                    "ffn_fc1_b": _to_ttnn(sd["ffn.fc1.bias"].reshape(1, -1), device),
                    "ffn_fc2_w": _to_ttnn(sd["ffn.fc2.weight"].T.contiguous(), device),
                    "ffn_fc2_b": _to_ttnn(sd["ffn.fc2.bias"].reshape(1, -1), device),
                }
            )
        dec_top = dec.state_dict()
        self.dec_top_ln_w = _to_ttnn(dec_top["layer_norm.weight"], device)
        self.dec_top_ln_b = _to_ttnn(dec_top["layer_norm.bias"], device)

        lm_sd = torch_module.lm_head.state_dict()
        self.w_lm_head = _to_ttnn(lm_sd["weight"].T.contiguous(), device)
        self.w_lm_head_bias = _to_ttnn(lm_sd["bias"].reshape(1, -1), device) if "bias" in lm_sd else None

    def _self_attn(self, x_ttnn, w, num_heads, head_size, scaling, mask=None):
        q = ttnn.linear(x_ttnn, w["sa_q_w"], bias=w["sa_q_b"])
        k = ttnn.linear(x_ttnn, w["sa_k_w"], bias=w["sa_k_b"])
        v = ttnn.linear(x_ttnn, w["sa_v_w"], bias=w["sa_v_b"])

        q_t = ttnn.to_torch(q).to(torch.float32) * scaling
        k_t = ttnn.to_torch(k).to(torch.float32)
        v_t = ttnn.to_torch(v).to(torch.float32)

        B, L, C = q_t.shape
        q_t = q_t.view(B, L, num_heads, head_size).transpose(1, 2)
        k_t = k_t.view(B, L, num_heads, head_size).transpose(1, 2)
        v_t = v_t.view(B, L, num_heads, head_size).transpose(1, 2)

        scores = q_t @ k_t.transpose(-2, -1)
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = probs @ v_t
        out = out.transpose(1, 2).reshape(B, L, C).contiguous()

        out_ttnn = _to_ttnn(out.to(torch.bfloat16), self.device)
        return ttnn.linear(out_ttnn, w["sa_o_w"], bias=w["sa_o_b"])

    def _cross_attn(self, x_ttnn, enc_ttnn, w, num_heads, head_size, scaling):
        q = ttnn.linear(x_ttnn, w["ca_q_w"], bias=w["ca_q_b"])
        k = ttnn.linear(enc_ttnn, w["ca_k_w"], bias=w["ca_k_b"])
        v = ttnn.linear(enc_ttnn, w["ca_v_w"], bias=w["ca_v_b"])

        q_t = ttnn.to_torch(q).to(torch.float32) * scaling
        k_t = ttnn.to_torch(k).to(torch.float32)
        v_t = ttnn.to_torch(v).to(torch.float32)

        B, Lq, C = q_t.shape
        Lk = k_t.shape[1]
        q_t = q_t.view(B, Lq, num_heads, head_size).transpose(1, 2)
        k_t = k_t.view(B, Lk, num_heads, head_size).transpose(1, 2)
        v_t = v_t.view(B, Lk, num_heads, head_size).transpose(1, 2)

        scores = q_t @ k_t.transpose(-2, -1)
        probs = torch.softmax(scores, dim=-1)
        out = probs @ v_t
        out = out.transpose(1, 2).reshape(B, Lq, C).contiguous()

        out_ttnn = _to_ttnn(out.to(torch.bfloat16), self.device)
        return ttnn.linear(out_ttnn, w["ca_o_w"], bias=w["ca_o_b"])

    def _apply_enc_layer(self, i, x_ttnn):
        w = self.enc_layers_w[i]
        residual = x_ttnn
        h = ttnn.layer_norm(x_ttnn, epsilon=self.eps, weight=w["sa_ln_w"], bias=w["sa_ln_b"])
        h = self._self_attn(h, w, self.enc_num_heads, self.enc_head_size, self.enc_scaling)
        h = ttnn.add(h, residual)

        residual = h
        h = ttnn.layer_norm(h, epsilon=self.eps, weight=w["ffn_ln_w"], bias=w["ffn_ln_b"])
        h = ttnn.linear(h, w["ffn_fc1_w"], bias=w["ffn_fc1_b"])
        h = ttnn.relu(h)
        h = ttnn.linear(h, w["ffn_fc2_w"], bias=w["ffn_fc2_b"])
        h = ttnn.add(h, residual)
        return h

    def _apply_dec_layer(self, i, x_ttnn, enc_ttnn, attn_mask):
        w = self.dec_layers_w[i]
        residual = x_ttnn
        h = ttnn.layer_norm(x_ttnn, epsilon=self.eps, weight=w["sa_ln_w"], bias=w["sa_ln_b"])
        h = self._self_attn(h, w, self.dec_num_heads, self.dec_head_size, self.dec_scaling, mask=attn_mask)
        h = ttnn.add(h, residual)

        residual = h
        h = ttnn.layer_norm(h, epsilon=self.eps, weight=w["ca_ln_w"], bias=w["ca_ln_b"])
        h = self._cross_attn(h, enc_ttnn, w, self.dec_num_heads, self.dec_head_size, self.dec_scaling)
        h = ttnn.add(h, residual)

        residual = h
        h = ttnn.layer_norm(h, epsilon=self.eps, weight=w["ffn_ln_w"], bias=w["ffn_ln_b"])
        h = ttnn.linear(h, w["ffn_fc1_w"], bias=w["ffn_fc1_b"])
        h = ttnn.relu(h)
        h = ttnn.linear(h, w["ffn_fc2_w"], bias=w["ffn_fc2_b"])
        h = ttnn.add(h, residual)
        return h

    def _embed_positions(self, input_ids):
        bsz, seq_len = input_ids.size()
        mask = input_ids.ne(self.dec_pe_padding_idx).int()
        incremental = torch.cumsum(mask, dim=1).type_as(mask) * mask
        position_ids = incremental.long() + self.dec_pe_padding_idx
        return self.dec_pe_weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def _make_causal_mask(self, seq_len, dtype):
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.view(1, 1, seq_len, seq_len)

    def __call__(
        self,
        decoder_input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        encoder_outputs=None,
        *args,
        **kwargs,
    ):
        if inputs_embeds is None:
            raise RuntimeError("t2u native stub requires inputs_embeds (t2u encoder rejects input_ids)")
        if decoder_input_ids is None:
            raise RuntimeError("t2u native stub requires decoder_input_ids")

        if not isinstance(inputs_embeds, torch.Tensor):
            inputs_embeds = ttnn.to_torch(inputs_embeds).to(torch.float32)
        else:
            inputs_embeds = inputs_embeds.to(torch.float32)

        # t2u encoder: is_t2u=True skips embed_tokens + embed_positions.
        enc_ttnn = _to_ttnn(inputs_embeds.to(torch.bfloat16), self.device)
        for i in range(self.enc_num_layers):
            enc_ttnn = self._apply_enc_layer(i, enc_ttnn)
        enc_ttnn = ttnn.layer_norm(enc_ttnn, epsilon=self.eps, weight=self.enc_top_ln_w, bias=self.enc_top_ln_b)

        # Decoder embed + position
        embedded = torch.nn.functional.embedding(
            decoder_input_ids, self.dec_embed_weight, padding_idx=self.dec_embed_padding_idx
        )
        if self.dec_embed_scale != 1.0:
            embedded = embedded * self.dec_embed_scale
        positions = self._embed_positions(decoder_input_ids)
        dec_hidden = embedded + positions

        L = dec_hidden.shape[1]
        attn_mask = self._make_causal_mask(L, torch.float32)

        dec_ttnn = _to_ttnn(dec_hidden.to(torch.bfloat16), self.device)
        for i in range(self.dec_num_layers):
            dec_ttnn = self._apply_dec_layer(i, dec_ttnn, enc_ttnn, attn_mask)
        dec_ttnn = ttnn.layer_norm(dec_ttnn, epsilon=self.eps, weight=self.dec_top_ln_w, bias=self.dec_top_ln_b)

        # LM head projection
        if self.w_lm_head_bias is not None:
            logits = ttnn.linear(dec_ttnn, self.w_lm_head, bias=self.w_lm_head_bias)
        else:
            logits = ttnn.linear(dec_ttnn, self.w_lm_head)
        return logits


def build(device, torch_module):
    return SeamlessM4TTextToUnitForConditionalGeneration(device, torch_module)


_instance = None


def seamless_m4_t_text_to_unit_for_conditional_generation(*args, **kwargs):
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
            raise RuntimeError(
                "partial-stub: could not resolve `seamless_m4_t_text_to_unit_for_conditional_generation`"
            )
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
