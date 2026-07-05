# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `speech_encoder_encoder` of facebook/hf-seamless-m4t-large.

Implements `SeamlessM4TConformerEncoder.forward` — 24 conformer encoder layers +
a final LayerNorm. The individual encoder sub-layers ran too large / stateful for
the current op palette (their standalone bring-up cascaded to CPU); this stub
runs the sub-layers via the reference torch modules (mixed execution — matches
the fallback plan) while doing input/output boundary and final LayerNorm on
device with ttnn.
"""
from __future__ import annotations

import torch
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


class SpeechEncoderEncoder:
    def __init__(self, device, torch_module):
        self.device = device
        self._embed_positions = torch_module.embed_positions
        self._layers = [layer for layer in torch_module.layers]

        sd = torch_module.layer_norm.state_dict()
        self.w_final_ln_w = ttnn.from_torch(sd["weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.w_final_ln_b = ttnn.from_torch(sd["bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._final_ln_eps = float(getattr(torch_module.layer_norm, "eps", 1e-05))

    def __call__(self, hidden_states, attention_mask=None, *args, **kwargs):
        if isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        rel_pos = None
        if self._embed_positions is not None:
            rel_pos = self._embed_positions(hidden_states)

        for layer in self._layers:
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                relative_position_embeddings=rel_pos,
                output_attentions=False,
                conv_attention_mask=conv_attention_mask,
            )
            hidden_states = layer_out[0]

        # Final LayerNorm on device
        x_ttnn = ttnn.from_torch(
            hidden_states.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        y = ttnn.layer_norm(x_ttnn, epsilon=self._final_ln_eps, weight=self.w_final_ln_w, bias=self.w_final_ln_b)
        return y


def build(device, torch_module):
    return SpeechEncoderEncoder(device, torch_module)


_instance = None


def speech_encoder_encoder(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `speech_encoder_encoder`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
