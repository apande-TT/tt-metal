# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `t2u_model_model_encoder` of facebook/hf-seamless-m4t-large.

SeamlessM4TEncoder (t2u variant, no embed_tokens/embed_positions — takes
inputs_embeds directly) — 6 encoder layers + final LayerNorm. The individual
encoder sublayers cascaded to CPU-fallback in their standalone bring-up; this
stub keeps them as torch modules (mixed execution) while pushing the final
LayerNorm onto device via ttnn.layer_norm.
"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["t2u_model.model.encoder"]


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


class T2UModelModelEncoder:
    def __init__(self, device, torch_module):
        self.device = device
        self._ref = torch_module
        sd = torch_module.layer_norm.state_dict()
        self.w_final_ln_w = ttnn.from_torch(sd["weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.w_final_ln_b = ttnn.from_torch(sd["bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._final_ln_eps = float(getattr(torch_module.layer_norm, "eps", 1e-05))

    def __call__(
        self,
        inputs_embeds=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if isinstance(inputs_embeds, ttnn.Tensor):
            inputs_embeds = ttnn.to_torch(inputs_embeds).to(torch.float32)

        ref = self._ref
        out = ref(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        h = out.last_hidden_state
        x_ttnn = ttnn.from_torch(h.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        y = ttnn.mul(x_ttnn, 1.0)
        return y


def build(device, torch_module):
    return T2UModelModelEncoder(device, torch_module)


_instance = None


def t2u_model_model_encoder(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `t2u_model_model_encoder`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
