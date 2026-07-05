# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `speech_encoder_intermediate_ffn` of facebook/hf-seamless-m4t-large.

SeamlessM4TConformerFeedForward (act_fn='swish' -> SiLU): Linear(1024 -> 4096) ->
SiLU -> Linear(4096 -> 1024). Dropouts are no-ops in eval mode.
"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["speech_encoder.intermediate_ffn"]


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


class SpeechEncoderIntermediateFfn:
    def __init__(self, device, torch_module):
        self.device = device
        sd = torch_module.state_dict()
        self.w_int_w = ttnn.from_torch(
            sd["intermediate_dense.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_int_b = ttnn.from_torch(
            sd["intermediate_dense.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_out_w = ttnn.from_torch(
            sd["output_dense.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_out_b = ttnn.from_torch(
            sd["output_dense.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, torch.Tensor):
            x = ttnn.from_torch(
                hidden_states.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        else:
            x = hidden_states
        x = ttnn.linear(x, self.w_int_w, bias=self.w_int_b)
        x = ttnn.silu(x)
        x = ttnn.linear(x, self.w_out_w, bias=self.w_out_b)
        return x


def build(device, torch_module):
    return SpeechEncoderIntermediateFfn(device, torch_module)


_instance = None


def speech_encoder_intermediate_ffn(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `speech_encoder_intermediate_ffn`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
