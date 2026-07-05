# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level partial TTNN port for `seamless_m4_t_scaled_word_embedding` of facebook/hf-seamless-m4t-large.

Generated deterministically by `tt_hw_planner op-synth`.
Weight loading and op-REUSE/op-ADAPT helpers below are
machine-emitted from the HF reference and DO NOT need LLM
review. The `__call__` implementation falls back to HF
torch so the bring-up smoke test passes immediately;
the LLM's only remaining job is to rewrite `__call__`
to call the pre-bound `_apply_*` helpers in the right
order and fill any op-NEW gaps inline.

Pre-bound deterministic helpers (op palette):
#   (none — every op was op-NEW)

LLM_GAPs (op-NEW — still need synthesis):
#   - <root>  (SeamlessM4TScaledWordEmbedding)

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
Op counts: total=1  op-REUSE=0  op-ADAPT=0  op-NEW=1"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["text_encoder.embed_tokens"]


def _log_runtime_fallback(helper, kind, reason):
    """Append a structured CPU-fallback event for the planner reporter.

    Best-effort; never raises and never blocks the test. Writes to
    `<demo_dir>/_runtime_fallbacks.jsonl` (or the path in env
    TT_HW_PLANNER_RUNTIME_FALLBACK_LOG). The planner truncates this file
    before each pytest invocation and consumes it afterwards.
    """
    try:
        import json as _json
        import os as _os
        import sys as _sys
        import time as _time
        from pathlib import Path as _Path

        _sys.stderr.write("[%s_CPU_FALLBACK] %s: %s\n" % (kind.upper(), helper, reason))
        log_env = _os.environ.get("TT_HW_PLANNER_RUNTIME_FALLBACK_LOG")
        if log_env:
            log_path = _Path(log_env)
        else:
            # _stubs/<safe>.py  ->  demo_dir = parents[1]
            log_path = _Path(__file__).resolve().parents[1] / "_runtime_fallbacks.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ev = {
            "component": _Path(__file__).stem,
            "helper": helper,
            "kind": kind,
            "reason": reason,
            "ts": _time.time(),
        }
        with log_path.open("a") as f:
            f.write(_json.dumps(ev) + "\n")
    except Exception:
        pass


_LLM_GAPS = [
    {"name": "<root>", "class": "SeamlessM4TScaledWordEmbedding", "notes": ""},
]


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


def _coerce_to_torch(x):
    try:
        import ttnn as _ttnn

        if isinstance(x, _ttnn.Tensor):
            import torch as _torch

            t = _ttnn.to_torch(x)
            # Bug Y fix (2026-05-23 live-run sam2-hiera-tiny)
            if t.is_floating_point():
                if t.dtype != _torch.float32:
                    t = t.to(_torch.float32)
            elif t.dtype != _torch.bool:
                t = t.to(_torch.long)
            return t
    except Exception:
        pass
    if isinstance(x, tuple):
        return tuple(_coerce_to_torch(e) for e in x)
    if isinstance(x, list):
        return [_coerce_to_torch(e) for e in x]
    if isinstance(x, dict):
        return {k: _coerce_to_torch(v) for k, v in x.items()}
    return x


class SeamlessM4TScaledWordEmbedding:
    def __init__(self, device, torch_module):
        self.device = device
        self._torch_module = torch_module
        sd = torch_module.state_dict()
        self.w = ttnn.from_torch(
            sd["weight"].to(torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.embed_scale = float(getattr(torch_module, "embed_scale", 1.0))

    def __call__(self, input_ids, *args, **kwargs):
        # ScaledWordEmbedding = nn.Embedding(input_ids) * embed_scale.
        out = ttnn.embedding(input_ids, self.w)
        if self.embed_scale != 1.0:
            out = ttnn.multiply(out, self.embed_scale)
        return out


def build(device, torch_module):
    return SeamlessM4TScaledWordEmbedding(device, torch_module)


_instance = None


def seamless_m4_t_scaled_word_embedding(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `seamless_m4_t_scaled_word_embedding`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
