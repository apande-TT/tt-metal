# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level partial TTNN port for `seamless_m4_t_feed_forward_network` of facebook/hf-seamless-m4t-large.

Generated deterministically by `tt_hw_planner op-synth`.
Weight loading and op-REUSE/op-ADAPT helpers below are
machine-emitted from the HF reference and DO NOT need LLM
review. The `__call__` implementation falls back to HF
torch so the bring-up smoke test passes immediately;
the LLM's only remaining job is to rewrite `__call__`
to call the pre-bound `_apply_*` helpers in the right
order and fill any op-NEW gaps inline.

Pre-bound deterministic helpers (op palette):
#   self._apply_fc1(x) -> ttnn.linear  (in=1024, out=8192, bias=True)
#   self._apply_fc2(x) -> ttnn.linear  (in=8192, out=1024, bias=True)
#   self._apply_act(x) -> ttnn.relu

LLM_GAPs (op-NEW — still need synthesis):
#   - dropout  (Dropout)

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
Op counts: total=4  op-REUSE=4  op-ADAPT=0  op-NEW=0"""
from __future__ import annotations

import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["text_encoder.layers.0.ffn"]


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
    {"name": "dropout", "class": "Dropout", "notes": ""},
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


class SeamlessM4TFeedForwardNetwork:
    def __init__(self, device, torch_module):
        self.device = device
        self._torch_module = torch_module
        sd = torch_module.state_dict()
        # op-REUSE: fc1  (Linear 1024 -> 8192, bias=True)
        self.w_fc1_weight = ttnn.from_torch(
            sd["fc1.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_fc1_bias = ttnn.from_torch(
            sd["fc1.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: fc2  (Linear 8192 -> 1024, bias=True)
        self.w_fc2_weight = ttnn.from_torch(
            sd["fc2.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_fc2_bias = ttnn.from_torch(
            sd["fc2.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: act  (RELU)

    def _apply_fc1(self, x):
        return ttnn.linear(x, self.w_fc1_weight, bias=self.w_fc1_bias)

    def _apply_fc2(self, x):
        return ttnn.linear(x, self.w_fc2_weight, bias=self.w_fc2_bias)

    def _apply_act(self, x):
        return ttnn.relu(x)

    def __call__(self, hidden_states, *args, **kwargs):
        # FFN block: fc1 -> act (relu) -> [dropout] -> fc2 -> [dropout].
        # Dropouts are no-ops in eval mode.
        x = self._apply_fc1(hidden_states)
        x = self._apply_act(x)
        x = self._apply_fc2(x)
        return x


def build(device, torch_module):
    return SeamlessM4TFeedForwardNetwork(device, torch_module)


_instance = None


def seamless_m4_t_feed_forward_network(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `seamless_m4_t_feed_forward_network`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
