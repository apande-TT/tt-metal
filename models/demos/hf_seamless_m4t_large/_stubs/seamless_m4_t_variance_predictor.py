# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level partial TTNN port for `seamless_m4_t_variance_predictor` of facebook/hf-seamless-m4t-large.

Generated deterministically by `tt_hw_planner op-synth`.
Weight loading and op-REUSE/op-ADAPT helpers below are
machine-emitted from the HF reference and DO NOT need LLM
review. The `__call__` implementation falls back to HF
torch so the bring-up smoke test passes immediately;
the LLM's only remaining job is to rewrite `__call__`
to call the pre-bound `_apply_*` helpers in the right
order and fill any op-NEW gaps inline.

Pre-bound deterministic helpers (op palette):
#   self._apply_activation_function(x) -> ttnn.relu
#   self._apply_ln1(x) -> ttnn.layer_norm  (shape=[1280], eps=1e-05)
#   self._apply_ln2(x) -> ttnn.layer_norm  (shape=[1280], eps=1e-05)
#   self._apply_proj(x) -> ttnn.linear  (in=1280, out=1, bias=True)

LLM_GAPs (op-NEW — still need synthesis):
#   - conv1  (Conv1d)
#   - dropout_module  (Dropout)
#   - conv2  (Conv1d)

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
Op counts: total=7  op-REUSE=5  op-ADAPT=2  op-NEW=0"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["vocoder.dur_predictor"]


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
    {"name": "conv1", "class": "Conv1d", "notes": ""},
    {"name": "dropout_module", "class": "Dropout", "notes": ""},
    {"name": "conv2", "class": "Conv1d", "notes": ""},
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


def _extract_conv1d_cfg(conv):
    w = ttnn.from_torch(conv.weight.detach().to(torch.float32), dtype=ttnn.bfloat16)
    b = None
    if conv.bias is not None:
        b_t = conv.bias.detach().to(torch.float32).reshape(1, 1, 1, -1)
        b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16)
    return {
        "w": w,
        "b": b,
        "in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
        "kernel_size": int(conv.kernel_size[0]),
        "stride": int(conv.stride[0]),
        "padding": int(conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding),
        "dilation": int(conv.dilation[0]),
        "groups": int(conv.groups),
    }


class SeamlessM4TVariancePredictor:
    def __init__(self, device, torch_module):
        self.device = device
        self._torch_module = torch_module
        sd = torch_module.state_dict()
        # op-REUSE: activation_function  (RELU)

        # op-REUSE: ln1  (LayerNorm [1280], eps=1e-05)
        self.w_ln1_weight = ttnn.from_torch(
            sd["ln1.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_ln1_bias = ttnn.from_torch(sd["ln1.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._eps_ln1 = 1e-05

        # op-REUSE: ln2  (LayerNorm [1280], eps=1e-05)
        self.w_ln2_weight = ttnn.from_torch(
            sd["ln2.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_ln2_bias = ttnn.from_torch(sd["ln2.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._eps_ln2 = 1e-05

        # op-REUSE: proj  (Linear 1280 -> 1, bias=True)
        self.w_proj_weight = ttnn.from_torch(
            sd["proj.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_proj_bias = ttnn.from_torch(
            sd["proj.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.conv1_cfg = _extract_conv1d_cfg(torch_module.conv1)
        self.conv2_cfg = _extract_conv1d_cfg(torch_module.conv2)

    def _apply_activation_function(self, x):
        return ttnn.relu(x)

    def _apply_ln1(self, x):
        return ttnn.layer_norm(x, epsilon=self._eps_ln1, weight=self.w_ln1_weight, bias=self.w_ln1_bias)

    def _apply_ln2(self, x):
        return ttnn.layer_norm(x, epsilon=self._eps_ln2, weight=self.w_ln2_weight, bias=self.w_ln2_bias)

    def _apply_proj(self, x):
        return ttnn.linear(x, self.w_proj_weight, bias=self.w_proj_bias)

    def _apply_conv(self, x_nlc, cfg, input_length):
        # x_nlc: ttnn ROW_MAJOR tensor shape (1, L, C). ttnn.conv1d takes
        # NLC directly (equivalent to torch's Conv1d on transposed NCL
        # input), so no permute needed relative to (B, T, C).
        out = ttnn.conv1d(
            input_tensor=x_nlc,
            weight_tensor=cfg["w"],
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            device=self.device,
            bias_tensor=cfg["b"],
            kernel_size=cfg["kernel_size"],
            stride=cfg["stride"],
            padding=cfg["padding"],
            dilation=cfg["dilation"],
            groups=cfg["groups"],
            batch_size=1,
            input_length=input_length,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.reshape(out, (1, input_length, cfg["out_channels"]))
        return out

    def __call__(self, hidden_states, *args, **kwargs):
        # forward(hidden_states): (B, T, C) ->
        #   x = conv1(x.transpose(1,2)); x = relu(x).transpose(1,2)
        #   x = ln1(x)  (dropout no-op in eval)
        #   x = conv2(x.transpose(1,2)); x = relu(x).transpose(1,2)
        #   x = ln2(x); return proj(x).squeeze(dim=2)  -> (B, T)
        # ttnn.conv1d works on NLC directly, so no transposes needed on
        # the ttnn side. Do LN+proj in TILE layout; convs need ROW_MAJOR.
        L = int(hidden_states.shape[-2])

        x = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        x = self._apply_conv(x, self.conv1_cfg, L)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self._apply_activation_function(x)
        x = self._apply_ln1(x)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = self._apply_conv(x, self.conv2_cfg, L)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = self._apply_activation_function(x)
        x = self._apply_ln2(x)

        x = self._apply_proj(x)  # (1, T, 1)
        # squeeze the last dim to match torch reference output (1, T)
        return ttnn.squeeze(x, -1)


def build(device, torch_module):
    return SeamlessM4TVariancePredictor(device, torch_module)


_instance = None


def seamless_m4_t_variance_predictor(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `seamless_m4_t_variance_predictor`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
