# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level partial TTNN port for `voxtral_encoder_layer` of mistralai/Voxtral-Mini-3B-2507.

Generated deterministically by `tt_hw_planner op-synth`.
Weight loading and op-REUSE/op-ADAPT helpers below are
machine-emitted from the HF reference and DO NOT need LLM
review. The `__call__` implementation falls back to HF
torch so the bring-up smoke test passes immediately;
the LLM's only remaining job is to rewrite `__call__`
to call the pre-bound `_apply_*` helpers in the right
order and fill any op-NEW gaps inline.

Pre-bound deterministic helpers (op palette):
#   self._apply_self_attn_k_proj(x) -> ttnn.linear  (in=1280, out=1280, bias=False)
#   self._apply_self_attn_v_proj(x) -> ttnn.linear  (in=1280, out=1280, bias=True)
#   self._apply_self_attn_q_proj(x) -> ttnn.linear  (in=1280, out=1280, bias=True)
#   self._apply_self_attn_out_proj(x) -> ttnn.linear  (in=1280, out=1280, bias=True)
#   self._apply_self_attn_layer_norm(x) -> ttnn.layer_norm  (shape=[1280], eps=1e-05)
#   self._apply_activation_fn(x) -> ttnn.gelu
#   self._apply_fc1(x) -> ttnn.linear  (in=1280, out=5120, bias=True)
#   self._apply_fc2(x) -> ttnn.linear  (in=5120, out=1280, bias=True)
#   self._apply_final_layer_norm(x) -> ttnn.layer_norm  (shape=[1280], eps=1e-05)

LLM_GAPs (op-NEW — still need synthesis):
#   (none — fully deterministic, no LLM_GAPs)

HF reference: transformers/src/transformers/models/voxtral/modeling_voxtral.py
Op counts: total=9  op-REUSE=9  op-ADAPT=0  op-NEW=0"""
from __future__ import annotations

import transformers

import ttnn

HF_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
_CANDIDATE_SUBMODULE_PATHS = ["audio_tower.layers.0"]


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


_LLM_GAPS = []


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


class VoxtralEncoderLayer:
    def __init__(self, device, torch_module):
        self.device = device
        self._torch_module = torch_module
        sd = torch_module.state_dict()
        # op-REUSE: self_attn.k_proj  (Linear 1280 -> 1280, bias=False)
        self.w_self_attn_k_proj_weight = ttnn.from_torch(
            sd["self_attn.k_proj.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: self_attn.v_proj  (Linear 1280 -> 1280, bias=True)
        self.w_self_attn_v_proj_weight = ttnn.from_torch(
            sd["self_attn.v_proj.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_self_attn_v_proj_bias = ttnn.from_torch(
            sd["self_attn.v_proj.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: self_attn.q_proj  (Linear 1280 -> 1280, bias=True)
        self.w_self_attn_q_proj_weight = ttnn.from_torch(
            sd["self_attn.q_proj.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_self_attn_q_proj_bias = ttnn.from_torch(
            sd["self_attn.q_proj.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: self_attn.out_proj  (Linear 1280 -> 1280, bias=True)
        self.w_self_attn_out_proj_weight = ttnn.from_torch(
            sd["self_attn.out_proj.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_self_attn_out_proj_bias = ttnn.from_torch(
            sd["self_attn.out_proj.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: self_attn_layer_norm  (LayerNorm [1280], eps=1e-05)
        self.w_self_attn_layer_norm_weight = ttnn.from_torch(
            sd["self_attn_layer_norm.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_self_attn_layer_norm_bias = ttnn.from_torch(
            sd["self_attn_layer_norm.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._eps_self_attn_layer_norm = 1e-05

        # op-REUSE: activation_fn  (GELU)

        # op-REUSE: fc1  (Linear 1280 -> 5120, bias=True)
        self.w_fc1_weight = ttnn.from_torch(
            sd["fc1.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_fc1_bias = ttnn.from_torch(
            sd["fc1.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: fc2  (Linear 5120 -> 1280, bias=True)
        self.w_fc2_weight = ttnn.from_torch(
            sd["fc2.weight"].T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_fc2_bias = ttnn.from_torch(
            sd["fc2.bias"].reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # op-REUSE: final_layer_norm  (LayerNorm [1280], eps=1e-05)
        self.w_final_layer_norm_weight = ttnn.from_torch(
            sd["final_layer_norm.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.w_final_layer_norm_bias = ttnn.from_torch(
            sd["final_layer_norm.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._eps_final_layer_norm = 1e-05

        attn = torch_module.self_attn
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _to_heads(self, x):
        t = int(x.shape[1])
        x = ttnn.reshape(x, (1, t, self.num_heads, self.head_dim))
        return ttnn.transpose(x, 1, 2)

    def _apply_self_attn_k_proj(self, x):
        return ttnn.linear(x, self.w_self_attn_k_proj_weight)

    def _apply_self_attn_v_proj(self, x):
        return ttnn.linear(x, self.w_self_attn_v_proj_weight, bias=self.w_self_attn_v_proj_bias)

    def _apply_self_attn_q_proj(self, x):
        return ttnn.linear(x, self.w_self_attn_q_proj_weight, bias=self.w_self_attn_q_proj_bias)

    def _apply_self_attn_out_proj(self, x):
        return ttnn.linear(x, self.w_self_attn_out_proj_weight, bias=self.w_self_attn_out_proj_bias)

    def _apply_self_attn_layer_norm(self, x):
        return ttnn.layer_norm(
            x,
            epsilon=self._eps_self_attn_layer_norm,
            weight=self.w_self_attn_layer_norm_weight,
            bias=self.w_self_attn_layer_norm_bias,
        )

    def _apply_activation_fn(self, x):
        return ttnn.gelu(x)

    def _apply_fc1(self, x):
        return ttnn.linear(x, self.w_fc1_weight, bias=self.w_fc1_bias)

    def _apply_fc2(self, x):
        return ttnn.linear(x, self.w_fc2_weight, bias=self.w_fc2_bias)

    def _apply_final_layer_norm(self, x):
        return ttnn.layer_norm(
            x,
            epsilon=self._eps_final_layer_norm,
            weight=self.w_final_layer_norm_weight,
            bias=self.w_final_layer_norm_bias,
        )

    def __call__(self, *args, **kwargs):
        # VoxtralEncoderLayer: pre-LN self-attn (bidirectional, no mask) + FFN.
        # The harness's all-ones attention_mask is dropped (encoder attention is
        # unmasked in the real forward). Scaling is applied to q (attn scale 1).
        x = args[0] if args else kwargs["hidden_states"]
        residual = x
        h = self._apply_self_attn_layer_norm(x)
        q = ttnn.mul(self._apply_self_attn_q_proj(h), self.scaling)
        k = self._apply_self_attn_k_proj(h)
        v = self._apply_self_attn_v_proj(h)
        q = self._to_heads(q)
        k = self._to_heads(k)
        v = self._to_heads(v)
        scores = ttnn.matmul(q, ttnn.transpose(k, 2, 3), compute_kernel_config=self.compute_kernel_config)
        scores = ttnn.softmax(scores, dim=-1)
        attn = ttnn.matmul(scores, v, compute_kernel_config=self.compute_kernel_config)
        attn = ttnn.transpose(attn, 1, 2)
        attn = ttnn.reshape(attn, (1, attn.shape[1], self.num_heads * self.head_dim))
        attn = self._apply_self_attn_out_proj(attn)
        x = ttnn.add(residual, attn)

        residual = x
        h = self._apply_final_layer_norm(x)
        h = ttnn.gelu(self._apply_fc1(h))
        h = self._apply_fc2(h)
        return ttnn.add(residual, h)


def build(device, torch_module):
    return VoxtralEncoderLayer(device, torch_module)


_instance = None


def voxtral_encoder_layer(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `voxtral_encoder_layer`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
