# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level partial TTNN port for `avg_pool1d` of mistralai/Voxtral-Mini-3B-2507.

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
#   - <root>  (AvgPool1d)

HF reference: transformers/src/transformers/models/voxtral/modeling_voxtral.py
Op counts: total=1  op-REUSE=0  op-ADAPT=0  op-NEW=1"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
_CANDIDATE_SUBMODULE_PATHS = ["audio_tower.avg_pooler"]


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
    {"name": "<root>", "class": "AvgPool1d", "notes": ""},
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


def _as_1tuple(v):
    """nn.AvgPool1d stores kernel_size/stride/padding as int or 1-tuple."""
    if isinstance(v, (tuple, list)):
        return int(v[0])
    return int(v)


class AvgPool1d:
    """Native ttnn port of `nn.AvgPool1d(kernel_size, stride, padding)`.

    Input is `(N, C, L)`; pooling is over the last (time) axis. We realize the
    fixed-weight sliding-window average as a single matmul against a constant
    pooling matrix `P` of shape `(L, L_out)` where, for output column `o`, the
    `kernel_size` input rows in that window carry weight `1/kernel_size` and the
    rest are zero. `out = input @ P` then averages each window with no offset
    tilization or row-major reshuffling. `P` is built lazily per input length
    and cached (the audio pooler always sees the same `L`).
    """

    def __init__(self, device, torch_module):
        self.device = device
        self._torch_module = torch_module
        self.kernel = _as_1tuple(torch_module.kernel_size)
        # nn.AvgPool1d defaults stride to kernel_size when stride is None.
        stride = getattr(torch_module, "stride", None)
        self.stride = self.kernel if stride is None else _as_1tuple(stride)
        self.padding = _as_1tuple(getattr(torch_module, "padding", 0))
        self._pool_mats = {}

    def _pool_matrix(self, L):
        mat = self._pool_mats.get(L)
        if mat is not None:
            return mat
        L_out = (L + 2 * self.padding - self.kernel) // self.stride + 1
        # Build the constant pooling matrix on host, then move to device once.
        P = torch.zeros(L, L_out, dtype=torch.float32)
        w = 1.0 / float(self.kernel)
        for o in range(L_out):
            start = o * self.stride - self.padding
            for k in range(self.kernel):
                idx = start + k
                if 0 <= idx < L:
                    P[idx, o] = w
        mat = ttnn.from_torch(P, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._pool_mats[L] = mat
        return mat

    def __call__(self, *args, **kwargs):
        x = args[0] if args else kwargs.get("input")
        # The PCC harness always hands us a ttnn tensor; guard the rare host path.
        if not isinstance(x, ttnn.Tensor):
            x = _ttnn_from_torch(x, self.device)
        L = int(x.shape[-1])
        P = self._pool_matrix(L)
        return ttnn.matmul(x, P)


def _ttnn_from_torch(t, device):
    return ttnn.from_torch(t.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def build(device, torch_module):
    return AvgPool1d(device, torch_module)


_instance = None


def avg_pool1d(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `avg_pool1d`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
