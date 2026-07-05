# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level partial TTNN port for `hifi_gan_residual_block` of facebook/hf-seamless-m4t-large.

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
#   - convs1.0  (Conv1d)
#   - convs1.1  (Conv1d)
#   - convs1.2  (Conv1d)
#   - convs2.0  (Conv1d)
#   - convs2.1  (Conv1d)
#   - convs2.2  (Conv1d)

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
Op counts: total=6  op-REUSE=0  op-ADAPT=6  op-NEW=0"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["vocoder.hifi_gan.resblocks.0"]


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
    {"name": "convs1.0", "class": "Conv1d", "notes": ""},
    {"name": "convs1.1", "class": "Conv1d", "notes": ""},
    {"name": "convs1.2", "class": "Conv1d", "notes": ""},
    {"name": "convs2.0", "class": "Conv1d", "notes": ""},
    {"name": "convs2.1", "class": "Conv1d", "notes": ""},
    {"name": "convs2.2", "class": "Conv1d", "notes": ""},
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


def _extract_conv1d(conv):
    """Extract Conv1d weights/config as ttnn-ready tensors + a params dict."""
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


class HifiGanResidualBlock:
    def __init__(self, device, torch_module):
        self.device = device
        self._torch_module = torch_module
        self.leaky_relu_slope = float(getattr(torch_module, "leaky_relu_slope", 0.1))
        self.convs1 = [_extract_conv1d(c) for c in torch_module.convs1]
        self.convs2 = [_extract_conv1d(c) for c in torch_module.convs2]

    def _apply_conv1d(self, x_nlc, cfg, input_length):
        # x_nlc: ttnn tensor, ROW_MAJOR, shape (1, L, C)
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
        # conv1d output can be sharded / TILE — normalize to DRAM ROW_MAJOR
        # 3D shape (1, out_length, out_channels) for the next stage.
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.reshape(out, (1, input_length, cfg["out_channels"]))
        return out

    def __call__(self, x, **kwargs):
        # Test hands us `x` with torch's NCL layout: (1, C, L). ttnn.conv1d
        # wants NLC (1, L, C) in ROW_MAJOR layout. Move to torch to permute
        # the axes cleanly (device-side permute on a TILE_LAYOUT tensor is
        # brittle here), then re-upload — this is a layout adapter, not a
        # compute fallback: all 6 convs + relus + adds still run on device.
        c_dim = int(x.shape[1])
        l_dim = int(x.shape[2])
        x_torch = ttnn.to_torch(x).to(torch.float32)
        x_nlc_torch = x_torch.permute(0, 2, 1).contiguous()  # (1, L, C)
        hidden = ttnn.from_torch(
            x_nlc_torch.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        for cfg1, cfg2 in zip(self.convs1, self.convs2):
            residual = hidden
            hidden = ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)
            hidden = ttnn.leaky_relu(hidden, self.leaky_relu_slope)
            hidden = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
            hidden = self._apply_conv1d(hidden, cfg1, input_length=l_dim)
            hidden = ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)
            hidden = ttnn.leaky_relu(hidden, self.leaky_relu_slope)
            hidden = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
            hidden = self._apply_conv1d(hidden, cfg2, input_length=l_dim)
            # residual + hidden — both NLC ROW_MAJOR, same shape
            hidden = ttnn.add(hidden, residual)

        # Permute back to NCL for the reference compare.
        out_torch = ttnn.to_torch(hidden).to(torch.float32)
        out_ncl_torch = out_torch.permute(0, 2, 1).contiguous()  # (1, C, L)
        return ttnn.from_torch(
            out_ncl_torch.to(torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )


def build(device, torch_module):
    return HifiGanResidualBlock(device, torch_module)


_instance = None


def hifi_gan_residual_block(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `hifi_gan_residual_block`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
