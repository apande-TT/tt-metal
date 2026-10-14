# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level partial TTNN port for `decoder_head` of mistralai/Devstral-Small-2-24B-Instruct-2512.

Generated deterministically by `tt_hw_planner op-synth`.
Weight loading and op-REUSE/op-ADAPT helpers below are
machine-emitted from the HF reference and DO NOT need LLM
review. The `__call__` implementation falls back to HF
torch so the bring-up smoke test passes immediately;
the LLM's only remaining job is to rewrite `__call__`
to call the pre-bound `_apply_*` helpers in the right
order and fill any op-NEW gaps inline.

Pre-bound deterministic helpers (op palette):
#   self._apply__root_(x) -> ttnn.linear  (in=5120, out=131072, bias=False)

LLM_GAPs (op-NEW — still need synthesis):
#   (none — fully deterministic, no LLM_GAPs)

HF reference: (none)
Op counts: total=1  op-REUSE=1  op-ADAPT=0  op-NEW=0"""
from __future__ import annotations

import torch
import transformers

import ttnn

HF_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
_CANDIDATE_SUBMODULE_PATHS = ["mask_decoder", "decoder", "decode_head", "classifier", "lm_head"]


def _mesh_kw(device, **kw):
    """Return ``ttnn.from_torch`` kwargs that are mesh-aware.

    On a plain ``ttnn.Device`` this returns the caller's kwargs unchanged plus
    ``device=device`` — byte-identical to the pre-existing single-chip call
    pattern. On a ``ttnn.MeshDevice`` this also inserts
    ``mesh_mapper=ttnn.ReplicateTensorToMesh(device)`` so ``from_torch`` knows
    how to fan the tensor across chips (without this on a MeshDevice, the C++
    upload path stalls or produces incorrect placement).
    """
    kw["device"] = device
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except AttributeError:
        pass
    return kw


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


class DecoderHead:
    """lm_head (Linear 5120 -> 131072, no bias), tensor-parallel over the mesh.

    Column-parallel: each device holds vocab/TP output columns of the weight; the per-device logits
    are all_gathered on the vocab dim so every device ends with the full [.., vocab] row.
    """

    def __init__(self, device, torch_module):
        self.device = device
        try:
            self.tp = int(device.get_num_devices())
        except Exception:
            self.tp = 1
        w = torch_module.weight.detach().T.contiguous().to(torch.bfloat16)  # (in, vocab)
        assert w.shape[-1] % self.tp == 0, f"vocab {w.shape[-1]} not divisible by TP={self.tp}"
        # bfloat8_b weight: the lm_head is the largest single DRAM read of a decode step (vocab/TP x 5120), so
        # block-float halves it; accumulation stays fp32.
        self.w__root__weight = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if self.tp > 1 else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )

    def _apply__root_(self, x):
        return ttnn.linear(x, self.w__root__weight, compute_kernel_config=self.compute_cfg)

    def __call__(self, x, *args, gather=True, **kwargs):
        """gather=False returns each device's (.., vocab/TP) shard (the decode sampler only needs .local)."""
        shape = tuple(x.shape)
        if len(shape) != 4:
            x = ttnn.reshape(x, (1,) * (4 - len(shape)) + shape)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        out = self._apply__root_(x)
        # The pre-gather (.., vocab/TP) shard stays reachable: the on-device sampler takes its top-k per device
        # and merges the TP candidate lists instead of sorting the full gathered vocab row.
        self.local = out
        if self.tp > 1 and gather:
            out = ttnn.all_gather(out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        return ttnn.reshape(out, shape[:-1] + (out.shape[-1],))


def build(device, torch_module):
    return DecoderHead(device, torch_module)


_instance = None


def decoder_head(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `decoder_head`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
