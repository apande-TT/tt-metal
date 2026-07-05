# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test — s2tt pipeline for facebook/hf-seamless-m4t-large.

Build + forward exactly matches demo/demo_s2tt.py (in-process; tracy only sees this process).
Bounded decode + model-agnostic profiler drain keep the tracy marker buffer under 12000.
"""
from __future__ import annotations

import math
import os
import time

import pytest
import torch

import ttnn
from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# small audio window keeps the encoder mel-spec dispatch-dense but bounded under tracy
PERF_AUDIO_SECONDS = float(os.environ.get("TT_PERF_AUDIO_SECONDS", "1.0"))
PERF_SR = int(os.environ.get("TT_PERF_SR", "16000"))
PERF_TGT_LANG = os.environ.get("TT_PERF_TGT_LANG", "eng")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


def _synth_audio(seconds: float = PERF_AUDIO_SECONDS, sr: int = PERF_SR) -> torch.Tensor:
    """Deterministic sinusoid — no external audio download required (matches demo_s2tt)."""
    t = torch.linspace(0.0, seconds, int(sr * seconds))
    return 0.1 * torch.sin(2 * math.pi * 220.0 * t)


def _open_device_matching_demo():
    """Demo self-opens via ttnn.open_device(device_id=0); mirror that here.

    Pass trace_region_size / num_command_queues through when TT_PERF_TRACE is set so the trace-replay
    block below can capture a trace on the same device the eager forward ran on. Fall back to the
    plain demo call if ttnn.open_device on this build does not accept those kwargs.
    """
    try:
        return ttnn.open_device(device_id=0, **_DEV_PARAMS)
    except TypeError:
        return ttnn.open_device(device_id=0)


def test_s2tt_perf():
    device = _open_device_matching_demo()
    try:
        # 1) build the pipeline EXACTLY as demo/demo_s2tt.py does
        pipe = build_pipeline(device)
        proc = pipe.hf_processor
        audio = _synth_audio().numpy()
        inputs = proc(audio=audio, sampling_rate=PERF_SR, return_tensors="pt")
        input_features = inputs["input_features"]

        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
        #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
        #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
        #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(device)
                    except Exception:
                        pass
                return r
            return inner

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        _fw0 = time.monotonic()
        try:
            # BOUNDED forward: cap the decode loop via N=PERF_MAX_NEW_TOKENS (demo default is 16).
            pcc, tt_tokens, hf_tokens = pipe.run_s2tt(
                input_features, tgt_lang=PERF_TGT_LANG, N=PERF_MAX_NEW_TOKENS
            )
            out = tt_tokens
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

                def _build_for_perf(dev):
                    return build_pipeline(dev)

                _prompt_ids = torch.tensor([[3]], dtype=torch.long)
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)