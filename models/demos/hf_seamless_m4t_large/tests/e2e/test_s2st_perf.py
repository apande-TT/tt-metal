# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test for facebook/hf-seamless-m4t-large s2st pipeline."""
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

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"

# small representative audio for a dispatch-dense perf pass (NOT a max-shape correctness run)
_AUDIO_SECONDS = float(os.environ.get("TT_PERF_AUDIO_SECONDS", "1.0"))
_AUDIO_SR = 16000


def _synth_audio(seconds: float = _AUDIO_SECONDS, sr: int = _AUDIO_SR) -> torch.Tensor:
    t = torch.linspace(0.0, seconds, int(sr * seconds))
    return 0.1 * torch.sin(2 * math.pi * 220.0 * t)


def test_s2st_perf():
    # DEVICE OPEN — self-open exactly as demo/demo_s2st.py does (ttnn.open_device single-chip),
    # threading trace_region_size / num_command_queues through when TT_PERF_TRACE is set.
    open_kwargs = {}
    if _PERF_TRACE:
        open_kwargs["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
        open_kwargs["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))
    device = ttnn.open_device(device_id=0, **open_kwargs)
    try:
        # drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        # operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
        # tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
        # transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
        # dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
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
            # build EXACTLY as demo/demo_s2st.py does
            pipe = build_pipeline(device)
            proc = pipe.hf_processor
            audio = _synth_audio().numpy()
            inputs = proc(audio=audio, sampling_rate=_AUDIO_SR, return_tensors="pt")
            input_features = inputs["input_features"]
            # BOUNDED: cap the decode via PERF_MAX_NEW_TOKENS (demo default is 8)
            out = pipe.run_s2st(
                input_features,
                tgt_lang="eng",
                spkr_id=0,
                N=PERF_MAX_NEW_TOKENS,
            )
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

                _prompt_ids = torch.zeros((1, 8), dtype=torch.long)
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)