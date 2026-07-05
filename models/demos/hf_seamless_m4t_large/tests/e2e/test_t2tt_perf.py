# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test for facebook/hf-seamless-m4t-large t2tt pipeline."""
from __future__ import annotations

import os
import time

import pytest
import ttnn

from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


def test_t2tt_perf():
    # Source demo self-opens the device with ttnn.open_device(device_id=0); mirror it here and
    # pass through perf params (trace_region_size / num_command_queues / l1_small_size).
    device = ttnn.open_device(device_id=0, **_DEV_PARAMS)
    try:
        # 1) build the pipeline EXACTLY as demo/demo_t2tt.py does
        pipe = build_pipeline(device)
        proc = pipe.hf_processor

        text = "Hello, my dog is cute."
        src_lang = "eng"
        tgt_lang = "fra"

        inputs = proc(text=text, src_lang=src_lang, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Cap the input size small (dispatch-dense pass, not a max-shape stress).
        _max_seq = int(os.environ.get("TT_PERF_MAX_SEQ", "128"))
        if input_ids.dim() >= 1 and input_ids.shape[-1] > _max_seq:
            input_ids = input_ids[..., :_max_seq]

        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
        #    tracks TOTAL device dispatch for ANY op mix.
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
            out = pipe.run_t2tt(input_ids, tgt_lang=tgt_lang, N=PERF_MAX_NEW_TOKENS)
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

                _prompt_ids = input_ids
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)