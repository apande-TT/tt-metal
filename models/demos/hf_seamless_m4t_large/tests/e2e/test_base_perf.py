# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test for hf_seamless_m4t_large 'base' pipeline — text output path."""
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
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))


def test_base_perf():
    # Source self-opens with ttnn.open_device(device_id=0) — match that here (no pytest fixture).
    device = ttnn.open_device(device_id=0, **_DEV_PARAMS)
    try:
        pipe = build_pipeline(device)
        proc = pipe.hf_processor
        inputs = proc(text="Hello, my dog is cute.", src_lang="eng", return_tensors="pt")
        input_ids = inputs["input_ids"]

        # MODEL-AGNOSTIC drain: wrap every ttnn FastOperation across ttnn + op submodules so the flush
        # counter tracks TOTAL device dispatch for any op mix; a curated list would under-count.
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
        out = None
        try:
            pcc, out_a, out_b = pipe.run_base(
                generate_speech=False,
                input_ids=input_ids,
                tgt_lang="fra",
                spkr_id=0,
                N=PERF_MAX_NEW_TOKENS,
            )
            out = out_a
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
                    from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline as _bp
                    return _bp(dev)

                _prompt_ids = input_ids
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)