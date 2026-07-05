# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test for facebook/hf-seamless-m4t-large T2ST pipeline."""
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

PERF_MAX_SEQ = int(os.environ.get("TT_PERF_MAX_SEQ", "128"))

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))


def test_t2st_perf():
    # The demo self-opens a single device via ttnn.open_device — match it here. Pass trace params
    # through when profiling; if the open signature does not accept them, fall back to the plain
    # open the demo uses (the trace-replay block stays guarded).
    _open_kwargs = {"device_id": 0, "l1_small_size": _DEV_PARAMS["l1_small_size"]}
    if _PERF_TRACE:
        _open_kwargs["trace_region_size"] = _DEV_PARAMS["trace_region_size"]
        _open_kwargs["num_command_queues"] = _DEV_PARAMS["num_command_queues"]
    try:
        device = ttnn.open_device(**_open_kwargs)
    except TypeError:
        device = ttnn.open_device(device_id=0)

    try:
        # 1) build the pipeline EXACTLY as demo/demo_t2st.py does
        pipe = build_pipeline(device)
        proc = pipe.hf_processor
        inputs = proc(text="Hello, my dog is cute.", src_lang="eng", return_tensors="pt")
        input_ids = inputs["input_ids"]
        # cap the input token length small for perf (representative dispatch, not max-shape stress)
        if input_ids.shape[-1] > PERF_MAX_SEQ:
            input_ids = input_ids[..., :PERF_MAX_SEQ]

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
            pcc, tt_wave, hf_wave = pipe.run_t2st(
                input_ids,
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
        assert tt_wave is not None  # perf only — NO PCC

        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

                def _build_for_perf(dev):
                    return build_pipeline(dev)

                _proc = pipe.hf_processor
                _prompt_ids = _proc(text="Hi.", src_lang="eng", return_tensors="pt")["input_ids"]
                if _prompt_ids.shape[-1] > 32:
                    _prompt_ids = _prompt_ids[..., :32]
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)