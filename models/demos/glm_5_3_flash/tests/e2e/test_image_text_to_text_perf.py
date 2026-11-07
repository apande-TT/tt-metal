# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PERFORMANCE profile of the GLM-5.3-Flash image-text-to-text pipeline (tt/pipeline.py).

Perf only -- NO PCC (correctness is gated by test_e2e_image_text_to_text.py). The pipeline is built exactly
as demo/demo_image_text_to_text.py builds it; every PIPELINE_STAGES entry (vision_encode, prefill, decode) is
captured as a device trace and replayed through the generic stage adapter, which prints TRACE_STAGE_MS per
stage and TRACE_PER_TOKEN_MS for the decode stage. Inputs (provenance): the published example stored under
_captured/e2e_example/ (pipeline-cat-chonk.jpeg + "Describe the image."), which each stage's
<stage>_trace_inputs() loads -- the image fixes the prompt length (ISL), so ISL is reported, not chosen.
"""
from __future__ import annotations

import os
import time

import pytest

import ttnn
from models.demos.glm_5_3_flash.tt import pipeline as P
from models.experimental.perf_automation.agent.perf_adapter import (
    resolve_batch,
    resolve_mesh_shape,
    resolve_trace_region,
)

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# eager path is the per-op profiling capture: bound it so the profiler's host buffers stay small
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
# 0 = ask the pipeline (it reads $TT_PERF_BATCH itself)
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# ABSENT = all layers; a positive value caps every repeated stack (tracy runs)
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None


def _stack_depth(var):
    v = (os.environ.get(var) or "").strip()
    return int(v) if (v.isdigit() and int(v) > 0) else None


_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=32)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _MESH_SHAPE[0] * _MESH_SHAPE[1] > 1:
    _DEV_PARAMS["fabric_config"] = ttnn.FabricConfig.FABRIC_1D
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = resolve_trace_region(P.TRACE_REGION_SIZE)
    _DEV_PARAMS["num_command_queues"] = 1


def _build(dev):
    kw = {} if PERF_BATCH <= 0 else {"batch": PERF_BATCH}
    return P.build_pipeline(
        dev,
        layers=PERF_LAYERS,
        vision_encode_layers=_stack_depth("TT_PERF_VISION_ENCODE_LAYERS"),
        prefill_layers=_stack_depth("TT_PERF_PREFILL_LAYERS"),
        decode_layers=_stack_depth("TT_PERF_DECODE_LAYERS"),
        **kw,
    )


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_image_text_to_text_perf(device_params, mesh_device):
    pipe = _build(mesh_device)
    enc = P.load_captured_example()
    print("PERF_ISL_TOKENS=%d" % int(enc["input_ids"].shape[-1]), flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print("PERF_BATCH_STREAMS=%d" % resolve_batch(pipe, PERF_BATCH), flush=True)

    def _eager_forward():
        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(mesh_device)
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
            out = P.run_image_text_to_text(
                pipe, pipe.encode(enc), max_new_tokens=_EAGER_OSL_TOKENS, keep_logits=False, use_trace=False
            )
            try:
                ttnn.ReadDeviceProfiler(mesh_device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0), flush=True)
        assert out is not None  # perf only -- NO PCC

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        # reuse the pipeline built above: a second full build does not fit next to the first
        measure_adapter(PipelineStageAdapter(lambda dev: pipe, None, batch=PERF_BATCH), mesh_device)

    def _try_traced():
        try:
            _traced_forward()
            return True
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
            return False

    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        if not _try_traced():
            print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working -- timing eagerly", flush=True)
            _eager_forward()
    else:
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()
