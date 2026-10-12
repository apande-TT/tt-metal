# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf (trace replay) for the Devstral pipeline: every PIPELINE_STAGES entry (encode, prefill, decode) is set up
from its own <stage>_trace_inputs(), captured as a trace and replayed on one command queue by the harness's
trace_replay.measure_adapter. Prints TRACE_STAGE_MS[<stage>] ... path=, TRACE_PER_TOKEN_MS and
TRACE_HEADLINE_UNIT. Perf only -- correctness is tests/e2e/test_e2e_*.py.
"""
from __future__ import annotations

import os
import time

import ttnn
from models.demos.devstral_small_2_24b_instruct_2512.mesh import DEFAULT_MESH, close_mesh, open_mesh
from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P
from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
_MESH_SHAPE = resolve_mesh_shape(default_rows=DEFAULT_MESH[0], default_cols=DEFAULT_MESH[1])
_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", str(P.TRACE_REGION_SIZE)))


def _build_for_perf(dev):
    return P.build_pipeline(dev, layers=PERF_LAYERS, batch=PERF_BATCH or None)


def test_text_generation_perf():
    device = open_mesh(trace_region_size=_TRACE_REGION if _PERF_TRACE else 0, mesh_shape=_MESH_SHAPE)
    try:
        _run(device)
    finally:
        close_mesh(device)


def _run(device):
    def _eager_forward():
        counter = [0]
        orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(device)
                    except Exception:  # noqa: BLE001
                        pass
                return r

            return inner

        pipe = _build_for_perf(device)
        processor = P.load_processor()
        from models.demos.devstral_small_2_24b_instruct_2512.demo.inputs import IMAGE_PROMPT, load_image

        req = P.encode_request(
            processor, IMAGE_PROMPT, image=load_image(), batch=pipe.batch, kv_capacity=pipe.kv_capacity
        )
        mods = [ttnn] + [getattr(ttnn, m, None) for m in ("transformer", "experimental")]
        for mod in [m for m in mods if m is not None]:
            for n in dir(mod):
                op = getattr(mod, n, None)
                if type(op).__name__ == "FastOperation":
                    orig.append((mod, n, op))
                    setattr(mod, n, _draining(op))
        t0 = time.monotonic()
        try:
            state = pipe.decode_prefill(req)
            for _ in range(_EAGER_OSL_TOKENS - 1):
                state = pipe.decode_step(state)
            ttnn.synchronize_device(device)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:  # noqa: BLE001
                pass
        finally:
            for mod, n, f in orig:
                setattr(mod, n, f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - t0) * 1000.0), flush=True)
        assert state is not None

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        prompt_ids = prompt_ids_for_isl(P.load_processor().tokenizer, PERF_ISL_TOKENS)
        print("PERF_ISL_TOKENS=%d" % prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        measure_adapter(PipelineStageAdapter(_build_for_perf, prompt_ids, batch=PERF_BATCH), device)

    def _try_traced():
        try:
            _traced_forward()
            return True
        except Exception as te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (te,), flush=True)
            return False

    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not profiling:
        if not _try_traced():
            print("TRACE_REPLAY_FALLBACK=eager", flush=True)
            _eager_forward()
    else:
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()
