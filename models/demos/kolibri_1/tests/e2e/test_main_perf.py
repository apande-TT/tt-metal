# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance test for the Kolibri-1 'main' (text generation) pipeline.

Builds and runs the TT pipeline exactly as tests/e2e/test_e2e_text_generation.py does (tt/pipeline.py's
build_pipeline on the (1, 4) mesh_device fixture with the demo's device_params), keeping ONLY the on-device
forward: no reference model, no golden, no PCC / agreement gates.
"""
from __future__ import annotations

import contextlib
import gc
import os
import time

import pytest

import ttnn
from models.demos.kolibri_1.demo.mesh import device_params as _demo_device_params
from models.demos.kolibri_1.tt import inputs as kin
from models.demos.kolibri_1.tt.pipeline import OSL_ENV, build_pipeline

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402,F401

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=4)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The source's own device params, verbatim; only fabric is gated on the resolved mesh spanning > 1 chip.
_DEV_PARAMS = dict(_demo_device_params())
if _MESH_SHAPE[0] * _MESH_SHAPE[1] <= 1:
    _DEV_PARAMS.pop("fabric_config", None)
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    _DEV_PARAMS["trace_region_size"] = max(
        int(_DEV_PARAMS.get("trace_region_size", 0) or 0),
        int(os.environ.get("TT_PERF_TRACE_REGION", "41943040")),
    )
    _DEV_PARAMS["num_command_queues"] = 1


def _model_batch():
    """BATCH BELONGS TO THE MODEL: a positive TT_PERF_BATCH overrides; 0/absent asks the model's own input
    module (kin.batch_size(), the batch the correctness test builds with), with the 0 sentinel hidden."""
    if PERF_BATCH > 0:
        return PERF_BATCH
    saved = os.environ.pop("TT_PERF_BATCH", None)
    try:
        return kin.batch_size()
    finally:
        if saved is not None:
            os.environ["TT_PERF_BATCH"] = saved


@contextlib.contextmanager
def _osl_cap(n):
    """Bound the pipeline's generation horizon through its own harness cap (OSL_ENV), restored after."""
    saved = os.environ.get(OSL_ENV)
    os.environ[OSL_ENV] = str(int(n))
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(OSL_ENV, None)
        else:
            os.environ[OSL_ENV] = saved


def _as_id_list(ids):
    if hasattr(ids, "reshape"):
        ids = ids.reshape(-1).tolist()
    return [int(x) for x in ids]


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_main_perf(mesh_device):
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

    B = _model_batch()
    print(f"PERF_BATCH_STREAMS={B}", flush=True)
    tok = kin.load_tokenizer()
    _isl_ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
    prompt = _as_id_list(_isl_ids)
    seeds = list(range(B))
    print("PERF_ISL_TOKENS=%d" % len(prompt), flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)

    def _eager_forward():
        print("PERF_EAGER_OSL_TOKENS=%d" % _EAGER_OSL_TOKENS, flush=True)
        with _osl_cap(_EAGER_OSL_TOKENS):
            pipe = build_pipeline(mesh_device, layers=PERF_LAYERS, batch=B)
            # --- per-stage marks (injected) ---------------------------------------------------
            # Runs HERE, at the end of the function that built the pipeline, because that object is a LOCAL of
            # this scope: an earlier version copied the test's own PipelineStageAdapter(...) arguments into the
            # profiling branch and raised NameError, since the generator had defined them inside another
            # function. Handed locals() rather than a name, so nothing depends on how the test spells things.
            print("STAGE_MARKS_ENTER", flush=True)
            try:
                from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

                print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals()), flush=True)
            except Exception as _tt_e2:  # noqa: BLE001
                print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)
            try:
                ttnn.ReadDeviceProfiler(mesh_device)  # drain build-time markers before the forward
            except Exception:
                pass
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
                    if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                out = pipe.generate(prompt, seeds)
                try:
                    ttnn.ReadDeviceProfiler(mesh_device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC
        tokens = out["tokens"]
        assert tokens and any(len(t) > 0 for t in tokens), "pipeline produced no tokens"
        print(f"[perf] generated {sum(len(t) for t in tokens)} tokens over {len(tokens)} rows", flush=True)
        del pipe, out
        gc.collect()

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        def _build_for_perf(dev):
            return build_pipeline(dev, layers=PERF_LAYERS, batch=B)

        # ISL: build the prompt to EXACTLY PERF_ISL_TOKENS tokens rather than writing an example
        # sentence, so the measurement condition is the tool's choice and not the generator's.
        _prompt_ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
        # traced. Falls back to the single decode contract for pipelines that expose only decode_step.
        measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), mesh_device)

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
            print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
            _eager_forward()
    else:
        # --- stage marks (injected by perf_test_gen) -------------------------------------
        # The measured region is bracketed by the conventional start/stop pair so the main report
        # slices exactly the ops run_head emitted; the pass below is additive and feeds per-stage
        # fidelity only. Injected rather than written by the generator: the skeleton is advisory and
        # a generated test simply omitted this, which is why five earlier attempts measured nothing.
        try:
            from models.experimental.perf_automation.agent import stage_marks as _tt_sm
        except Exception:  # noqa: BLE001
            _tt_sm = None
        if _tt_sm is not None:
            _tt_sm.signpost("start")
        _eager_forward()
        if _tt_sm is not None:
            _tt_sm.signpost("stop")
        if _PERF_TRACE:
            _try_traced()
