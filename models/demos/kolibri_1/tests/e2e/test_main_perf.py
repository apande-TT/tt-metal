# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test for the Kolibri-1 'main' pipeline (text generation).

Builds and runs the TT pipeline (tt/pipeline.py, the same code the demo and the e2e correctness gate
run) on the same mesh topology as tests/e2e/test_e2e_text_generation.py, keeping ONLY the on-device
forward: no reference model, no golden, no PCC / agreement checks.
"""
import gc
import os
import time

import pytest
import ttnn

from models.demos.kolibri_1.demo.mesh import device_params
from models.demos.kolibri_1.tt import inputs as kin
from models.demos.kolibri_1.tt.pipeline import OSL_ENV, build_pipeline

# ONE VARIABLE FOR ONE THING. TT_PERF_OSL_TOKENS is both the declared and the executed unit.
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS. 128 in / 128 out is the industry-standard short-context
# benchmark point. Both are env-overridable; the markers below record what actually ran.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# EAGER-PATH BOUND. The eager forward wraps every ttnn op to drain the device profiler; a long decode
# there piles up the profiler's per-(chip,core) host buffers until the HOST OOMs. The traced path
# (profiler-off) keeps the full PERF_OSL_TOKENS; only the eager path is bounded. Never exceeds the
# declared OSL.
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
# BATCH BELONGS TO THE MODEL. 0 means "ask the pipeline"; any positive value overrides.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
# Pass PERF_LAYERS straight to the builder: None is every builder's own all-layers value. Do NOT
# default it to a number here -- that would silently cap the full-depth gate.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape honours them, defaulting to the source's own (1, 4) mesh_device parametrize.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: F401,E402

_MESH_SHAPE = tuple(resolve_mesh_shape(default_rows=1, default_cols=4))

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The source's own device params (demo/mesh.py), verbatim; ONLY fabric is gated on a multi-chip mesh.
_DEV_PARAMS = dict(device_params())
if _MESH_SHAPE[0] * _MESH_SHAPE[1] <= 1:
    _DEV_PARAMS.pop("fabric_config", None)
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    _DEV_PARAMS["trace_region_size"] = int(
        os.environ.get(
            "TT_PERF_TRACE_REGION", str(max(41943040, int(_DEV_PARAMS.get("trace_region_size") or 0)))
        )
    )
    _DEV_PARAMS["num_command_queues"] = 1


def _model_batch():
    """The batch the model declares (kin.batch_size() with no override in the environment), unless
    TT_PERF_BATCH is positive. TT_PERF_BATCH=0 means "ask the model", so it is hidden from
    kin.batch_size(), which would otherwise read it as a literal zero."""
    if PERF_BATCH > 0:
        return PERF_BATCH
    saved = os.environ.pop("TT_PERF_BATCH", None)
    try:
        return int(kin.batch_size())
    finally:
        if saved is not None:
            os.environ["TT_PERF_BATCH"] = saved


def _ids_list(ids):
    if hasattr(ids, "reshape"):
        return [int(t) for t in ids.reshape(-1).tolist()]
    return [int(t) for t in ids]


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_main_perf(mesh_device):
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

    tok = kin.load_tokenizer()

    def _eager_forward():
        B = _model_batch()
        print(f"PERF_BATCH_STREAMS={B}", flush=True)
        prompt = _ids_list(prompt_ids_for_isl(tok, PERF_ISL_TOKENS))
        print("PERF_ISL_TOKENS=%d" % len(prompt), flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_EAGER_OSL_TOKENS=%d" % _EAGER_OSL_TOKENS, flush=True)
        seeds = list(range(B))

        # The pipeline's own horizon cap (the same switch the harness uses) bounds the eager decode.
        saved_osl = os.environ.get(OSL_ENV)
        os.environ[OSL_ENV] = str(_EAGER_OSL_TOKENS)
        try:
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

            # Drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
            # operation (type 'FastOperation') across ttnn + its op submodules.
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
            steps = []
            _fw0 = time.monotonic()
            try:
                out = pipe.generate(prompt, seeds, on_step=lambda i, lg, ids: steps.append(i))
                try:
                    ttnn.ReadDeviceProfiler(mesh_device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        finally:
            if saved_osl is None:
                os.environ.pop(OSL_ENV, None)
            else:
                os.environ[OSL_ENV] = saved_osl
        print(f"PERF_EAGER_STEPS={len(steps)}", flush=True)
        assert out is not None  # perf only — NO PCC
        assert out["tokens"] and len(out["tokens"]) == B, "pipeline produced no output"
        del pipe, out
        gc.collect()

    def _traced_forward():
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

        def _build_for_perf(dev):
            from models.demos.kolibri_1.tt.pipeline import build_pipeline as _build

            return _build(dev, layers=PERF_LAYERS, batch=_model_batch())

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

    # MEASUREMENT ORDER — two consumers, two different needs, and running both is not free.
    #   TRACY PROFILING RUN (TT_METAL_DEVICE_PROFILER=1, layer-capped): needs BOTH products. The
    #     op-wrapped eager forward IS the per-op capture; the trace pass supplies
    #     TRACE_PER_TOKEN_MS for throughput. Two different measurements, so both run.
    #   FULL-PIPELINE GATE (no tracy, TT_PERF_LAYERS=0, FULL depth): needs exactly ONE whole-model
    #     latency. Running both builds the model TWICE at full depth on one device -- the second
    #     build has no memory left for its KV cache and dies before any marker is printed.
    # So the gate runs TRACE FIRST and only falls back to the eager forward when trace genuinely
    # could not be measured. That is the designed contract: trace by default, eager as the fallback.
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
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter as _TtPSA
        except Exception:  # noqa: BLE001
            _tt_sm = None
        if _tt_sm is not None:
            _tt_sm.signpost("start")
        _eager_forward()
        if _tt_sm is not None:
            _tt_sm.signpost("stop")
        if _PERF_TRACE:
            _try_traced()