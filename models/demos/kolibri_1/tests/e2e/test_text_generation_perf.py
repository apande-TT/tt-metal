# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kolibri-1 text-generation PERFORMANCE test (perf only, no PCC).

Builds and runs the pipeline exactly as demo/demo_text_generation.py does: the demo's own open_mesh()
(QB2 1x4 Blackhole mesh, TP=4), build_pipeline(mesh, layers=..., batch=...), then
pipe.generate(prompt, seeds, max_new_tokens=...), with user b sampling under seed b. Every device op
runs in this process so tracy sees all of them.
"""
from __future__ import annotations

import gc
import inspect
import os
import time

import pytest  # noqa: F401
import torch

import ttnn
from models.demos.kolibri_1.demo.mesh import close_mesh, open_mesh
from models.demos.kolibri_1.tt import inputs as kin
from models.demos.kolibri_1.tt.pipeline import build_pipeline

# ONE VARIABLE FOR ONE THING. There were two: PERF_OSL_TOKENS, which the test PRINTED, and
# PERF_OSL_TOKENS, which the loop actually RAN -- with its own default of 4, from the generator's
# first commit. So a declared OSL of 128 was reported while 4 executed, and every profile sampled a
# thirty-second of the request it claimed to measure. Two names for one quantity is how a setting can
# be honoured and ignored at the same time. TT_PERF_OSL_TOKENS is GONE; a probe that wants a
# cheaper unit sets TT_PERF_OSL_TOKENS, and then the declared unit and the executed one still agree.
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS, and they default to a REALISTIC operating point rather
# than to whatever example prompt reads naturally. Left unspecified, a generated perf test used the
# shortest prompt that proves the pipeline runs -- on llama3_1_8b_p150 that was "The capital of
# France is", six tokens, and nothing recorded that the throughput number was a six-token one.
# Decode is weight-bandwidth bound so ISL barely moves tok/s/u (measured: 0.5% from ISL 6 to 128),
# but TTFT, prefill cost and any long-context claim all depend on it, so the default must be a
# figure someone would actually quote. 128 in / 128 out is the industry-standard short-context
# benchmark point. Both are env-overridable; the markers below record what actually ran, so a
# reader never has to guess the conditions.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# EAGER-PATH BOUND. The eager forward wraps every ttnn op to drain the device profiler; on a
# high-op-count model a long decode there makes the profiler's never-freed per-(chip,core) host
# buffers pile up until the HOST OOMs -- a 6-layer build has reached 224 GB this way. The traced
# path, which produces the reported OSL number, does not (it runs profiler-off), so only the eager
# path needs bounding: it runs a SHORT decode capped here while the trace keeps the full
# PERF_OSL_TOKENS. Env-overridable; never exceeds the declared OSL.
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
# BATCH BELONGS TO THE MODEL, not to this generator. It was written into the generated test as a
# literal `batch=1`, so a pipeline emit-e2e built to serve 8 users was measured serving one, and its
# aggregate throughput under-reported by 8x. 0 means "ask the pipeline" -- it already knows, via
# max_batch_size / batch_size / batch, whichever it exposes -- and any positive value overrides, for
# sweeping batch without rebuilding the demo. Unlike ISL and OSL, which are the TOOL's choice of
# measurement condition, batch is a property of the artifact under test.
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
# resolve_mesh_shape is how a run honours them. Give it the SOURCE's own shape as the default, so an
# unset env behaves exactly as the demo does. The demo SELF-OPENS (demo/mesh.open_mesh, QB2 1x4,
# TP=4), so this test self-opens the same way and hands that mesh to the pipeline.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402,F401

_SOURCE_MESH = (1, 4)  # demo_text_generation: QB2 1x4 Blackhole mesh, TP=4
_MESH_SHAPE = resolve_mesh_shape(default_rows=_SOURCE_MESH[0], default_cols=_SOURCE_MESH[1])

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The demo's open_mesh owns fabric / L1 setup; the only params this test adds are the trace region,
# and only when that open accepts them (otherwise the trace block falls back by itself).
_DEV_PARAMS = {}
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))
    _DEV_PARAMS["num_command_queues"] = 1


def _shape_kwargs(params, rows, cols):
    """How open_mesh takes a mesh shape, if it takes one at all."""
    for name in ("mesh_shape", "shape"):
        if name in params:
            dflt = params[name].default
            if isinstance(dflt, (tuple, list)) or (name == "shape" and type(dflt).__name__ != "MeshShape"):
                return {name: (rows, cols)}
            return {name: ttnn.MeshShape(rows, cols)}
    for r, c in (("rows", "cols"), ("num_rows", "num_cols")):
        if r in params and c in params:
            return {r: rows, c: cols}
    return None


def _open_perf_mesh():
    """Open the mesh the way the demo does (open_mesh()). Returns (mesh, opened_by_hand)."""
    rows, cols = _MESH_SHAPE
    try:
        params = inspect.signature(open_mesh).parameters
    except (TypeError, ValueError):
        params = {}
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kw = {}
    if (rows, cols) != _SOURCE_MESH:
        shape_kw = _shape_kwargs(params, rows, cols)
        if shape_kw is None:
            # open_mesh hardwires the demo's 1x4 and cannot take the planned shape: open it directly,
            # with fabric only when the planned mesh spans more than one chip.
            print("MESH_OPEN=direct %dx%d (open_mesh takes no shape)" % (rows, cols), flush=True)
            if rows * cols > 1:
                ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            return ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **_DEV_PARAMS), True
        kw.update(shape_kw)
    trace_kw = {k: v for k, v in _DEV_PARAMS.items() if k in params or var_kw}
    if trace_kw:
        try:
            return open_mesh(**kw, **trace_kw), False
        except TypeError as _e:
            print("TRACE_OPEN_KWARGS_REJECTED=%r" % (_e,), flush=True)
    return open_mesh(**kw), False


def _build_batch_kw():
    """The demo's batch argument: an explicit TT_PERF_BATCH override, else the model's own default."""
    if PERF_BATCH > 0:
        return {"batch": PERF_BATCH}
    try:
        b = int(kin.batch_size())
    except Exception:  # noqa: BLE001
        b = 0
    return {"batch": b} if b > 0 else {}


def _pipe_batch(pipe, build_kw):
    """Ask the pipeline what batch it was built for (one seed per user, as the demo does)."""
    if PERF_BATCH > 0:
        return PERF_BATCH
    for attr in ("max_batch_size", "batch_size", "batch"):
        v = getattr(pipe, attr, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return int(build_kw.get("batch") or 32)  # the demo's documented default


def _isl_prompt_ids(tok):
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

    return prompt_ids_for_isl(tok, PERF_ISL_TOKENS)


def _as_demo_prompt(ids, ref):
    """Shape the ISL prompt ids like kin.encode_prompt's output, which pipe.generate expects."""
    flat = [int(t) for t in ids.reshape(-1).tolist()]
    if isinstance(ref, (list, tuple)):
        return flat
    if isinstance(ref, torch.Tensor):
        return torch.tensor(flat, dtype=ref.dtype).reshape(*([1] * (ref.dim() - 1)), -1)
    return ids


def test_text_generation_perf():
    mesh, _opened_by_hand = _open_perf_mesh()
    try:

        def _eager_forward():
            gc.collect()
            counter = [0]
            _orig = []

            def _draining(fn):
                def inner(*a, **k):
                    r = fn(*a, **k)
                    counter[0] += 1
                    if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                        try:
                            ttnn.ReadDeviceProfiler(mesh)
                        except Exception:
                            pass
                    return r

                return inner

            tok = kin.load_tokenizer()
            _ids = _isl_prompt_ids(tok)
            prompt = _as_demo_prompt(_ids, kin.encode_prompt(tok, kin.CARD_MESSAGE))
            print("PERF_ISL_TOKENS=%d" % int(_ids.shape[-1]), flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            print("PERF_EAGER_OSL_TOKENS=%d" % _EAGER_OSL_TOKENS, flush=True)

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            try:
                build_kw = _build_batch_kw()
                pipe = build_pipeline(mesh, layers=PERF_LAYERS, **build_kw)
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
                    ttnn.ReadDeviceProfiler(mesh)
                except Exception:
                    pass
                seeds = list(range(_pipe_batch(pipe, build_kw)))
                print("PERF_BATCH_USERS=%d" % len(seeds), flush=True)
                _fw0 = time.monotonic()
                out = pipe.generate(prompt, seeds, max_new_tokens=_EAGER_OSL_TOKENS)
                try:
                    ttnn.ReadDeviceProfiler(mesh)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None  # perf only — NO PCC
            assert len(out["tokens"]) == len(seeds)
            print(
                "[perf] %d users, %s steps, %d tokens"
                % (len(seeds), out.get("steps"), sum(len(ids) for ids in out["tokens"])),
                flush=True,
            )
            del pipe
            gc.collect()

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                from models.demos.kolibri_1.tt.pipeline import build_pipeline

                return build_pipeline(dev, layers=PERF_LAYERS, **_build_batch_kw())

            # ISL: build the prompt to EXACTLY PERF_ISL_TOKENS tokens rather than writing an example
            # sentence, so the measurement condition is the tool's choice and not the generator's.
            _prompt_ids = _isl_prompt_ids(kin.load_tokenizer())
            print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
            # traced. Falls back to the single decode contract for pipelines that expose only decode_step.
            measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), mesh)

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
            except Exception:  # noqa: BLE001
                _tt_sm = None
            if _tt_sm is not None:
                _tt_sm.signpost("start")
            _eager_forward()
            if _tt_sm is not None:
                _tt_sm.signpost("stop")
            if _PERF_TRACE:
                _try_traced()
    finally:
        close_mesh(mesh)
        if _opened_by_hand:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
