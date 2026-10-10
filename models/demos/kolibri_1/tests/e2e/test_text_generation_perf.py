# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kolibri-1 text_generation perf: the demo's pipeline (tt/pipeline.py) on the self-opened QB2 mesh
(1x4 Blackhole, TP=4), run in-process so tracy sees every device op.

Trace replay (TRACE_PER_TOKEN_MS / TRACE_STAGE_MS) is the default measurement; the op-wrapped eager
forward (FORWARD_WALL_MS) is its fallback, and the per-op capture under TT_METAL_DEVICE_PROFILER=1.
"""
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

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL are the tool's measurement conditions; the markers below record what actually ran.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# EAGER-PATH BOUND: the op-wrapped eager decode is short so the profiler's per-(chip,core) host
# buffers cannot grow until the host OOMs; the traced path keeps the full PERF_OSL_TOKENS.
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
# BATCH BELONGS TO THE MODEL: 0 means ask the pipeline; a positive value overrides.
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
# resolve_mesh_shape is how a run honours them. The default is the source's own shape: demo/mesh.py
# opens the QB2's 1x4 Blackhole mesh (TP=4).
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: F401,E402

_SOURCE_MESH = (1, 4)
_MESH_SHAPE = resolve_mesh_shape(default_rows=_SOURCE_MESH[0], default_cols=_SOURCE_MESH[1])

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_TRACE_REGION_ENV = os.environ.get("TT_PERF_TRACE_REGION")
_TRACE_REGION = int(_TRACE_REGION_ENV or "41943040")


def _open_perf_mesh():
    """Open the mesh the way the demo does (demo/mesh.py open_mesh), with the trace region reserved at
    open when TT_PERF_TRACE and open_mesh accepts it, and the tool's planned shape. Returns (mesh, close)."""
    try:
        params = inspect.signature(open_mesh).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    if _PERF_TRACE:
        if "trace_region_size" in params:
            _d = params["trace_region_size"].default
            # Never shrink the source's own reservation unless the tool explicitly asked for a size.
            kw["trace_region_size"] = (
                max(_TRACE_REGION, _d) if (_TRACE_REGION_ENV is None and isinstance(_d, int)) else _TRACE_REGION
            )
        if "num_command_queues" in params:
            kw["num_command_queues"] = 1
    rows, cols = _MESH_SHAPE
    for _k in ("mesh_shape", "shape"):
        if _k in params:
            _d = params[_k].default
            kw[_k] = (rows, cols) if isinstance(_d, (tuple, list)) else ttnn.MeshShape(rows, cols)
            return open_mesh(**kw), close_mesh
    if (rows, cols) == _SOURCE_MESH:
        return open_mesh(**kw), close_mesh
    # open_mesh takes no shape and the tool planned a different topology: open MeshShape(rows, cols)
    # directly, with fabric only when the mesh spans more than one chip.
    multi = rows * cols > 1
    if multi:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dkw = {"trace_region_size": _TRACE_REGION, "num_command_queues": 1} if _PERF_TRACE else {}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **dkw)

    def _close(m):
        ttnn.close_mesh_device(m)
        if multi:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    return mesh, _close


def _build_batch():
    """The batch the demo builds with: TT_PERF_BATCH when positive, else kin.batch_size() (the demo's
    default); None lets build_pipeline answer with its own declared batch."""
    if PERF_BATCH > 0:
        return PERF_BATCH
    try:
        b = int(kin.batch_size())
    except Exception:  # noqa: BLE001
        return None
    return b if b > 0 else None


def _batch_kw():
    b = _build_batch()
    return {"batch": b} if b else {}


def _pipe_batch(pipe):
    """Users the built pipeline serves (one seed each, as the demo does)."""
    for name in ("batch", "max_batch_size", "batch_size"):
        v = getattr(pipe, name, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    b = _build_batch()
    assert b, "pipeline exposes no batch and none was requested"
    return b


def _demo_prompt(tok, ids):
    """The ISL ids in the container the demo hands generate() (kin.encode_prompt's return type)."""
    flat = [int(t) for t in torch.as_tensor(ids).reshape(-1).tolist()]
    ref = kin.encode_prompt(tok, kin.CARD_MESSAGE)
    if torch.is_tensor(ref):
        return torch.tensor(flat, dtype=ref.dtype).reshape(*([1] * (ref.dim() - 1)), len(flat))
    return flat


def test_text_generation_perf():
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

    tok = kin.load_tokenizer()
    mesh, _close_mesh = _open_perf_mesh()
    try:
        # 1) build the pipeline EXACTLY as demo/demo_text_generation.py does
        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        #    operation (type 'FastOperation') across ttnn + its op submodules.
        def _eager_forward():
            _ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
            prompt = _demo_prompt(tok, _ids)
            print("PERF_ISL_TOKENS=%d" % _ids.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            print("PERF_EAGER_OSL_TOKENS=%d" % _EAGER_OSL_TOKENS, flush=True)
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

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            try:
                pipe = build_pipeline(mesh, layers=PERF_LAYERS, **_batch_kw())
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
                seeds = list(range(_pipe_batch(pipe)))
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
            assert out is not None and len(out["tokens"]) > 0  # perf only — NO PCC
            del pipe

        def _traced_forward():
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            def _build_for_perf(dev):
                return build_pipeline(dev, layers=PERF_LAYERS, **_batch_kw())

            # ISL: build the prompt to EXACTLY PERF_ISL_TOKENS tokens rather than writing an example
            # sentence, so the measurement condition is the tool's choice and not the generator's.
            _prompt_ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
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

        # MEASUREMENT ORDER — trace by default, eager as the fallback; under tracy both run (the eager
        # forward is the per-op capture, the trace pass supplies TRACE_PER_TOKEN_MS).
        _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        if _PERF_TRACE and not _PROFILING:
            if not _try_traced():
                print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
                gc.collect()
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
                gc.collect()
                _try_traced()
    finally:
        _close_mesh(mesh)