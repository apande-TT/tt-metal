# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Performance test for the `audio_chat` pipeline of `mistralai/Voxtral-Mini-3B-2507`.

Builds and runs the pipeline EXACTLY as `demo/demo_audio_chat.py` does -- the demo SELF-OPENS its
device (`ttnn.open_device(device_id=0, l1_small_size=..., trace_region_size=...)`), calls the
module-level `build_pipeline(device)` factory and then `pipe.run_audio_chat(batch, ...)` on a batch
from `tt.inputs.build_audio_chat_inputs`.  This test lifts that exact wiring, IN-PROCESS (so every
device op is visible to tracy) and BOUNDED (so the 12000-marker buffer survives).  Perf only: no PCC.
"""

from __future__ import annotations

import inspect
import os
import time

import pytest  # noqa: F401  -- pytest collects this module
import torch

import ttnn
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.inputs import AUDIO_CLIPS, build_audio_chat_inputs
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.pipeline import build_pipeline

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
HEAD = "audio_chat"
# Lifted verbatim from demo/demo_audio_chat.py.
L1_SMALL_SIZE = 24576
DEMO_TRACE_REGION_SIZE = 23887872

# ONE VARIABLE FOR ONE THING: the value the test PRINTS is the value the loop RUNS.
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- the measurement conditions, env-overridable and echoed below so a reader never has to
# guess what actually ran.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL. 0 == "ask the pipeline"; a positive value narrows it, never widens.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# HEAVY AXIS for an audio model = AUDIO FRAMES / clip length. The demo feeds 30s clips; under tracy
# every op is instrumented, so the profiled pass trims the RAW AUDIO itself to a short representative
# window via the input builder's own `seconds` knob. Small default, env-overridable.
PERF_AUDIO_SECONDS = float(os.environ.get("TT_PERF_AUDIO_SECONDS", "2.0"))

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape is how a run honours them. The demo opens a SINGLE device, so 1x1 is the default
# and a bare manual run behaves exactly as the demo does.
from models.experimental.perf_automation.agent.perf_adapter import (  # noqa: E402
    resolve_batch,
    resolve_mesh_shape,
)

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The demo SELF-OPENS its device, so this test lifts that exact open call rather than taking a
# `device` fixture (a fixture would profile a different topology than the demo builds for).
_DEV_KWARGS = {"l1_small_size": L1_SMALL_SIZE, "trace_region_size": DEMO_TRACE_REGION_SIZE}
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    _DEV_KWARGS["trace_region_size"] = max(
        DEMO_TRACE_REGION_SIZE, int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))
    )
    _DEV_KWARGS["num_command_queues"] = 1
# The demo sets no fabric_config, and the resolved mesh is 1x1 by default; nothing to gate.


def _supported_kwargs(fn, **kwargs) -> dict:
    """Keep only the kwargs `fn` actually accepts (it may take **kwargs, in which case: all)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _open_device():
    """Open EXACTLY as the demo does (single device), widening only if the tool planned a mesh."""
    rows, cols = _MESH_SHAPE
    if rows * cols > 1:
        return ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **_DEV_KWARGS), True
    return ttnn.open_device(device_id=0, **_DEV_KWARGS), False


def _close_device(dev, is_mesh: bool) -> None:
    if dev is None:
        return
    try:
        ttnn.close_mesh_device(dev) if is_mesh else ttnn.close_device(dev)
    except Exception as _ce:  # noqa: BLE001 -- teardown must not mask the measurement
        print("DEVICE_CLOSE_WARN=%r" % (_ce,), flush=True)


def _build_pipe(dev):
    """The module-level factory emit-e2e emitted, with the demo's build args.

    `layers` -- NOT `n_layers` -- is this builder's depth kwarg (build_pipeline filters on
    {batch_size, prefill_capacity, kv_capacity, layers} and silently drops anything else, so a
    misspelled name means every profile builds all 32 layers). PERF_LAYERS is None for "all".
    """
    return build_pipeline(dev, layers=PERF_LAYERS)


def _pipeline_batch(pipe) -> int:
    """The batch the PIPELINE was built for. TT_PERF_BATCH may only narrow it, never widen it."""
    n = None
    try:
        v = resolve_batch(pipe, PERF_BATCH)
        if isinstance(v, int) and v > 0:
            n = v
    except Exception:  # noqa: BLE001 -- fall through to the attribute scan
        pass
    if n is None:
        for attr in ("max_batch_size", "batch_size", "batch", "B"):
            v = getattr(pipe, attr, None)
            if isinstance(v, int) and v > 0:
                n = v
                break
    if n is None:
        n = len(AUDIO_CLIPS) or 1
    if PERF_BATCH > 0:
        n = min(n, PERF_BATCH)
    # The input builder only knows the clips it ships with.
    return max(1, min(n, len(AUDIO_CLIPS) or 1))


def _build_batch(pipe):
    """The demo's real audio batch, sized by the PIPELINE and trimmed on the audio-length axis."""
    n = _pipeline_batch(pipe)
    kwargs = {"n": n, "batch_size": n, "seconds": PERF_AUDIO_SECONDS}
    batch = build_audio_chat_inputs(**_supported_kwargs(build_audio_chat_inputs, **kwargs))
    print("PERF_STREAMS=%d" % getattr(batch, "batch_size", n), flush=True)
    print("PERF_AUDIO_SECONDS=%.3f" % PERF_AUDIO_SECONDS, flush=True)
    return batch


def _prompt_ids_exact(n_tokens: int) -> torch.Tensor:
    """EXACTLY `n_tokens` prompt ids from the tool's own helper -- not a hand-written sentence."""
    ids = None
    try:
        from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl
        from models.tt_transformers.demo.voxtral_mini_3b_2507.tt import inputs as _inp

        ids = prompt_ids_for_isl(getattr(_inp, "tokenizer", None) or _inp.get_tokenizer(), n_tokens)
    except Exception as _pe:  # noqa: BLE001 -- ids-only fallback still yields the exact ISL
        print("PERF_PROMPT_IDS_FALLBACK=%r" % (_pe,), flush=True)
    if ids is None:
        ids = torch.arange(1, n_tokens + 1, dtype=torch.long)
    if not torch.is_tensor(ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
    return ids.reshape(1, -1).to(torch.long)


def test_audio_chat_perf():
    device, _is_mesh = _open_device()
    try:

        def _eager_forward():
            counter = [0]
            _orig = []

            def _draining(fn):
                def inner(*a, **k):
                    r = fn(*a, **k)
                    counter[0] += 1
                    if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                        try:
                            ttnn.ReadDeviceProfiler(device)  # 'device' = mesh_device on multi-chip
                        except Exception:
                            pass
                    return r

                return inner

            print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            pipe = _build_pipe(device)
            batch = _build_batch(pipe)

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                # BOUNDED: PERF_OSL_TOKENS decode steps -- the same value the test declares above.
                out = pipe.run_audio_chat(batch, max_new_tokens=PERF_OSL_TOKENS)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None  # perf only — NO PCC
            # --- per-stage marks (injected) ---------------------------------------------------
            # Runs HERE, at the end of the function that built the pipeline, because that object is a LOCAL of
            # this scope: an earlier version copied the test's own PipelineStageAdapter(...) arguments into the
            # profiling branch and raised NameError, since the generator had defined them inside another
            # function. Handed locals() rather than a name, so nothing depends on how the test spells things.
            print("STAGE_MARKS_ENTER", flush=True)
            try:
                from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

                print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals(), device), flush=True)
            except Exception as _tt_e2:  # noqa: BLE001
                print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                # The RESIDENT, STAGE-EXPOSING pipeline object: the module-level factory emit-e2e
                # emitted, carrying PIPELINE_STAGES = ["encode", "prefill", "decode"] plus each
                # stage's <stage>_trace_inputs / _trace_setup / _trace_step hooks. Every stage
                # derives its OWN inputs from the captured goldens, so nothing host-side is needed
                # inside the captured region.
                from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.pipeline import (
                    build_pipeline as _build,
                )

                return _build(dev, layers=PERF_LAYERS)

            # ISL: EXACTLY PERF_ISL_TOKENS tokens from the tool's own builder -- not an example
            # sentence, so the measurement condition is the tool's choice and is recorded below.
            _prompt_ids = _prompt_ids_exact(PERF_ISL_TOKENS)
            print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
            # traced. Falls back to the single decode contract for pipelines that expose only
            # decode_step.
            measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), device)

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
        #   FULL-PIPELINE GATE (no tracy, FULL depth): needs exactly ONE whole-model latency.
        #     Running both builds the model TWICE at full depth on one device -- the second build has
        #     no memory left for its KV cache and dies before any marker is printed.
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
            # fidelity only.
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
        _close_device(device, _is_mesh)
