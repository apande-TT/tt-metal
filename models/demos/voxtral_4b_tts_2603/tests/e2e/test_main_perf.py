# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the `main` pipeline of `mistralai/Voxtral-4B-TTS-2603`.

Built and driven EXACTLY as the correctness test / demo does --
`tt.pipeline.build_pipeline(device, heads=("text_generation", "acoustic"), ...)` then
`pipeline.run_acoustic(input_ids=...)`, chaining the text backbone into the acoustic section on
the DEVICE -- with every device op executed IN THIS PROCESS so tracy can see it.

Perf only: the torch/HF reference build and every PCC comparison from the source test are DROPPED.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import acoustic as ac  # noqa: F401  (the demo's import)
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline

# A 26-layer 4B text build plus the acoustic section. pytest.ini's repo-wide 300s guard is sized
# for unit tests and is a flake here; a bound is still enforced, so a genuine hang fails.
pytestmark = pytest.mark.timeout(3600)

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- the declared measurement conditions, both env-overridable, both echoed below.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL. 0 = ask the pipeline (it declares `batch`); positive overrides.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. A POSITIVE value caps the profiled window; ABSENT means ALL LAYERS (None). This model runs
# repeating block stacks behind BOTH PIPELINE_STAGES entries, so each stage gets its own override
# and `layers` stays the default for any stack a per-stage argument does not name.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
_pl_prefill = (os.environ.get("TT_PERF_PREFILL_LAYERS") or "").strip()
PERF_PREFILL_LAYERS = int(_pl_prefill) if (_pl_prefill.isdigit() and int(_pl_prefill) > 0) else None
_pl_decode = (os.environ.get("TT_PERF_DECODE_LAYERS") or "").strip()
PERF_DECODE_LAYERS = int(_pl_decode) if (_pl_decode.isdigit() and int(_pl_decode) > 0) else None

# THE HEAVY AXIS for this model is TOKENS -- batch x real tokens per prompt, recomputed by every
# block (the graduated stack carries no KV cache), and the acoustic section is fed the resulting
# hidden state, so the same axis sets its frame count too. The package's own small operating point
# is `common.DEFAULT_SEQ_LEN` = 32 real tokens, far below `max_position_embeddings` (128000).
# Kept small and env-overridable.
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", str(common.DEFAULT_SEQ_LEN)))

# TOPOLOGY. The source self-opens ONE device (ttnn.open_device(device_id=..., l1_small_size=24576)),
# so this test self-opens the same way; resolve_mesh_shape still decides the shape so --devices /
# --mesh are honoured, defaulting to the source's own single chip.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEVICE_ID = int(os.environ.get("TT_PERF_DEVICE_ID", os.environ.get("VOXTRAL_E2E_DEVICE_ID", "0")))
_L1_SMALL = 24576
# The package's own trace region (device_session.TRACE_REGION_SIZE / test_trace_and_host_ops.py).
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", str(200 * 1024 * 1024)))

_OPEN_KWARGS = {"l1_small_size": _L1_SMALL}
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, with a single command queue.
    _OPEN_KWARGS["trace_region_size"] = _TRACE_REGION
    _OPEN_KWARGS["num_command_queues"] = 1


def _open_device():
    """The source's own open, with the trace region reserved once, at open, for every run."""
    rows, cols = _MESH_SHAPE
    if rows * cols > 1:
        return ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **_OPEN_KWARGS), True
    return ttnn.open_device(device_id=_DEVICE_ID, **_OPEN_KWARGS), False


def _close_device(dev, is_mesh):
    if is_mesh:
        ttnn.close_mesh_device(dev)
    else:
        ttnn.close_device(dev)


def prompt_ids_for_isl(tokenizer, n_tokens: int):
    """EXACTLY `n_tokens` real ids from the package's own tekken tokenizer over its own prompt
    texts -- the measurement condition, not a hand-written sentence."""
    n = max(1, int(n_tokens))
    ids = tokenizer.encode(common.PROMPT_TEXTS[0], bos=True)
    i = 1
    while len(ids) < n:
        ids.extend(tokenizer.encode(common.PROMPT_TEXTS[i % len(common.PROMPT_TEXTS)], bos=False))
        i += 1
    return torch.tensor(ids[:n], dtype=torch.long).unsqueeze(0)


def _build_args():
    """The source's build args: both heads (the acoustic section is fed by the text backbone)."""
    return {
        "heads": ("text_generation", "acoustic"),
        "layers": PERF_LAYERS,
        "prefill_layers": PERF_PREFILL_LAYERS,
        "decode_layers": PERF_DECODE_LAYERS,
        "batch": (PERF_BATCH if PERF_BATCH > 0 else None),
    }


def test_main_perf():
    device, _is_mesh = _open_device()
    try:
        print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_SEQ_LEN=%d" % PERF_SEQ_LEN, flush=True)
        print("PERF_MESH_SHAPE=%dx%d" % _MESH_SHAPE, flush=True)

        # 1) build + run the pipeline EXACTLY as the source does
        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY
        #    ttnn operation (type 'FastOperation') across ttnn + its op submodules.
        def _eager_forward():
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

            pipeline = build_pipeline(device, **_build_args())
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
            # The row count comes from the pipeline, never from a literal here.
            batch = resolve_batch(pipeline, PERF_BATCH)
            input_ids, _texts = common.build_batch_inputs(batch=batch, seq_len=PERF_SEQ_LEN)
            print("PERF_BATCH_ROWS=%d" % int(input_ids.shape[0]), flush=True)

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                out = pipeline.run_acoustic(input_ids=input_ids)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None and out["semantic_logits"] is not None  # perf only — NO PCC

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline

                return build_pipeline(dev, **_build_args())

            _prompt_ids = prompt_ids_for_isl(common.load_tokenizer(), PERF_ISL_TOKENS)
            print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), device)

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
    finally:
        _close_device(device, _is_mesh)
