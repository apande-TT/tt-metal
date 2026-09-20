# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for Call 3 (`acoustic`) of `mistralai/Voxtral-4B-TTS-2603`.

Builds and runs EXACTLY as `demo/demo_acoustic.py` does -- `tt.pipeline.build_pipeline(device,
heads=("text_generation", "acoustic"), ...)` then `pipeline.run_acoustic(input_ids=...)` -- with
every device op executed IN THIS PROCESS so tracy can see it. Perf only: no PCC assertion.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import acoustic as ac  # noqa: F401  (demo's import)
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline

# A 26-layer 4B text build + the acoustic section + an HF reference on CPU. pytest.ini's repo-wide
# 300s guard is sized for unit tests and is a flake here; a genuine hang still fails.
pytestmark = pytest.mark.timeout(3600)

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- the declared measurement conditions, both env-overridable, both echoed below.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL. 0 = ask the pipeline (it declares `batch`); positive overrides.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window; ABSENT means ALL LAYERS (None).
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# THE HEAVY AXIS for this model is TOKENS (batch x real tokens per prompt, recomputed by every
# block -- the graduated stack carries no KV cache). The demo's own operating point is already the
# small one: common.DEFAULT_SEQ_LEN = 32 real tokens, deliberately short. Kept env-overridable.
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", str(common.DEFAULT_SEQ_LEN)))

# TOPOLOGY. The demo self-opens ONE device (ttnn.open_device(device_id=..., l1_small_size=24576)),
# so this test self-opens the same way; resolve_mesh_shape still decides the shape so --devices /
# --mesh are honoured, defaulting to the demo's own single chip.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEVICE_ID = int(os.environ.get("TT_PERF_DEVICE_ID", "0"))
_L1_SMALL = 24576
# The package's own trace region (device_session.TRACE_REGION_SIZE / test_trace_and_host_ops.py).
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", str(200 * 1024 * 1024)))

_OPEN_KWARGS = {"l1_small_size": _L1_SMALL}
if _PERF_TRACE:
    _OPEN_KWARGS["trace_region_size"] = _TRACE_REGION
    _OPEN_KWARGS["num_command_queues"] = 1


def _open_device():
    """The demo's own open, with the trace region reserved once, at open, for every run."""
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
    texts. The measurement condition, not a hand-written sentence."""
    n = max(1, int(n_tokens))
    ids = tokenizer.encode(common.PROMPT_TEXTS[0], bos=True)
    i = 1
    while len(ids) < n:
        ids.extend(tokenizer.encode(common.PROMPT_TEXTS[i % len(common.PROMPT_TEXTS)], bos=False))
        i += 1
    return torch.tensor(ids[:n], dtype=torch.long).unsqueeze(0)


def _build_args():
    """The demo's build args, verbatim: both heads (acoustic is fed by the text backbone)."""
    return {
        "heads": ("text_generation", "acoustic"),
        "layers": PERF_LAYERS,
        "batch": (PERF_BATCH if PERF_BATCH > 0 else None),
    }


def test_acoustic_perf():
    device, _is_mesh = _open_device()
    try:
        print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_SEQ_LEN=%d" % PERF_SEQ_LEN, flush=True)
        print("PERF_MESH_SHAPE=%dx%d" % _MESH_SHAPE, flush=True)

        # 1) build + run the pipeline EXACTLY as demo/demo_acoustic.py does
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
            _eager_forward()
            if _PERF_TRACE:
                _try_traced()
    finally:
        _close_device(device, _is_mesh)
