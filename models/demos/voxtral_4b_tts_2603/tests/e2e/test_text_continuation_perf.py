# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the `text_continuation` pipeline of `mistralai/Voxtral-4B-TTS-2603`.

Built as `demo/demo_text_continuation.py` does -- `build_pipeline(device, heads=("text_continuation",))` -- and measured through the per-stage trace contract
(`PIPELINE_STAGES` = prefill / decode / acoustic / vocode, each `<stage>_trace_setup` +
`<stage>_trace_step`), trace+1CQ. The eager path runs ONE teacher-forced `run_text_continuation`
forward over PERF_ISL_TOKENS tokens per row. Perf only: no PCC.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline

pytestmark = pytest.mark.timeout(3600)

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
_pl_prefill = (os.environ.get("TT_PERF_PREFILL_LAYERS") or "").strip()
PERF_PREFILL_LAYERS = int(_pl_prefill) if (_pl_prefill.isdigit() and int(_pl_prefill) > 0) else None
_pl_decode = (os.environ.get("TT_PERF_DECODE_LAYERS") or "").strip()
PERF_DECODE_LAYERS = int(_pl_decode) if (_pl_decode.isdigit() and int(_pl_decode) > 0) else None
_pl_acoustic = (os.environ.get("TT_PERF_ACOUSTIC_LAYERS") or "").strip()
PERF_ACOUSTIC_LAYERS = int(_pl_acoustic) if (_pl_acoustic.isdigit() and int(_pl_acoustic) > 0) else None
_pl_vocode = (os.environ.get("TT_PERF_VOCODE_LAYERS") or "").strip()
PERF_VOCODE_LAYERS = int(_pl_vocode) if (_pl_vocode.isdigit() and int(_pl_vocode) > 0) else None

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEVICE_ID = int(os.environ.get("TT_PERF_DEVICE_ID", os.environ.get("VOXTRAL_E2E_DEVICE_ID", "0")))
# The demo's own trace region (demo_text_continuation.py / device_session.TRACE_REGION_SIZE).
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", str(200 * 1024 * 1024)))

_OPEN_KWARGS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _OPEN_KWARGS["trace_region_size"] = _TRACE_REGION
    _OPEN_KWARGS["num_command_queues"] = 1


def _open_device():
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
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl as _p

    ids = _p(tokenizer, n_tokens)
    return ids.reshape(1, -1)


def _build_args(hf_model):
    # kv_capacity is left to the build's own default (`default_tts_kv_capacity`): the stage hooks
    # feed the pipeline's own voiced prompt, pinned at the prefill trace capacity, and the decode
    # stage needs room past it.
    return {
        "model": hf_model,
        "heads": ("text_continuation",),
        "layers": PERF_LAYERS,
        "prefill_layers": PERF_PREFILL_LAYERS,
        "decode_layers": PERF_DECODE_LAYERS,
        "acoustic_layers": PERF_ACOUSTIC_LAYERS,
        "vocode_layers": PERF_VOCODE_LAYERS,
        "batch": (PERF_BATCH if PERF_BATCH > 0 else None),
    }


def test_text_continuation_perf():
    device, _is_mesh = _open_device()
    try:
        print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_MESH_SHAPE=%dx%d" % _MESH_SHAPE, flush=True)
        hf_model = common.load_reference_model()

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

            pipe = build_pipeline(device, **_build_args(hf_model))
            print("STAGE_MARKS_ENTER", flush=True)
            try:
                from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

                print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals(), device), flush=True)
            except Exception as _tt_e2:  # noqa: BLE001
                print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)
            _rows = resolve_batch(pipe, PERF_BATCH)
            _row = prompt_ids_for_isl(common.load_tokenizer(), PERF_ISL_TOKENS)
            input_ids = _row.repeat(_rows, 1)
            print("PERF_BATCH_ROWS=%d" % _rows, flush=True)

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                out = pipe.run_text_continuation(input_ids=input_ids)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None  # perf only -- NO PCC

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            _prompt_ids = prompt_ids_for_isl(common.load_tokenizer(), PERF_ISL_TOKENS)

            def _build_for_perf(dev):
                pipe = build_pipeline(dev, **_build_args(hf_model))
                # The text_continuation head's ONE forward (run_text_continuation's device part) as
                # the traced stage: ids are uploaded once in setup, the step reads only resident
                # tensors, so the captured region is host-free.
                _rows = resolve_batch(pipe, PERF_BATCH)
                _buf = {}

                def continuation_trace_inputs():
                    return _prompt_ids.reshape(1, -1).repeat(_rows, 1)

                def continuation_trace_setup(inputs):
                    _buf["ids"] = pipe.prepare_prompt(inputs)
                    _buf["rows"] = int(inputs.shape[0]) * int(inputs.shape[1])
                    return _buf

                def continuation_trace_step():
                    picks, _logits = pipe.continuation.score(_buf["ids"])
                    return picks

                def continuation_trace_items():
                    return _buf.get("rows", _rows * int(_prompt_ids.shape[-1]))

                pipe.continuation_trace_inputs = continuation_trace_inputs
                pipe.continuation_trace_setup = continuation_trace_setup
                pipe.continuation_trace_step = continuation_trace_step
                pipe.continuation_trace_items = continuation_trace_items
                pipe.PIPELINE_STAGES = ["continuation"]
                # One teacher-forced forward over the whole prompt is this pipeline's unit of work.
                pipe.PIPELINE_UNIT = "step"
                return pipe

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
