# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the `text_to_speech` pipeline of `mistralai/Voxtral-4B-TTS-2603` (no PCC).

Built exactly as `demo/demo_text_to_speech.py` does: the device is self-opened with the demo's own
parameters, the reference model is loaded once, the voiced speech request is the model's own
`[AUDIO]`-placeholder layout with the default preset voice, and the pipeline comes from
`pipeline.build_pipeline(device, model=..., heads=("text_to_speech",), batch=..., kv_capacity=...)`.

The heavy axis of a TTS model is AUDIO FRAMES (one decode step + one acoustic frame each), so the
frame horizon is capped at PERF_OSL_TOKENS (the traced path) / _EAGER_OSL_TOKENS (the op-wrapped
eager path) instead of the demo's 256-frame safety cap.

Trace path: `measure_adapter(PipelineStageAdapter(...))` traces every entry of `PIPELINE_STAGES`
(prefill / decode / acoustic / vocode) through the pipeline's own `<stage>_trace_inputs` ->
`<stage>_trace_setup` -> `<stage>_trace_step` hooks, trace + 1 CQ.
"""
from __future__ import annotations

import gc
import os
import time

import pytest

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline
from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline

pytestmark = pytest.mark.timeout(3600)

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
# Per-stack overrides build_pipeline takes (None = the global PERF_LAYERS / every layer).
_pl_prefill = (os.environ.get("TT_PERF_PREFILL_LAYERS") or "").strip()
PERF_PREFILL_LAYERS = int(_pl_prefill) if (_pl_prefill.isdigit() and int(_pl_prefill) > 0) else None
_pl_decode = (os.environ.get("TT_PERF_DECODE_LAYERS") or "").strip()
PERF_DECODE_LAYERS = int(_pl_decode) if (_pl_decode.isdigit() and int(_pl_decode) > 0) else None
_pl_acoustic = (os.environ.get("TT_PERF_ACOUSTIC_LAYERS") or "").strip()
PERF_ACOUSTIC_LAYERS = int(_pl_acoustic) if (_pl_acoustic.isdigit() and int(_pl_acoustic) > 0) else None
_pl_vocode = (os.environ.get("TT_PERF_VOCODE_LAYERS") or "").strip()
PERF_VOCODE_LAYERS = int(_pl_vocode) if (_pl_vocode.isdigit() and int(_pl_vocode) > 0) else None

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402

# The demo opens ONE device (`ttnn.open_device(device_id=...)`): a 1x1 topology.
_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEVICE_ID = int(os.environ.get("TT_PERF_DEVICE_ID", os.environ.get("VOXTRAL_DEVICE_ID", "0")))
# The demo's own device configuration (demo_text_to_speech.py): l1_small 24576, a 200 MB trace
# region and one command queue.
_OPEN_KWARGS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _OPEN_KWARGS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", str(200 * 1024 * 1024)))
    _OPEN_KWARGS["num_command_queues"] = 1
else:
    _OPEN_KWARGS["trace_region_size"] = 200 * 1024 * 1024
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

    return _p(tokenizer, n_tokens).reshape(-1)


def _speech_request(rows: int):
    """The demo's speech request: SPEECH_TEXTS (repeated up to `rows`, as the demo fills a short
    request) in the model's voiced layout with the demo's default preset voice."""
    texts = list(common.SPEECH_TEXTS[:rows])
    if len(texts) < rows:
        texts = (texts * rows)[:rows]
    return common.build_voice_prompt(texts, common.DEFAULT_VOICE)


def test_text_to_speech_perf():
    common.use_all_cpu_threads()
    device, _is_mesh = _open_device()
    try:
        print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_MESH_SHAPE=%dx%d" % _MESH_SHAPE, flush=True)
        hf_model = common.load_reference_model()
        # Every speech row tokenizes to one common width, so ONE row sizes the KV cache the way the
        # demo does (`tts_kv_capacity(prompt_len, max_frames)`), at the capped frame horizon.
        _probe_ids, _, _ = _speech_request(1)
        _prompt_len = int(_probe_ids.shape[-1])
        _frames = max(PERF_OSL_TOKENS, _EAGER_OSL_TOKENS)
        print("TTS_REQUEST_TOKENS=%d  max_frames=%d" % (_prompt_len, _frames), flush=True)

        def _build_args():
            return {
                "model": hf_model,
                "heads": ("text_to_speech",),
                "layers": PERF_LAYERS,
                "prefill_layers": PERF_PREFILL_LAYERS,
                "decode_layers": PERF_DECODE_LAYERS,
                "acoustic_layers": PERF_ACOUSTIC_LAYERS,
                "vocode_layers": PERF_VOCODE_LAYERS,
                "batch": (PERF_BATCH if PERF_BATCH > 0 else None),
                "kv_capacity": pipeline.tts_kv_capacity(_prompt_len, _frames),
            }

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

            pipe = build_pipeline(device, **_build_args())
            _rows = resolve_batch(pipe, PERF_BATCH)
            input_ids, audio_mask, voice_embedding = _speech_request(_rows)
            voice = pipe.stage_voice(audio_mask, voice_embedding, input_ids=input_ids)
            print("PERF_BATCH_ROWS=%d  prompt_tokens=%d" % (_rows, int(input_ids.shape[1])), flush=True)

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                out = pipe.run_text_to_speech(input_ids=input_ids, max_frames=_EAGER_OSL_TOKENS, voice=voice)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None and out["frames_decoded"] > 0  # perf only -- NO PCC
            print("frames decoded: %d  (%s)" % (out["frames_decoded"], out["stop_reason"]), flush=True)
            del out, voice, pipe
            gc.collect()

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                # The RESIDENT pipeline object: it carries PIPELINE_STAGES and every
                # <stage>_trace_inputs / _trace_setup / _trace_step hook.
                pipe = build_pipeline(dev, **_build_args())
                # THE RECURRING STAGE. `decode` is the autoregressive step the frame loop repeats --
                # one call advances every row by one audio frame -- so it states ONE item per call,
                # the adapter's own decode-step convention. Left at B it states no recurring stage,
                # and the headline degrades to a whole-pipeline sum. Per audio frame the real cost
                # is TRACE_STAGE_MS[decode] + TRACE_STAGE_MS[acoustic].
                pipe.decode_trace_items = lambda: 1
                return pipe

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
