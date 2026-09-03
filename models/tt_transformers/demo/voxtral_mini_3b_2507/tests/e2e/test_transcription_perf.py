# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PERFORMANCE test for the `transcription` pipeline of mistralai/Voxtral-Mini-3B-2507.

Same wiring as `demo/demo_transcription.py` -- `tt/inputs.build_transcription_inputs(...)` +
`tt/pipeline.build_pipeline(device)` + `pipe.run_transcription(batch, max_new_tokens=...)` --
run entirely IN-PROCESS so every device op is visible to tracy.  The demo's HF golden and its
`--compare-hf` PCC gate are DROPPED: this is perf only.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.inputs import (
    AUDIO_TOKEN_ID,
    BEGIN_AUDIO_ID,
    BOS_ID,
    INST_END_ID,
    INST_ID,
    N_AUDIO_TOKENS_PER_CHUNK,
    build_transcription_inputs,
    get_tokenizer,
)
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.pipeline import build_pipeline

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS.  ISL for this head has a STRUCTURAL FLOOR: every
# transcription prompt is `<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 lang:<xx> [/INST]`, so the
# 3-token prefix + 375 audio placeholders + the language run + [/INST] are always present.
# The language run is grown to hit PERF_ISL_TOKENS exactly when the request is ABOVE that
# floor; the ACTUAL prompt length is what gets printed, so the conditions are never guessed.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL: 0 = ask the pipeline (VoxtralPipeline declares DECODE_BATCH=8 as
# `self.B`); a positive value overrides for sweeping without rebuilding the demo.  The clip
# count fed below is DERIVED from that batch -- there is no separate stream/clip cap.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH.  A POSITIVE TT_PERF_LAYERS caps the profiled depth so the marker buffer does not
# overflow; ABSENT means ALL LAYERS -- `layers=None` is this builder's own all-layers value
# (see tt/pipeline._profiling_depth).  Never a numeric default, never 0.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY.  demo/demo_transcription.py opens ONE device (`ttnn.open_device(device_id=0, ...)`),
# a genuine single-device pipeline, so this test keeps the repo's `device`/`device_params`
# fixture and resolve_mesh_shape's default is the source's own 1x1 -- which is still what lets
# --devices/--mesh reshape the run.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# `l1_small_size` matches the demo verbatim (conv1d/halo scratch of the audio tower).
_DEV_PARAMS = {"l1_small_size": 24576}
# The demo sets NO fabric_config (single-chip open), so none is gated on _MESH_SHAPE here.
if _PERF_TRACE:
    # Sized for the LARGEST traced stage (prefill: pinned C=512 x the LM stack); the demo's
    # 23887872 covers only the decode step.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "94371840"))
    _DEV_PARAMS["num_command_queues"] = 1

HEAD = "transcription"
# `<s> [INST] [BEGIN_AUDIO]` + [AUDIO]*375 + `lang:xx` + `[/INST]`
_AUDIO_START = 3
_DEFAULT_LANGUAGE = "en"

_PIPE = {}
_INPUTS = {}


def _language_for_isl(target: int) -> str:
    """Grow the frozen language run until the WHOLE prompt is `target` tokens (floor permitting).

    The heavy axis of this model is TOKENS, and the token count comes from the RAW PROMPT, so the
    prompt is sized here rather than copied at some full-length production shape.  Below the
    structural floor (~382 ids) nothing can be trimmed: the 375 [AUDIO] placeholders are pinned by
    the config (one 30 s chunk -> 1500 encoder frames / 4).
    """
    lang = _DEFAULT_LANGUAGE
    try:
        tok = get_tokenizer()
        fixed = _AUDIO_START + N_AUDIO_TOKENS_PER_CHUNK + 1  # prefix + audio run + [/INST]
        need = int(target) - fixed - len(tok.encode("lang:"))
        while len(tok.encode(lang)) < need:
            lang = lang + " en"
    except Exception as exc:  # noqa: BLE001 -- ISL sizing must never fail the measurement
        print("PERF_ISL_PAD_SKIPPED=%r" % (exc,), flush=True)
    return lang


def prompt_ids_for_isl(tokenizer, n_tokens: int) -> torch.Tensor:
    """The FULL transcription prompt as ids, sized to `n_tokens` (floor permitting).

    Exactly the frozen layout `tt/inputs.TRANSCRIPTION_TEMPLATE` describes, so the ids handed to
    the trace adapter are the ones the pipeline is actually built to run -- not a hand-written
    example sentence, and not a length this generator chose.
    """
    text = tokenizer.encode("lang:%s" % _language_for_isl(n_tokens))
    ids = (
        [BOS_ID, INST_ID, BEGIN_AUDIO_ID] + [AUDIO_TOKEN_ID] * N_AUDIO_TOKENS_PER_CHUNK + text + [INST_END_ID]
    )
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _get_pipe(dev):
    """Build the resident pipeline ONCE per device -- exactly as the demo builds it."""
    key = id(dev)
    if key not in _PIPE:
        # None == every layer; never 0.
        opts = {"layers": PERF_LAYERS}
        if PERF_BATCH > 0:
            opts["batch_size"] = PERF_BATCH
        _PIPE.clear()
        _PIPE[key] = build_pipeline(dev, **opts)
    return _PIPE[key]


def _batch_inputs(n: int):
    """The demo's own input builder, at the batch the PIPELINE declares.

    `n` streams == `n` clips: the demo's `_resolve_clips` cycles the 8 built-in clips to fit the
    batch, and `build_transcription_inputs` does the same internally, so the clip count is
    DERIVED from the pipeline's batch and never capped independently.
    """
    key = int(n)
    if key not in _INPUTS:
        _INPUTS[key] = build_transcription_inputs(language=_language_for_isl(PERF_ISL_TOKENS), n=key)
    return _INPUTS[key]


def _patch_trace_inputs(pipe, batch):
    """Feed the per-stage trace hooks from the REAL transcription input builder.

    The pipeline's own `*_trace_inputs()` torch.load `_captured/voxtral_encoder/{args,output}.pt`,
    which this tree does not ship (only manifest.json), so every stage would raise
    FileNotFoundError and the adapter would drop all three.  The shapes handed back here are
    identical to the ones those hooks describe -- real mel, real ids/audio_start/n_audio/prompt_len
    from the input builder, and audio embeds at the projector's own output shape (values are
    irrelevant: this is perf, not PCC).  Everything is built HOST-SIDE and pre-uploaded by the
    pipeline's unmodified `*_trace_setup`, i.e. OUTSIDE the captured region.
    """
    mel = batch.input_features[0:1].float()
    n_audio = int(batch.n_audio_tokens)
    audio_embeds = torch.randn(pipe.B, n_audio, pipe.hidden, dtype=torch.float32) * 0.02
    ids = batch.input_ids[: pipe.B]
    stage_inputs = {
        "input_ids": ids,
        "audio_embeds": audio_embeds,
        "audio_start": int(batch.audio_start),
        "n_audio_tokens": n_audio,
        "prompt_len": int(batch.prompt_len),
    }
    pipe.encode_trace_inputs = lambda: mel
    pipe.prefill_trace_inputs = lambda: dict(stage_inputs)
    pipe.decode_trace_inputs = lambda: dict(stage_inputs)
    return stage_inputs


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_transcription_perf(device_params, device):
    print("PERF_MESH_SHAPE=%dx%d" % (_MESH_SHAPE[0], _MESH_SHAPE[1]), flush=True)
    print("PERF_LAYERS=%s" % (PERF_LAYERS,), flush=True)

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

        pipe = _get_pipe(device)
        n = resolve_batch(pipe, PERF_BATCH)
        batch = _batch_inputs(n)
        # OSL is bounded by the resident KV capacity: prompt_len + osl must fit KV_C.
        osl = max(1, min(PERF_OSL_TOKENS, pipe.KV_C - int(batch.prompt_len)))
        print("PERF_ISL_TOKENS=%d" % int(batch.prompt_len), flush=True)
        print("PERF_OSL_TOKENS=%d" % osl, flush=True)
        print("PERF_BATCH_STREAMS=%d" % pipe.B, flush=True)

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))
        _fw0 = time.monotonic()
        try:
            out = pipe.run_transcription(batch, max_new_tokens=osl)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        pipe = _get_pipe(device)
        batch = _batch_inputs(resolve_batch(pipe, PERF_BATCH))
        _patch_trace_inputs(pipe, batch)

        def _build_for_perf(dev):
            p = _get_pipe(dev)
            _patch_trace_inputs(p, _batch_inputs(resolve_batch(p, PERF_BATCH)))
            return p

        # The REAL prompt the builder produced is the measurement condition; prompt_ids_for_isl
        # describes the same frozen layout and is echoed so a drift between them is visible.
        _prompt_ids = batch.input_ids[0].reshape(1, -1)
        print("PERF_ISL_DECLARED=%d" % prompt_ids_for_isl(get_tokenizer(), PERF_ISL_TOKENS).shape[-1], flush=True)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_BATCH_STREAMS=%d" % pipe.B, flush=True)
        measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), device)

    def _try_traced():
        try:
            _traced_forward()
            return True
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
            import traceback

            traceback.print_exc()
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
