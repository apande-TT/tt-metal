# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PERFORMANCE test for the `main` pipeline of mistralai/Voxtral-Mini-3B-2507.

Same wiring as `tests/e2e/test_e2e_pipeline.py` -- `tt/inputs.build_*_inputs(...)` +
`tt/pipeline.build_pipeline(device)` + `pipe.run_head(batch, max_new_tokens=...)` over BOTH
task heads (`audio_chat`, `transcription`) -- run entirely IN-PROCESS so every device op is
visible to tracy.  The e2e gate's reference/torch model (`tt/reference.hf_reference`), its
graduated-module inventory, its static/runtime nativeness gates and EVERY PCC comparison are
DROPPED: this is perf only.
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
    DEFAULT_INSTRUCTION,
    INST_END_ID,
    INST_ID,
    N_AUDIO_TOKENS_PER_CHUNK,
    build_audio_chat_inputs,
    build_transcription_inputs,
    get_tokenizer,
)
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.pipeline import PIPELINE_STAGES, build_pipeline

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS.  ISL for this model has a STRUCTURAL FLOOR: every
# prompt is `<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 <text> [/INST]`, so the 3-token prefix + 375
# audio placeholders + the text run + [/INST] are always present.  The text run is grown to hit
# PERF_ISL_TOKENS exactly when the request is ABOVE that floor; the ACTUAL prompt length is what
# gets printed, so the conditions are never guessed from the request.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL: 0 = ask the pipeline (VoxtralPipeline declares DECODE_BATCH=8 as
# `self.B`); a positive value overrides for sweeping without rebuilding the demo.  The clip count
# fed below is DERIVED from that batch -- there is no separate stream/clip cap.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH.  A POSITIVE TT_PERF_LAYERS caps the profiled depth so the marker buffer does not
# overflow; ABSENT means ALL LAYERS -- `layers=None` is this builder's own all-layers value
# (see tt/pipeline._profiling_depth).  Never a numeric default, never 0.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
# PER-STAGE DEPTH.  This model runs 2 repeating block stacks across the stages it declares in
# PIPELINE_STAGES (encode / prefill / decode), so each stage carries its own cap with the SAME
# None-means-all-layers contract.  `layers` above stays the default for any stack a per-stage
# argument does not name.
_pl_encode = (os.environ.get("TT_PERF_ENCODE_LAYERS") or "").strip()
PERF_ENCODE_LAYERS = int(_pl_encode) if (_pl_encode.isdigit() and int(_pl_encode) > 0) else None
_pl_prefill = (os.environ.get("TT_PERF_PREFILL_LAYERS") or "").strip()
PERF_PREFILL_LAYERS = int(_pl_prefill) if (_pl_prefill.isdigit() and int(_pl_prefill) > 0) else None
_pl_decode = (os.environ.get("TT_PERF_DECODE_LAYERS") or "").strip()
PERF_DECODE_LAYERS = int(_pl_decode) if (_pl_decode.isdigit() and int(_pl_decode) > 0) else None
# SELF-CAP UNDER TRACY ONLY.  A tracy capture of every block x every decode token overflows the
# marker buffer ("Too many source locations") and the run then livelocks in serialisation, so when
# the profiler is armed and NO depth arrived from the harness we fall back to a small window.
# Gated on TT_METAL_DEVICE_PROFILER so the full-depth PCC / trace+1cq / op-signature gates -- which
# ask for all layers by clearing these vars -- are untouched, and skipped outright when the harness
# explicitly asks for all layers.
if (
    os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    and os.environ.get("PERF_MCP_FORCE_ALL_LAYERS") != "1"
    and PERF_LAYERS is None
    and PERF_ENCODE_LAYERS is None
    and PERF_PREFILL_LAYERS is None
    and PERF_DECODE_LAYERS is None
):
    _fallback = (os.environ.get("PERF_MCP_PROFILE_FALLBACK_LAYERS") or "2").strip()
    _fallback = int(_fallback) if (_fallback.isdigit() and int(_fallback) > 0) else 2
    PERF_LAYERS = _fallback
    PERF_ENCODE_LAYERS = _fallback
    PERF_PREFILL_LAYERS = _fallback
    PERF_DECODE_LAYERS = _fallback
    print("PERF self-cap: no depth arrived under tracy, capping to %d layers" % _fallback, flush=True)

# TOPOLOGY.  tests/e2e/test_e2e_pipeline.py takes the repo's `device`/`device_params` fixture
# (a genuine single-device pipeline -- build_pipeline(device) with no MeshShape anywhere), so
# this test keeps that fixture and resolve_mesh_shape's default is the source's own 1x1 -- which
# is still what lets --devices/--mesh reshape the run.
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# `l1_small_size` matches the source verbatim (conv1d/halo scratch of the audio tower).
_DEV_PARAMS = {"l1_small_size": 24576}
# The source sets NO fabric_config (single-chip open), so none is gated on _MESH_SHAPE here.
if _PERF_TRACE:
    # Sized for the LARGEST traced stage (prefill: pinned C=512 x the LM stack); the source's
    # 23887872 covers only the decode step.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "94371840"))
    _DEV_PARAMS["num_command_queues"] = 1

# `<s> [INST] [BEGIN_AUDIO]` + [AUDIO]*375 + <text> + `[/INST]`
_AUDIO_START = 3
_PROMPT_FIXED_TOKENS = _AUDIO_START + N_AUDIO_TOKENS_PER_CHUNK + 1
_DEFAULT_LANGUAGE = "en"

# BOTH heads, in the source's own order.  The first one is the representative stream for the
# trace adapter; the eager forward runs the whole main pipeline over both.
HEADS = ("audio_chat", "transcription")

_PIPE = {}
_INPUTS = {}


def _pad_run(base: str, target: int, filler: str) -> str:
    """Grow a frozen text run until the WHOLE prompt is `target` tokens (floor permitting).

    The heavy axis of this model is TOKENS, and the token count comes from the RAW PROMPT, so the
    prompt is sized here rather than copied at some full-length production shape.  Below the
    structural floor (~379 ids) nothing can be trimmed: the 375 [AUDIO] placeholders are pinned by
    the config (one 30 s chunk -> 1500 encoder frames / 4).
    """
    run = base
    need = int(target) - _PROMPT_FIXED_TOKENS
    if need <= 0:
        return run
    try:
        tok = get_tokenizer()
        while len(tok.encode(run)) < need:
            run = run + filler
    except Exception as exc:  # noqa: BLE001 -- ISL sizing must never fail the measurement
        print("PERF_ISL_PAD_SKIPPED=%r" % (exc,), flush=True)
    return run


def _instruction_for_isl(target: int) -> str:
    return _pad_run(DEFAULT_INSTRUCTION, target, " please")


def _language_for_isl(target: int) -> str:
    return _pad_run(_DEFAULT_LANGUAGE, target, " en")


def prompt_ids_for_isl(tokenizer, n_tokens: int) -> torch.Tensor:
    """The FULL audio_chat prompt as ids, sized to `n_tokens` (floor permitting).

    Exactly the frozen layout `tt/inputs.AUDIO_CHAT_TEMPLATE` describes, so the ids handed to the
    trace adapter are the ones the pipeline is actually built to run -- not a hand-written example
    sentence, and not a length this generator chose.
    """
    text = tokenizer.encode(_instruction_for_isl(n_tokens))
    ids = [BOS_ID, INST_ID, BEGIN_AUDIO_ID] + [AUDIO_TOKEN_ID] * N_AUDIO_TOKENS_PER_CHUNK + text + [INST_END_ID]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _get_pipe(dev):
    """Build the resident pipeline ONCE per device -- exactly as the source builds it."""
    key = id(dev)
    if key not in _PIPE:
        # None == every layer; never 0.  The per-stage names are the ones the tool adds to
        # build_pipeline; build_pipeline filters kwargs it does not know, so an unbuilt stage
        # cap is inert rather than fatal.
        opts = {
            "layers": PERF_LAYERS,
            "encode_layers": PERF_ENCODE_LAYERS,
            "prefill_layers": PERF_PREFILL_LAYERS,
            "decode_layers": PERF_DECODE_LAYERS,
        }
        if PERF_BATCH > 0:
            opts["batch_size"] = PERF_BATCH
        _PIPE.clear()
        _PIPE[key] = build_pipeline(dev, **opts)
    return _PIPE[key]


def _batch_inputs(head: str, n: int):
    """The source's own input builders, at the batch the PIPELINE declares.

    `n` streams == `n` clips: `build_*_inputs` cycles the built-in clips to fit the batch, so the
    clip count is DERIVED from the pipeline's batch and never capped independently.
    """
    key = (head, int(n))
    if key not in _INPUTS:
        if head == "audio_chat":
            _INPUTS[key] = build_audio_chat_inputs(instruction=_instruction_for_isl(PERF_ISL_TOKENS), n=int(n))
        else:
            _INPUTS[key] = build_transcription_inputs(language=_language_for_isl(PERF_ISL_TOKENS), n=int(n))
    return _INPUTS[key]


def _patch_trace_inputs(pipe, batch):
    """Feed the per-stage trace hooks from the REAL input builder.

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
def test_main_perf(device_params, device):
    print("PERF_MESH_SHAPE=%dx%d" % (_MESH_SHAPE[0], _MESH_SHAPE[1]), flush=True)
    print("PERF_LAYERS=%s" % (PERF_LAYERS,), flush=True)
    print(
        "PERF_STAGE_LAYERS encode=%s prefill=%s decode=%s"
        % (PERF_ENCODE_LAYERS, PERF_PREFILL_LAYERS, PERF_DECODE_LAYERS),
        flush=True,
    )
    print("PERF_PIPELINE_STAGES=%s" % (list(PIPELINE_STAGES),), flush=True)

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
        batches = [_batch_inputs(h, n) for h in HEADS]
        # OSL is bounded by the resident KV capacity: prompt_len + osl must fit KV_C.  ONE osl is
        # declared and ONE is run, for every head, so the printed unit is the executed unit.
        osl = max(1, min([PERF_OSL_TOKENS] + [pipe.KV_C - int(b.prompt_len) for b in batches]))
        print("PERF_ISL_TOKENS=%d" % int(batches[0].prompt_len), flush=True)
        print("PERF_OSL_TOKENS=%d" % osl, flush=True)
        print("PERF_BATCH_STREAMS=%d" % pipe.B, flush=True)

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        # --- per-stage marks (injected) ---------------------------------------------------
        # Runs HERE, at the end of the function that built the pipeline, because that object is a LOCAL of
        # this scope: an earlier version copied the test's own PipelineStageAdapter(...) arguments into the
        # profiling branch and raised NameError, since the generator had defined them inside another
        # function. Handed locals() rather than a name, so nothing depends on how the test spells things.
        print("STAGE_MARKS_ENTER", flush=True)
        try:
            from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

            print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals(), device, bind=_patch_trace_inputs), flush=True)
        except Exception as _tt_e2:  # noqa: BLE001
            print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))
        _fw0 = time.monotonic()
        try:
            out = None
            for head, batch in zip(HEADS, batches):
                out = pipe.run_head(batch, max_new_tokens=osl)
                assert out is not None, "%s: pipeline produced no output" % head
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
        batch = _batch_inputs(HEADS[0], resolve_batch(pipe, PERF_BATCH))
        _patch_trace_inputs(pipe, batch)

        def _build_for_perf(dev):
            # RETURNS the resident, stage-exposing pipeline object (PIPELINE_STAGES +
            # <stage>_trace_setup/_trace_step hooks), never a run result.
            p = _get_pipe(dev)
            _patch_trace_inputs(p, _batch_inputs(HEADS[0], resolve_batch(p, PERF_BATCH)))
            return p

        # The REAL prompt the builder produced is the measurement condition; prompt_ids_for_isl
        # describes the same frozen layout and is echoed so a drift between them is visible.
        _prompt_ids = batch.input_ids[0].reshape(1, -1)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_ISL_DECLARED=%d" % prompt_ids_for_isl(get_tokenizer(), PERF_ISL_TOKENS).shape[-1], flush=True)
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
