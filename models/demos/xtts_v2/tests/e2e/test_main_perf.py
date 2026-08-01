# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the 'main' pipeline of coqui/XTTS-v2 (TP=8 x DP=1 mesh).

Builds and runs the SHARED chained TTNN pipeline (tt/pipeline.py) exactly as the
correctness test (tests/e2e/test_e2e_tts.py) does -- same module-level
``build_pipeline`` factory, same SELF-OPENED 1x8 mesh + l1_small reservation, same
input contract (DVAE cond mel + 16 kHz speaker reference + text tokens) -- but with
the reference/torch model construction DROPPED, every PCC / comp_pcc comparison
DROPPED, and the work BOUNDED so the profiler's marker buffer never overflows.
Perf only.

Run: ./python_env/bin/python -m pytest models/demos/xtts_v2/tests/e2e/test_main_perf.py -s
"""
from __future__ import annotations

import os
import time

import pytest  # noqa: F401  (pytest collects this module)
import torch

import ttnn
from models.demos.xtts_v2.tt import pipeline as P

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS, echoed below so a reader never has to guess them.
# For THIS model the "input length" axis is the TEXT/PHONEME token count: it sets the GPT prefix
# (32 cond rows | text L+2 | start_audio) and therefore the sequence capacity every traced GPT
# stage runs at, i.e. it is the axis that drives op/dispatch count. The industry 128-in default
# belongs to a text LLM; for a TTS a full-length phoneme string is a correctness-stress size that
# makes every instrumented forward orders of magnitude slower, so the default here is a SMALL
# representative phoneme count (env-overridable). OSL is the generated AUDIO-CODE frame count,
# capped by PERF_MAX_NEW_TOKENS the same way (decode is measured one steady-state AR step at a time).
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "16"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", str(PERF_MAX_NEW_TOKENS)))
# Streams per program. The correctness test runs DECODE_BATCH=4 independent streams to prove the
# batch axis carries independent audio; for a perf profile that is 4x the dispatch of the same
# steady-state step, so the profiled run pins ONE stream (env-overridable) and reports batch=1.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "1"))
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
# Pass PERF_LAYERS straight to the builder: None is every builder's own all-layers value. Do NOT
# default it to a number here -- that would silently cap the full-depth gate.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape is how a run honours them. The SOURCE (tests/e2e/test_e2e_tts.py) SELF-OPENS
# ttnn.MeshShape(1, 8), so that is the default: an unset env behaves exactly as the source does.
# A pytest `device` fixture would silently disable the pipeline's TP=8 sharding (shard_active
# False), so this test lifts the source's own open call instead.
from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape  # noqa: E402

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=8)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# Lifted VERBATIM from the source's _DEV_PARAMS: the speaker encoder's convolutions are native
# ttnn.conv2d and the vocoder trunk is native ttnn.conv1d/conv_transpose2d; both run a sliding-
# window/halo gather whose sharding + config tensors allocate from the dedicated L1_SMALL pool.
# That pool is 0 B unless reserved at device open, and coming up short surfaces as a TT_FATAL
# "Out of Memory ... bank size is 0 B", not an API error. This test SELF-OPENS the mesh, so there
# is no device_params fixture.
_DEV_PARAMS = {"l1_small_size": 131072}
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    # All THREE stages (prefill / decode / vocode) are distinct traced programs and their trace
    # buffers together do not fit the stock 23 MB reservation -- mesh_trace.cpp asserts
    # get_trace_buffers_size() <= trace_region_size and the run would degrade to eager SILENTLY.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "50331648"))
    _DEV_PARAMS["num_command_queues"] = 1

# One resident build per device: under tracy BOTH the eager forward and the trace pass run, and
# building this pipeline twice on one mesh would re-upload every GPT / HiFiGAN weight set.
_PIPE_CACHE = {}


def _open_perf_mesh():
    """The SOURCE's own device open, honouring the tool's planned topology.

    FABRIC only when the resolved mesh spans MORE THAN ONE chip: the source hardcodes FABRIC_1D
    for its 1x8, and carried into a 1x1 run that still trains ethernet across every visible chip
    and can time out ("Fabric Router Sync: Timeout") before the model is even built.
    """
    rows, cols = int(_MESH_SHAPE[0]), int(_MESH_SHAPE[1])
    if rows * cols > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **_DEV_PARAMS)
    print("[perf] opened mesh %dx%d params=%r" % (rows, cols, _DEV_PARAMS), flush=True)
    return dev


def _close_perf_mesh(dev):
    ttnn.close_mesh_device(dev)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass


def _captured(name):
    """The model's own captured reference tensors (same loader tt/pipeline.py uses)."""
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(here, "..", "..", "_captured", name))
    a = torch.load(os.path.join(base, "args.pt"), map_location="cpu", weights_only=False)
    return list(a) if isinstance(a, (list, tuple)) else [a]


def prompt_ids_for_isl(tokenizer, n_tokens):
    """Build a prompt of EXACTLY ``n_tokens`` XTTS text tokens -> LongTensor [1, n_tokens].

    ISL is the tool's measurement condition, not an example sentence someone typed, so the text is
    tiled/truncated to hit the requested length exactly. (perf_automation ships this name in its
    skeleton but no importable implementation, and the XTTS tokenizer needs its ``lang`` kwarg,
    so the helper lives here.)
    """
    n = max(1, int(n_tokens))
    ids = []
    base = ("it took me quite a long time to develop a voice and now that i have it "
            "i am not going to be silent ")
    try:
        chunk = list(tokenizer.encode(base, lang="en"))
        if not chunk:
            raise ValueError("tokenizer returned no ids")
        while len(ids) < n:
            ids.extend(chunk)
    except Exception as e:  # noqa: BLE001 - tokenizer unavailable: tile the captured reference ids
        print("[perf] tokenizer unavailable (%r); tiling captured reference tokens" % (e,), flush=True)
        cap = [int(x) for x in _captured("g_p_t")[0].reshape(-1).tolist()]
        while len(ids) < n:
            ids.extend(cap)
    return torch.tensor([ids[:n]], dtype=torch.long)


def _perf_inputs(model, batch=1):
    """The pipeline's 4-tuple input contract with the ONE axis that drives traced dispatch count
    (text / phoneme tokens) pinned to PERF_ISL_TOKENS, at PERF_BATCH streams.

    Built from the model's OWN captured cond-mel + shipped speaker reference rather than
    re-encoding B full-length clips, so no forward runs at a correctness-stress length.
    """
    B = max(1, int(batch))
    cond_mel = _captured("conditioning_encoder")[0].float()
    audio_codes = _captured("g_p_t")[2].long()
    ref_wav = P.load_reference_audio_16k()
    text_inputs = prompt_ids_for_isl(getattr(model, "tokenizer", None), PERF_ISL_TOKENS)
    if B > 1:
        cond_mel = cond_mel.repeat(B, 1, 1)
        audio_codes = audio_codes.repeat(B, 1)
        ref_wav = ref_wav.repeat(B, 1, 1)
        text_inputs = text_inputs.repeat(B, 1)
    return cond_mel, ref_wav, text_inputs, audio_codes


def _build_for_perf(dev):
    """Return the RESIDENT, STAGE-EXPOSING pipeline object -- the one carrying PIPELINE_STAGES plus
    the per-stage trace hooks -- built on the passed-in device exactly as the source builds it."""
    key = id(dev)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    from models.demos.xtts_v2.tt.pipeline import build_pipeline, load_reference_model

    model = load_reference_model()
    # PERF_LAYERS goes STRAIGHT to the builder (None == all layers); build_pipeline absorbs it
    # through **kwargs since XTTS's depth is fixed by the checkpoint.
    pipe = build_pipeline(dev, model=model, batch=PERF_BATCH, n_layers=PERF_LAYERS)
    # Bound the traced stages on this model's heavy axis: every <stage>_trace_inputs() feeds the
    # SHORT prompt + single stream instead of the pipeline's full-length 4-stream reference case
    # (which re-encodes 4 real speaker clips and prefills a much longer GPT prefix).
    _inputs = _perf_inputs(model, batch=PERF_BATCH)
    for _st in list(getattr(P, "PIPELINE_STAGES", [])):
        setattr(pipe, "%s_trace_inputs" % _st, (lambda inp=_inputs: inp))
    pipe._perf_inputs = _inputs
    _PIPE_CACHE[key] = pipe
    return pipe


# The perf harness points pytest at this node but selects the case by the PIPELINE TASK name
# ("tts"), i.e. `-k test_tts_perf`.  `-k` matches marker keywords as well as the function name, so
# carrying the task-named marker keeps this node selectable under either spelling instead of
# collecting 1 item and deselecting it ("0 selected" -> the gate reports 'workload did not run').
@pytest.mark.test_tts_perf
def test_main_perf():
    device = _open_perf_mesh()
    try:
        # 1) build the pipeline EXACTLY as tests/e2e/test_e2e_tts.py does
        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
        #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts
        #    (sdpa/eltwise/transpose/reduction slip through) and the 12000-marker buffer overflows,
        #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
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

            pipe = _build_for_perf(device)
            cond_mel, ref_wav, text_inputs, _codes = pipe._perf_inputs
            print("PERF_ISL_TOKENS=%d" % text_inputs.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                # BOUNDED: the AR decode is capped at PERF_MAX_NEW_TOKENS audio codes and the
                # prefix at PERF_ISL_TOKENS text tokens. NO reference model, NO PCC comparison.
                out = pipe.run_tts(cond_mel, ref_wav, text_inputs, audio_codes=None,
                                   horizon=PERF_MAX_NEW_TOKENS, generate=True)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None  # perf only — NO PCC
            assert out["waveform"] is not None

        def _traced_forward():
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            # ISL: built to EXACTLY PERF_ISL_TOKENS tokens (the same prompt every traced stage
            # consumes), so the measurement condition is the tool's choice, not the generator's.
            _prompt_ids = _build_for_perf(device)._perf_inputs[2]
            print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry
            # (prefill / decode / vocode) gets traced. The device passed here is the SELF-OPENED
            # mesh, so the trace runs the SAME sharded topology as the eager forward.
            measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH),
                            device)

        def _try_traced():
            try:
                _traced_forward()
                return True
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
                return False

        # MEASUREMENT ORDER: trace is the DEFAULT path, eager is its fallback. Never both under the
        # full-pipeline gate -- building this pipeline twice at full depth on one mesh leaves no
        # memory for the second build's caches, which the gate can only report as 'no markers'.
        _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        if _PERF_TRACE and not _PROFILING:
            if not _try_traced():
                print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly",
                      flush=True)
                _eager_forward()
        else:
            _eager_forward()
            if _PERF_TRACE:
                _try_traced()
    finally:
        _PIPE_CACHE.clear()
        _close_perf_mesh(device)


if __name__ == "__main__":
    test_main_perf()
