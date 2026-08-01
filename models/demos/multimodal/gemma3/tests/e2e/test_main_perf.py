# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the gemma-3 TEXT ('main') pipeline.

Derived from tests/e2e/test_pcc_hf.py, but this is PERF ONLY: the HF reference model and every
PCC / allclose comparison are gone. What remains is the on-device TTNN forward, built exactly as
the demo builds it (models/demos/multimodal/gemma3/tt/pipeline.py::build_pipeline), run BOUNDED,
and measured two ways: trace-replay per-stage latency (the headline) with an eager, profiler-drained
forward as the fallback.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

# Pin the model identity exactly as the e2e conftest does, in case this file is collected alone.
os.environ.setdefault("HF_MODEL", "google/gemma-3-12b-it")
HF_MODEL_ID = os.environ.get("HF_MODEL", "google/gemma-3-12b-it")

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
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
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
# Pass PERF_LAYERS straight to the builder: None is every builder's own all-layers value. Do NOT
# default it to a number here -- that would silently cap the full-depth gate.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# The KV-cache window. NOT a production max: the profiled axis for an LLM is TOKENS (ISL + decode
# steps), and both are capped small above, so this only has to be big enough to hold them.
PERF_MAX_SEQ_LEN = int(os.environ.get("TT_PERF_MAX_SEQ_LEN", "1024"))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "1"))

# TOPOLOGY. The source (tests/e2e/test_pcc_hf.py) uses the `mesh_device` FIXTURE parametrized with
# (1, 1), so this test keeps that fixture and feeds it resolve_mesh_shape's answer -- which honours
# the tool's --devices/--mesh plan (TT_PERF_MESH_ROWS/COLS) and returns the source's own 1x1 when
# the env is unset.
from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
# The source sets NO fabric_config, so none is added here regardless of mesh size.
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = 1


def _prompt_ids_for_isl(tokenizer, n_tokens: int) -> torch.Tensor:
    """EXACTLY n_tokens ids, so the measurement condition is the tool's choice, not a hand-written
    example sentence. Prefers the shared helper; falls back to a deterministic local build."""
    try:
        from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

        ids = prompt_ids_for_isl(tokenizer, n_tokens)
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.long()
    except (ImportError, AttributeError):
        pass

    filler = (
        "The capital of France is Paris, a city whose history, architecture and institutions have "
        "been documented in considerable detail over many centuries of European record keeping. "
    )
    body = tokenizer(filler, add_special_tokens=False)["input_ids"]
    if not body:
        body = [1]
    bos = getattr(tokenizer, "bos_token_id", None)
    out = [int(bos)] if bos is not None else []
    while len(out) < n_tokens:
        out.extend(int(t) for t in body)
    return torch.tensor([out[:n_tokens]], dtype=torch.long)


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_main_perf(device_params, mesh_device, reset_seeds):
    device = mesh_device  # 'device' = mesh_device on multi-chip

    def _tokenizer():
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(HF_MODEL_ID)

    def _build(dev):
        # EXACTLY the demo's build args (tt/pipeline.py::build_pipeline wraps prepare_generator_args
        # + GemmaMultimodalGenerator). No depth argument: this builder reads TT_PERF_LAYERS itself,
        # and ABSENT means all layers.
        from models.demos.multimodal.gemma3.tt.pipeline import build_pipeline

        return build_pipeline(dev, max_seq_len=PERF_MAX_SEQ_LEN, batch_size=PERF_BATCH, instruct=True)

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

        prompt_ids = _prompt_ids_for_isl(_tokenizer(), PERF_ISL_TOKENS)
        print("PERF_ISL_TOKENS=%d" % prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        generator = _build(device)

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))
        _fw0 = time.monotonic()
        try:
            # BOUNDED: one prefill over the ISL-sized prompt + PERF_MAX_NEW_TOKENS decode steps.
            state = generator.decode_prefill(prompt_ids)
            for _ in range(max(1, PERF_MAX_NEW_TOKENS)):
                state = generator.decode_step(state)
            out = state
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
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

        def _build_for_perf(dev):
            return _build(dev)

        # ISL: build the prompt to EXACTLY PERF_ISL_TOKENS tokens rather than writing an example
        # sentence, so the measurement condition is the tool's choice and not the generator's.
        _prompt_ids = _prompt_ids_for_isl(_tokenizer(), PERF_ISL_TOKENS)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
        # traced. Falls back to the single decode contract for pipelines that expose only decode_step.
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
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()
