# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance test for the Llama-3.1-8B-Instruct main pipeline (prefill + decode).

Measures end-to-end latency of the TTNN forward pass without correctness assertions.
"""
from __future__ import annotations

import os
import time

import pytest
import torch
import ttnn

os.environ.setdefault("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))

# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
# Pass PERF_LAYERS straight to the builder: None is every builder's own all-layers value. Do NOT
# default it to a number here -- that would silently cap the full-depth gate.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape is how a run honours them. Give it the SOURCE's own shape as the default, so an
# unset env behaves exactly as the demo does.
from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
# FABRIC only when the resolved mesh spans MORE THAN ONE chip.
if _MESH_SHAPE[0] * _MESH_SHAPE[1] > 1:
    _DEV_PARAMS["fabric_config"] = True
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))


@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_main_perf(mesh_device, device_params):
    """End-to-end latency for Llama-3.1-8B prefill + decode on TTNN device."""
    from transformers import AutoTokenizer

    from models.demos.llama3_1_8b_p150.tt.pipeline import build_pipeline

    def _eager_forward():
        tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        prompt = "The capital of France is"
        ids = tok(prompt, return_tensors="pt").input_ids

        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(mesh_device)
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

        _fw0 = time.monotonic()
        try:
            generator = build_pipeline(mesh_device, max_seq_len=1024, batch_size=1, num_layers=PERF_LAYERS)
            prompt_ids = ids
            decoding_pos = [int(prompt_ids.shape[1])]

            # Prefill: process entire prompt
            prefill_out = generator.prefill_forward_text(
                prompt_ids,
                page_table=generator.page_table,
                kv_cache=generator.tt_kv_cache,
                prompt_lens=decoding_pos,
                sampling_params=None,
                warmup_prefill=True,
                enable_trace=True,
            )

            # Extract logits and greedy-sample token
            if isinstance(prefill_out, (tuple, list)):
                prefill_logits = None
                for item in prefill_out:
                    if isinstance(item, torch.Tensor):
                        prefill_logits = item
                        break
                if prefill_logits is None:
                    raise TypeError("no tensor in prefill output")
            else:
                prefill_logits = prefill_out

            cur_tok = prefill_logits.argmax(-1, keepdim=True)  # [batch, 1]
            cur_pos = torch.tensor(decoding_pos)

            # Decode loop: generate PERF_MAX_NEW_TOKENS - 1 additional tokens
            # (1 from prefill + N-1 from decode = N total)
            for i in range(PERF_MAX_NEW_TOKENS - 1):
                out = generator.decode_forward(
                    cur_tok,
                    cur_pos,
                    enable_trace=True,
                    page_table=generator.page_table,
                    kv_cache=generator.tt_kv_cache,
                    reset_batch=(i == 0),
                    sampling_params=None,
                    prompt_tokens=prompt_ids,
                    output_tokens=cur_tok,
                )

                # Extract logits and greedy-sample token
                if isinstance(out, (tuple, list)):
                    out_logits = None
                    for item in out:
                        if isinstance(item, torch.Tensor):
                            out_logits = item
                            break
                    if out_logits is None:
                        raise TypeError("no tensor in decode output")
                else:
                    out_logits = out

                cur_tok = out_logits.argmax(-1, keepdim=True)  # [batch, 1]
                cur_pos = cur_pos + 1

            try:
                ttnn.ReadDeviceProfiler(mesh_device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)

        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert prefill_out is not None

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        def _build_for_perf(dev):
            return build_pipeline(dev, max_seq_len=1024, batch_size=1, num_layers=PERF_LAYERS)

        tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        _prompt_ids = tok("The capital of France is", return_tensors="pt").input_ids
        # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
        # traced (+2CQ where the stage stages its inputs). Falls back to the single decode
        # contract for pipelines that expose only decode_step.
        measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1), mesh_device, mode="auto")

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
