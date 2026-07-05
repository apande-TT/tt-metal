# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance test for the 'text_generation' pipeline of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Builds and runs the SHARED chained TTNN pipeline EXACTLY as demo/demo_text_generation.py does
(tt/pipeline.py) fully IN-PROCESS so tracy can profile every device op. Perf only -- no PCC.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# Trace-replay per-token latency (GPU-comparable T/S/U). MODEL-AGNOSTIC + OFF-BY-DEFAULT-SAFE:
# TT_PERF_TRACE=1 (default) adds trace_region_size + num_command_queues to the device open so the
# per-token block below CAN capture a device trace; TT_PERF_TRACE=0 restores the plain eager open
# (exactly the old behavior -> guaranteed non-breaking escape hatch for tight-memory models).
_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "120000000"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))  # 2 = trace+2CQ overlap path


def test_text_generation_perf():
    # ---- device open: MATCH THE DEMO exactly (self-open the mesh via pl.open_pipeline_mesh). ----
    # The pipeline is tensor-parallel on a mesh; a single `device` fixture would silently disable
    # sharding (shard_active=False) and profile the wrong single-chip config. When TT_PERF_TRACE is
    # set we try to thread trace_region_size / num_command_queues through the same open; if the open
    # does not accept them we fall back to the plain open exactly as the demo does.
    device = None
    is_mesh = False
    if _PERF_TRACE:
        try:
            device, is_mesh = pl.open_pipeline_mesh(
                l1_small_size=_DEV_PARAMS["l1_small_size"],
                trace_region_size=_DEV_PARAMS["trace_region_size"],
                num_command_queues=_DEV_PARAMS["num_command_queues"],
            )
        except TypeError:
            device = None
    if device is None:
        device, is_mesh = pl.open_pipeline_mesh(l1_small_size=_DEV_PARAMS["l1_small_size"])

    try:
        # 1) build the pipeline EXACTLY as demo/demo_text_generation.py does
        tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.eval()
        eos = int(getattr(model.config, "eos_token_id", 2))

        prompt = os.environ.get("TT_PERF_PROMPT", "The capital of France is")
        input_ids = tok(prompt, return_tensors="pt")["input_ids"]

        pipe = pl.build_pipeline(device, model, compose=True)
        print(f"[perf] mesh={is_mesh} shard_active={pipe.shard_active}", flush=True)

        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
        #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
        #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
        #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
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

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        _fw0 = time.monotonic()
        try:
            new_ids, _ = pipe.generate(input_ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
            out = new_ids
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only -- NO PCC

        # ---- clean, GPU-comparable per-token latency via trace-replay (GENERIC + guarded) ----
        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

                os.environ["TT_PERF_LAYERS"] = "0"  # trace the FULL model (depth cap above may have capped it)

                def _build_for_perf(dev):
                    # REUSE the pipeline built in step 1 (return that same object). Do NOT build a
                    # second copy: a 2nd full build of a large resident model OOMs the mesh, and its
                    # layer children are already resident.
                    return pipe

                _prompt_ids = input_ids
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")  # prints TRACE_PER_TOKEN_MS=<ms>

                if hasattr(pipe, "prefill_trace_step") and os.environ.get("TT_PERF_PREFILL_TRACE") == "1":
                    from models.experimental.perf_automation.agent.trace_replay import measure_prefill

                    pipe.prefill_trace_setup(_prompt_ids)
                    measure_prefill(
                        device,
                        pipe.prefill_trace_step,
                        write_inputs=getattr(pipe, "prefill_write_inputs", None),
                        mode="auto",
                    )
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        pl.close_pipeline_mesh(device, is_mesh)