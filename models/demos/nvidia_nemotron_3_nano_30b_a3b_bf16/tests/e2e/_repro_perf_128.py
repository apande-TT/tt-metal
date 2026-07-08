"""Reproduce the published perf table row (TTFT + decode T/S/U) at ISL=128, OSL=128.

Mirrors what measure_adapter (decode) already does, and provides the missing prefill
counterpart inline (measure_prefill is not shipped in trace_replay.py on this branch).
Both are 2CQ trace-replay: I/O staged on cq1, execute_trace on cq0, gated by a record/
wait event pair, exactly like _replay_2cq in trace_replay.py.
"""
from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DEV_PARAMS = {
    "l1_small_size": 24576,
    "trace_region_size": int(os.environ.get("TT_PERF_TRACE_REGION", "120000000")),
    "num_command_queues": 2,
}

WARMUP = int(os.environ.get("TT_TRACE_WARMUP_ITERS", "3"))
PREFILL_ITERS = int(os.environ.get("TT_PREFILL_ITERS", "8"))
DECODE_ITERS = int(os.environ.get("TT_TRACE_REPLAY_ITERS", "128"))
ISL = int(os.environ.get("TT_ISL", "128"))


def _measure_2cq(device, step_fn, write_inputs, iters, warmup):
    for _ in range(warmup):
        step_fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    step_fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        if callable(write_inputs):
            write_inputs()
            ev = ttnn.record_event(device, 1)
            ttnn.wait_for_event(0, ev)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    per_iter_ms = ((time.perf_counter() - t0) / iters) * 1000.0
    try:
        ttnn.release_trace(device, tid)
    except Exception:
        pass
    return per_iter_ms


def _build_isl_prompt(tok, target_len: int) -> torch.Tensor:
    text = (
        "The city of Paris is the capital of France. It sits on the Seine and is "
        "famous for the Eiffel Tower, the Louvre, and Notre-Dame. The country of France "
        "borders Belgium, Luxembourg, Germany, Switzerland, Italy, Monaco, Andorra, and "
        "Spain. Historically the region was part of Gaul, then the Frankish kingdom, and "
        "later the French Republic that we know today. "
    ) * 8
    ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    if ids.shape[1] >= target_len:
        return ids[:, :target_len]
    # tile to reach target length
    while ids.shape[1] < target_len:
        need = target_len - ids.shape[1]
        ids = torch.cat([ids, ids[:, :need]], dim=1)
    return ids[:, :target_len]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    input_ids = _build_isl_prompt(tok, ISL)
    print(f"[perf] ISL(actual)={input_ids.shape[1]} target={ISL}", flush=True)

    device, is_mesh = pl.open_pipeline_mesh(**DEV_PARAMS)
    try:
        pipe = pl.build_pipeline(device, model, compose=True)
        print(f"[perf] mesh={is_mesh} shard_active={pipe.shard_active}", flush=True)

        pipe.prefill_trace_setup(input_ids)
        pf_ms = _measure_2cq(
            device,
            pipe.prefill_trace_step,
            pipe.prefill_write_inputs,
            iters=PREFILL_ITERS,
            warmup=WARMUP,
        )
        print(
            f"PREFILL_TRACE_MS={pf_ms:.4f} PATH=trace+2cq ISL={input_ids.shape[1]} "
            f"warmup={WARMUP} iters={PREFILL_ITERS}",
            flush=True,
        )

        from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

        def _build(_dev):
            return pipe

        adapter = PipelineDecodeAdapter(_build, input_ids, batch=1)
        adapter.setup(device)
        dec_ms = _measure_2cq(
            device,
            adapter.step,
            getattr(adapter, "write_inputs", None),
            iters=DECODE_ITERS,
            warmup=WARMUP,
        )
        tps = 1000.0 / dec_ms if dec_ms > 0 else 0.0
        print(
            f"DECODE_PER_TOKEN_MS={dec_ms:.4f} PATH=trace+2cq OSL_ITERS={DECODE_ITERS} "
            f"TOKENS_PER_SEC_PER_USER={tps:.2f} batch=1",
            flush=True,
        )
    finally:
        pl.close_pipeline_mesh(device, is_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
