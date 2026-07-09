# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone trace + 2CQ perf harness for the Voxtral pipeline.

The original scorecard was produced by `perf_automation.agent.trace_replay`
(measure_prefill / measure_adapter), which is not committed anywhere. This
reimplements the same measurement directly against the perf hooks the pipeline
already exposes:

  PREFILL (audio front-end, one-shot):
     prefill_capture()          -> traced encoder->reshape->projector
     prefill_write_inputs()     -> CQ1 feats upload (2CQ overlap)

  DECODE (per-token, fixed T=32 window, host-op-free):
     decode_prefill()           -> build resident inputs
     decode_step()              -> traced full causal-LM forward + on-device argmax
     decode_write_inputs()      -> CQ1 token-id upload (2CQ overlap)

Emits KEY=value lines: PREFILL_TRACE_MS, DECODE_TRACE_ONLY_MS,
DECODE_TRACE_2CQ_MS, TTFT_MS, DECODE_PER_TOKEN_MS, TOKENS_PER_SEC.
Perf only — no PCC (correctness is the e2e gate).
"""
import os
import time

import ttnn
from models.demos.hf_eager.voxtral_mini_3b_2507.tt import pipeline as P

WARMUP = int(os.environ.get("PERF_WARMUP", "3"))
ITERS = int(os.environ.get("PERF_ITERS", "20"))
PREFILL_ITERS = int(os.environ.get("PERF_PREFILL_ITERS", "10"))
SECONDS = float(os.environ.get("PERF_SECONDS", "2.0"))
DEVICE_ID = int(os.environ.get("PERF_DEVICE_ID", "0"))
TRACE_REGION = int(os.environ.get("PERF_TRACE_REGION", "200000000"))


def _sync(dev):
    ttnn.synchronize_device(dev)


def _time_trace_only(dev, tid, iters):
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    _sync(dev)
    return (time.perf_counter() - t0) / iters * 1000.0


def _time_trace_2cq(dev, tid, write_inputs, iters):
    """Overlap CQ1 host upload with CQ0 traced compute, event-synchronized."""
    _sync(dev)
    op_event = ttnn.record_event(dev, 0)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.wait_for_event(1, op_event)  # CQ1 waits until CQ0 has consumed prior inputs
        write_inputs()  # copy_host_to_device_tensor on CQ1
        write_event = ttnn.record_event(dev, 1)
        ttnn.wait_for_event(0, write_event)  # CQ0 waits for the upload
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(dev, 0)
    _sync(dev)
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    dev = ttnn.open_device(
        device_id=DEVICE_ID, l1_small_size=24576, trace_region_size=TRACE_REGION, num_command_queues=2
    )
    try:
        model = P.load_hf_model()
        proc = P.load_processor()
        input_ids, input_features, n_audio, atid, prompt = P.build_inputs(proc, model, seconds=SECONDS)
        pipe = P.VoxtralTTPipeline(dev, model)
        print(f"CONTEXT_LEN={int(input_ids.shape[1])} N_AUDIO={n_audio}", flush=True)

        # ---------------- PREFILL (audio front-end) ---------------- #
        pipe._ensure_prefill_capture_built()
        for _ in range(WARMUP):
            pipe.prefill_capture()
        _sync(dev)
        pf_tid = ttnn.begin_trace_capture(dev, cq_id=0)
        pipe.prefill_capture()
        ttnn.end_trace_capture(dev, pf_tid, cq_id=0)
        _sync(dev)
        prefill_ms = _time_trace_2cq(dev, pf_tid, pipe.prefill_write_inputs, PREFILL_ITERS)
        ttnn.release_trace(dev, pf_tid)
        print(f"PREFILL_TRACE_MS={prefill_ms:.4f}", flush=True)

        # ---------------- DECODE (per-token) ---------------- #
        pipe.decode_prefill()
        for _ in range(WARMUP):
            pipe.decode_step(None)
        _sync(dev)
        dec_tid = ttnn.begin_trace_capture(dev, cq_id=0)
        pipe.decode_step(None)
        ttnn.end_trace_capture(dev, dec_tid, cq_id=0)
        _sync(dev)
        dec_only_ms = _time_trace_only(dev, dec_tid, ITERS)
        dec_2cq_ms = _time_trace_2cq(dev, dec_tid, lambda: pipe.decode_write_inputs(None), ITERS)
        ttnn.release_trace(dev, dec_tid)
        print(f"DECODE_TRACE_ONLY_MS={dec_only_ms:.4f}", flush=True)
        print(f"DECODE_TRACE_2CQ_MS={dec_2cq_ms:.4f}", flush=True)

        # ---------------- derived scorecard ---------------- #
        ttft = prefill_ms + dec_2cq_ms  # prefill front-end + first token
        tps = 1000.0 / dec_2cq_ms
        print(f"TTFT_MS={ttft:.4f}", flush=True)
        print(f"DECODE_PER_TOKEN_MS={dec_2cq_ms:.4f}", flush=True)
        print(f"TOKENS_PER_SEC={tps:.4f}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
