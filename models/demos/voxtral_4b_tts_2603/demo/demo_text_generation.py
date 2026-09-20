# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 demo -- real text generation on the composed graduated TTNN stack.

    python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_generation

Builds through `tt.pipeline.build_pipeline` and calls
`pipeline.run_text_generation(...)` -- the SAME function
`tests/e2e/test_e2e_text_generation.py` calls, so there is one copy of the
wiring and a green test implies a working demo.

The demo runs the FREE-RUNNING pass only: the stack feeds its own on-device
argmax back in. The teacher-forced measurement pass (and the HF golden it needs)
belongs to the test.
"""
from __future__ import annotations

import argparse
import time

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline

# This checkpoint's tied text head is effectively untrained: next-token entropy is 11.417 nats
# against 11.784 for a uniform draw over the 131072-token vocabulary, and top-1 probability is
# ~6e-4. The continuations below are therefore NOT coherent English. That is a property of the
# checkpoint, not of the port -- the port is gated on logits PCC against HF, never on coherence.
UNTRAINED_HEAD_NOTE = (
    "NOTE: this checkpoint's tied text head is effectively untrained (next-token entropy 11.417 "
    "nats vs 11.784 for uniform over 131072 tokens, top-1 prob ~6e-4), so the continuations are "
    "NOT coherent English. That is a property of the checkpoint, not a bug in the port. The port "
    "is gated on logits PCC against HF (see tests/e2e/test_e2e_text_generation.py), not on "
    "readable output."
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=common.DEFAULT_BATCH, help="independent prompts driven at once")
    p.add_argument("--seq-len", type=int, default=common.DEFAULT_SEQ_LEN, help="real tokens per prompt (no padding)")
    p.add_argument("--horizon", type=int, default=None, help="new tokens to generate (default: resolved from config)")
    p.add_argument("--layers", type=int, default=None, help="cap the decoder depth (default: every layer)")
    p.add_argument("--device-id", type=int, default=0, help="device to open")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    input_ids, prompt_texts = common.build_batch_inputs(batch=args.batch, seq_len=args.seq_len)
    print(f"[demo] {input_ids.shape[0]} independent prompts x {input_ids.shape[1]} real tokens", flush=True)

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        started = time.time()
        pipeline = build_pipeline(
            device,
            heads=("text_generation",),
            layers=args.layers,
            batch=int(input_ids.shape[0]),
        )
        print(f"[demo] pipeline built in {time.time() - started:.1f}s: {pipeline.describe()}", flush=True)

        started = time.time()
        result = pipeline.run_text_generation(
            input_ids=input_ids,
            horizon=args.horizon,
            teacher_forced=False,
        )
        elapsed = time.time() - started

        batch = result["batch"]
        horizon = result["horizon"]
        print("")
        print("=" * 100)
        print(
            f"[demo] batch={batch}  prompt_len={result['prompt_len']}  horizon={horizon} "
            f"({result['horizon_provenance']})  wall={elapsed:.1f}s"
        )
        print("=" * 100)
        for i in range(batch):
            print(f"\n--- sample {i:02d} ---")
            print(f"  prompt      : {result['prompt_texts'][i]!r}")
            print(f"  continuation: {result['texts'][i]!r}")
            print(f"  new ids     : {result['generated_ids'][i, result['prompt_len']:].tolist()}")
        print("")
        print("=" * 100)
        print(UNTRAINED_HEAD_NOTE)
        print("=" * 100, flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
