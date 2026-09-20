# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 3 demo -- the checkpoint's `acoustic_transformer` section on TTNN.

    python -m models.demos.voxtral_4b_tts_2603.demo.demo_acoustic

Builds through `tt.pipeline.build_pipeline` and calls `pipeline.run_acoustic(...)`
-- the SAME function `tests/e2e/test_e2e_acoustic.py` calls, so there is one copy
of the wiring and a green test implies a working demo.

The two sections are CHAINED ON THE DEVICE: the text backbone's last hidden state
is handed to the acoustic stack as a device tensor and never round-trips through
the host. What comes back is the acoustic hidden state and the semantic-codebook
logits.

SCOPE, STATED UP FRONT: this is not a waveform. Turning semantic codes into audio
needs the checkpoint's `audio_tokenizer` (a weight-normed causal-conv +
sliding-window-attention vocoder) and the flow-matching sampler that drives
`input_projection` / `time_projection`, and this checkpoint ships no reference
implementation for either -- so neither is ported, and neither is pretended.
"""
from __future__ import annotations

import argparse
import time

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import acoustic as ac
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=common.DEFAULT_BATCH, help="independent prompts driven at once")
    p.add_argument("--seq-len", type=int, default=common.DEFAULT_SEQ_LEN, help="real tokens per prompt (no padding)")
    p.add_argument("--layers", type=int, default=None, help="cap the TEXT decoder depth (default: every layer)")
    p.add_argument("--device-id", type=int, default=0, help="device to open")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    input_ids, _ = common.build_batch_inputs(batch=args.batch, seq_len=args.seq_len)
    print(f"[demo] {input_ids.shape[0]} independent prompts x {input_ids.shape[1]} real tokens", flush=True)
    print(ac.dump_section_report(), flush=True)

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        started = time.time()
        pipeline = build_pipeline(
            device,
            heads=("text_generation", "acoustic"),
            layers=args.layers,
            batch=int(input_ids.shape[0]),
        )
        print(f"[demo] pipeline built in {time.time() - started:.1f}s: {pipeline.describe()}", flush=True)

        started = time.time()
        result = pipeline.run_acoustic(input_ids=input_ids)
        elapsed = time.time() - started

        hidden, logits = result["acoustic_hidden"], result["semantic_logits"]
        print("")
        print("=" * 100)
        print(f"[demo] acoustic blocks={result['n_layers']}  wall={elapsed:.1f}s")
        print(f"[demo] acoustic hidden  {tuple(hidden.shape)}")
        print(f"[demo] semantic logits  {tuple(logits.shape)}  (codebook width {logits.shape[-1]})")
        print("=" * 100)
        top = logits.argmax(dim=-1)
        for i in range(min(4, int(top.shape[0]))):
            print(f"  sample {i:02d} top semantic codes: {top[i, :12].tolist()}")
        print("=" * 100, flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
