# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable demo for `mistralai/Voxtral-Mini-3B-2507` — audio -> text on Tenstorrent.

Loads real processor input (a 16 kHz waveform + a text prompt), runs the SHARED
chained TTNN pipeline (`tt/pipeline.py` — the exact code the e2e test asserts on),
and prints the generated text.

Run:
    cd /home/ttuser/tt-metal
    TT_METAL_HOME=/home/ttuser/tt-metal \
    PYTHONPATH=/home/ttuser/tt-metal-pr46283:/home/ttuser/tt-metal \
    /home/ttuser/tt-metal/python_env/bin/python -m \
        models.demos.hf_eager.voxtral_mini_3b_2507.demo.demo_transcribe --max-new-tokens 16
"""
from __future__ import annotations

import argparse

import ttnn
from models.demos.hf_eager.voxtral_mini_3b_2507.tt import pipeline as P


def main():
    ap = argparse.ArgumentParser(description="Voxtral-Mini-3B audio->text TTNN demo")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0, help="synthetic waveform length")
    ap.add_argument("--prompt", type=str, default="\nWhat is said in the audio?")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        print("Loading Voxtral model + processor ...", flush=True)
        model = P.load_hf_model()
        proc = P.load_processor()
        input_ids, input_features, n_audio, atid, prompt = P.build_inputs(
            proc, model, seconds=args.seconds, prompt=args.prompt
        )
        print(f"Prompt: {args.prompt!r}", flush=True)
        print(
            f"Input: {input_ids.shape[1]} tokens ({n_audio} audio placeholders), "
            f"input_features={tuple(input_features.shape)}",
            flush=True,
        )

        pipe = P.VoxtralTTPipeline(device, model)
        out = pipe.run(input_ids, input_features, max_new_tokens=args.max_new_tokens)

        text = proc.tokenizer.decode(out["tt_tokens"], skip_special_tokens=True)
        print("\n================ GENERATED (TTNN) ================", flush=True)
        print(f"tokens: {out['tt_tokens']}", flush=True)
        print(f"text  : {text!r}", flush=True)
        print("==================================================", flush=True)
        print(f"graduated stubs invoked: {sorted(out['invoked'])}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
