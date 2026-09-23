# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 demo: teacher-forced causal-LM continuation over the whole-stack bodies and the graduated LM head.

    python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_continuation

Runs the SAME `pipeline.run_text_continuation` the e2e test asserts on: one forward over the real
text, printing the model's greedy next-token pick at every position.

READ THE OUTPUT AS A PARITY CHECK, NOT AS LANGUAGE. This is a TTS checkpoint: the backbone emits
AUDIO codebook tokens and its tied TEXT head is effectively untrained, so the continuation is
near-uniform garbage even when the load is bit-correct. What matters is that it is the SAME
garbage the HF reference produces, which is what `--compare` shows.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voxtral-4B-TTS-2603 text continuation on TTNN")
    parser.add_argument("--text", action="append", default=None, help="a prompt; repeat for more")
    parser.add_argument("--batch", type=int, default=common.DEFAULT_BATCH)
    parser.add_argument("--layers", type=int, default=None, help="cap the depth of the text stack (None = all 26)")
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("VOXTRAL_DEVICE_ID", "0")))
    parser.add_argument("--compare", action="store_true", help="also run the HF golden and print the per-sample PCC")
    args = parser.parse_args(argv)

    common.use_all_cpu_threads()
    hf_model = common.load_reference_model()
    tok = common.load_tokenizer()

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=24576,
        trace_region_size=200 * 1024 * 1024,
        num_command_queues=1,
    )
    try:
        pipe = pipeline.build_pipeline(
            device, model=hf_model, heads=("text_continuation",), layers=args.layers, batch=args.batch
        )
        if args.text:
            length = min(len(tok.encode(t, bos=True)) for t in args.text)
            input_ids = torch.tensor([tok.encode(t, bos=True)[:length] for t in args.text], dtype=torch.long)
            prompts = args.text
        else:
            input_ids, prompts = common.build_batch_inputs(batch=args.batch, seq_len=common.full_prompt_len(args.batch))

        print(f"batch={input_ids.shape[0]} tokens per row={input_ids.shape[1]} (every position scored)")
        result = pipe.run_text_continuation(input_ids=input_ids)
        picks = result["next_tokens"]
        for i in range(min(8, picks.shape[0])):
            ids = picks[i].tolist()
            print(f"  [{i:02d}] {prompts[i][:48]!r} -> next after each position {ids} {tok.decode(ids)!r}")
        if picks.shape[0] > 8:
            print(f"  ... {picks.shape[0] - 8} more samples")

        if args.compare:
            from models.demos.voxtral_4b_tts_2603.reference import golden

            hf = golden.hf_reference_text_continuation(hf_model, input_ids)
            seq = int(input_ids.shape[1])
            scores = [
                min(common.pcc(result["logits"][i, s], hf["logits"][i, s]) for s in range(seq))
                for i in range(input_ids.shape[0])
            ]
            agree = float((picks == hf["next_tokens"]).float().mean())
            print(f"next-token agreement with the reference: {agree:.6f}")
            print(f"e2e PCC={min(scores)}")
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
