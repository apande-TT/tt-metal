# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 demo: greedy causal-LM continuation over the whole-stack bodies and the graduated LM head.

    python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_continuation --horizon 4

Runs the SAME `pipeline.run_text_continuation` the e2e test asserts on.

READ THE OUTPUT AS A PARITY CHECK, NOT AS LANGUAGE. This is a TTS checkpoint: the backbone emits
AUDIO codebook tokens and its tied TEXT head is effectively untrained, so the continuation is
near-uniform garbage even when the load is bit-correct. What matters is that it is the SAME
garbage the HF reference produces, which is what `--compare` shows.
"""
from __future__ import annotations

import argparse
import sys

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voxtral-4B-TTS-2603 text continuation on TTNN")
    parser.add_argument("--text", action="append", default=None, help="a prompt; repeat for more")
    parser.add_argument("--horizon", type=int, default=4, help="greedy steps to decode")
    parser.add_argument("--batch", type=int, default=common.DEFAULT_BATCH)
    parser.add_argument("--layers", type=int, default=None, help="cap the depth of the text stack (None = all 26)")
    parser.add_argument("--device-id", type=int, default=0)
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
            input_ids, prompts = common.build_batch_inputs(batch=args.batch)

        eos = common.eos_token_id(hf_model)
        print(f"batch={input_ids.shape[0]} prompt_tokens={input_ids.shape[1]} " f"horizon={args.horizon} eos={eos}")
        result = pipe.run_text_continuation(input_ids=input_ids, horizon=args.horizon, eos_id=eos)

        for i in range(min(8, result["tokens"].shape[0])):
            ids = result["tokens"][i].tolist()
            print(f"  [{i:02d}] {prompts[i][:48]!r} -> {ids} {tok.decode(ids)!r}")
        if result["tokens"].shape[0] > 8:
            print(f"  ... {result['tokens'].shape[0] - 8} more samples")

        if args.compare:
            from models.demos.voxtral_4b_tts_2603.reference import golden

            hf = golden.hf_reference_text_continuation(hf_model, input_ids, args.horizon, eos_id=eos)
            steps = min(len(result["step_logits"]), len(hf["step_logits"]))
            scores = [
                min(common.pcc(result["step_logits"][s][i], hf["step_logits"][s][i]) for s in range(steps))
                for i in range(input_ids.shape[0])
            ]
            match = torch.equal(result["tokens"][:, :steps], hf["tokens"][:, :steps])
            print(f"tokens identical to the reference: {bool(match)}")
            print(f"e2e PCC={min(scores)}")
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
