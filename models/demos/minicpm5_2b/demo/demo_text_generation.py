# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5-2B text generation on Tenstorrent hardware.

Runs the same chained pipeline the e2e test runs (tt/pipeline.py): tokenizes the model card's chat
example (or --prompt), generates B samples (one seed each) on device and prints the decoded text.

    python -m models.demos.minicpm5_2b.demo.demo_text_generation
"""
from __future__ import annotations

import argparse

from models.demos.minicpm5_2b.demo.device import opened_device
from models.demos.minicpm5_2b.tt import inputs as tt_inputs
from models.demos.minicpm5_2b.tt.pipeline import build_pipeline, load_hf_model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", default=None, help="user message (default: the model card's example)")
    ap.add_argument("--batch", type=int, default=tt_inputs.batch_size(), help="samples per call (one seed each)")
    ap.add_argument("--base-seed", type=int, default=tt_inputs.BASE_SEED, help="sample i uses base_seed + i")
    ap.add_argument("--max-new-tokens", type=int, default=tt_inputs.EXAMPLE_MAX_NEW_TOKENS)
    ap.add_argument("--layers", type=int, default=None, help="cap the decoder depth (profiling only)")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--show", type=int, default=4, help="how many samples to print")
    args = ap.parse_args(argv)

    tokenizer = tt_inputs.load_tokenizer()
    if args.prompt is None:
        ids = tt_inputs.example_input_ids(tokenizer)
    else:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=tt_inputs.EXAMPLE_ADD_GENERATION_PROMPT,
            enable_thinking=tt_inputs.EXAMPLE_ENABLE_THINKING,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
    ids = ids.repeat(args.batch, 1)
    seeds = [args.base_seed + i for i in range(args.batch)]

    with opened_device(device_id=args.device_id) as device:
        pipe = build_pipeline(
            device,
            model=load_hf_model(),
            layers=args.layers,
            batch=args.batch,
            prompt_len=ids.shape[-1],
            max_new_tokens=args.max_new_tokens,
        )
        noise = tt_inputs.gumbel_noise(seeds, args.max_new_tokens, pipe.vocab)
        res = pipe.run_text_generation(ids, noise, max_new_tokens=args.max_new_tokens)

    print(f"stop_reason={res['stop_reason']} steps={res['steps']} batch={args.batch}")
    for b in range(min(args.show, args.batch)):
        n = int(res["lengths"][b])
        text = tokenizer.decode(res["tokens"][b, :n], skip_special_tokens=tt_inputs.EXAMPLE_SKIP_SPECIAL_TOKENS)
        print(f"--- sample {b} (seed {seeds[b]}, {n} tokens) ---\n{text}")
    return res


if __name__ == "__main__":
    main()
