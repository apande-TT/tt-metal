# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Call 2 -- text -> text on Tenstorrent hardware.

The FLUX.2-klein-9B `text_encoder` IS a `Qwen3ForCausalLM` with its own
`generation_config`, so greedy generation is one of this checkpoint's real task
heads.  Decoding stops on the model's own `eos_token_id` ([151645, 151643]);
`--max-new-tokens` is the safety cap, and the same cap is given to the HF
reference so the two sequences are compared over the same length.

    ./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_text_generation \
        --prompt "Describe a red apple in one short sentence." --max-new-tokens 32 --compare-hf
"""

from __future__ import annotations

import argparse
import time

from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.mesh import open_flux_mesh
from models.demos.flux_2_klein_9b.tt.pipeline import build_pipeline


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLUX.2-klein-9B text encoder as a causal LM, on a 1x8 mesh")
    p.add_argument("--prompt", default="Describe a red apple in one short sentence.")
    p.add_argument("--max-new-tokens", type=int, default=32, help="safety cap; the stop rule is the model's eos")
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="decode N INDEPENDENT prompts in lockstep, one program per step "
        "(N distinct prompts from reference.batch_text_prompts); --prompt is used when N == 1",
    )
    p.add_argument("--layers", type=int, default=None, help="cap the decoder depth")
    p.add_argument("--compare-hf", action="store_true", help="also run model.generate() and compare")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    R.ensure_flux_imports()

    with open_flux_mesh() as device:
        pipe = build_pipeline(device, layers=args.layers, batch=args.batch)
        prompt = args.prompt if args.batch == 1 else R.batch_text_prompts(args.batch)
        start = time.time()
        text, ids = pipe.run_text_generation(prompt, max_new_tokens=args.max_new_tokens, return_ids=True)
        elapsed = time.time() - start

    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    texts = [text] if isinstance(text, str) else list(text)
    rows = [ids] if isinstance(prompt, str) else list(ids)
    for i, (one, out, row) in enumerate(zip(prompts, texts, rows)):
        print(f"[{i}] prompt   : {one!r}")
        print(f"[{i}] tt output: {out!r}  ({len(row)} tokens)")
    print(f"tt wall time: {elapsed:.1f}s for {len(prompts)} stream(s) (stop ids {R.stop_token_ids()})")

    if args.compare_hf:
        _, ref_rows, _ = R.hf_text_generation_logits_batch(prompts, args.max_new_tokens)
        matches = []
        for i, (row, ref) in enumerate(zip(rows, ref_rows)):
            n = min(len(row), len(ref))
            matches.append(sum(int(a == b) for a, b in zip(row[:n], ref[:n])) / max(n, 1))
            print(f"[{i}] hf output: {R.load_tokenizer().decode(ref, skip_special_tokens=True)!r}")
        print(f"token match={min(matches)} (worst of {len(matches)} stream(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
