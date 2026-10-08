# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kolibri-1 text generation on the QB2 (1x4 Blackhole mesh, TP=4), through the same pipeline the e2e
test gates (tt/pipeline.py).

    python -m models.demos.kolibri_1.demo.demo_text_generation
    python -m models.demos.kolibri_1.demo.demo_text_generation --message "Was ist die Hauptstadt von Frankreich?" --batch 4

Every user gets the same chat prompt (enable_thinking=False); user b samples with seed (seed_offset + b)
under generation_config's rule (top_k=128, top_p=0.97, temperature=1.0) and stops on its stop token.
"""
from __future__ import annotations

import argparse
import time

from models.demos.kolibri_1.demo.mesh import close_mesh, open_mesh
from models.demos.kolibri_1.tt import inputs as kin
from models.demos.kolibri_1.tt.pipeline import build_pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--message", default=kin.CARD_MESSAGE, help="user message (default: the model card's example)")
    ap.add_argument("--batch", type=int, default=kin.batch_size(), help="users per call (default $TT_PERF_BATCH or 32)")
    ap.add_argument("--seed-offset", type=int, default=0, help="user b samples with seed seed_offset + b")
    ap.add_argument("--layers", type=int, default=None, help="cap the decoder depth (default: all 50 layers)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="stop earlier than the stop token / KV cap")
    args = ap.parse_args(argv)

    tok = kin.load_tokenizer()
    prompt = kin.encode_prompt(tok, args.message)
    seeds = [args.seed_offset + b for b in range(args.batch)]
    mesh = open_mesh()
    try:
        pipe = build_pipeline(mesh, layers=args.layers, batch=args.batch)
        t0 = time.time()
        out = pipe.generate(prompt, seeds, max_new_tokens=args.max_new_tokens)
        dt = time.time() - t0
        for b, ids in enumerate(out["tokens"]):
            end = "stop token" if out["ended_on_stop"][b] else "cap"
            print(
                f"\n=== user {b} (seed {seeds[b]}, {len(ids)} tokens, ended on {end}) ===\n{tok.decode(ids, skip_special_tokens=True)}"
            )
        n = sum(len(ids) for ids in out["tokens"])
        print(f"\n[demo] {args.batch} users, {out['steps']} steps, {n} tokens in {dt:.1f}s")
    finally:
        close_mesh(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
