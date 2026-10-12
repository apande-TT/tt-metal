# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Demo, Call 1 -- text_generation: chat prompt -> generated text with Devstral-Small-2-24B on a 1x4 mesh.

    python -m models.demos.devstral_small_2_24b_instruct_2512.demo.demo_text_generation [--prompt "..."]

Runs tt.pipeline.run_text_generation -- the same chained TT forward the e2e test gates -- for `--batch` rows of
the same prompt, each sampled with its own seed, decoding until every row emits EOS.
"""
from __future__ import annotations

import argparse

from models.demos.devstral_small_2_24b_instruct_2512.demo.inputs import TEXT_PROMPT
from models.demos.devstral_small_2_24b_instruct_2512.mesh import close_mesh, open_mesh
from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", default=TEXT_PROMPT, help="user message (sent through the chat template)")
    ap.add_argument("--batch", type=int, default=P.requested_batch(), help="rows per call (<= 32), one seed each")
    ap.add_argument("--seed-offset", type=int, default=0, help="row r samples with seed r + offset")
    ap.add_argument("--layers", type=int, default=None, help="cap every stack's depth (default: full model)")
    ap.add_argument("--show", type=int, default=4, help="how many rows to print")
    args = ap.parse_args(argv)

    mesh = open_mesh(trace_region_size=P.TRACE_REGION_SIZE)
    try:
        pipe = P.build_pipeline(mesh, layers=args.layers, batch=args.batch)
        processor = P.load_processor()
        seeds = [r + args.seed_offset for r in range(pipe.batch)]
        req = P.encode_request(processor, args.prompt, batch=pipe.batch, seeds=seeds, kv_capacity=pipe.kv_capacity)
        res = P.run_text_generation(pipe, req)
        texts = P.decode_text(processor, res["tokens"], pipe.eos_token_id)
        print(f"==USER 0 - OUTPUT ({res['steps']} decode steps, stop={'cap' if res['hit_cap'] else 'eos'})")
        for r, text in enumerate(texts[: args.show]):
            print(f"--- row {r} (seed {seeds[r]}) ---\n{text}\n")
        return texts
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
