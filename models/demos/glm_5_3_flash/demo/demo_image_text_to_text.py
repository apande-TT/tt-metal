# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash image-text-to-text demo on a 32-chip Blackhole Galaxy (TP=32).

    python -m models.demos.glm_5_3_flash.demo.demo_image_text_to_text
    python -m models.demos.glm_5_3_flash.demo.demo_image_text_to_text --image <url-or-path> --prompt "..." --batch 32 --sample

Runs the SAME chained pipeline the e2e test gates (models/demos/glm_5_3_flash/tt/pipeline.py): HF processor ->
vision tower -> 45-layer text model -> lm_head -> on-device token choice, decoding until every sample emits an
eos_token_id from generation_config.
"""
from __future__ import annotations

import argparse
import time

import ttnn
from models.demos.glm_5_3_flash.tt import pipeline as P


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--image", default=P.EXAMPLE_IMAGE_URL, help="image URL or local path (default: the published example)"
    )
    ap.add_argument("--prompt", default=P.EXAMPLE_PROMPT)
    ap.add_argument("--batch", type=int, default=None, help="samples per call (default: $TT_PERF_BATCH or 1)")
    ap.add_argument(
        "--sample", action="store_true", help="sample with generation_config temperature/top_p, seed = row index"
    )
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--max-image-tokens", type=int, default=P.DEFAULT_MAX_IMAGE_TOKENS)
    ap.add_argument(
        "--reasoning-effort", choices=["low", "high", "max"], default=None, help="chat-template thinking budget"
    )
    ap.add_argument("--max-new-tokens", type=int, default=None, help="optional cap (default: decode until eos)")
    ap.add_argument(
        "--layers", type=int, default=None, help="cap every repeated stack (profiling only; not the real model)"
    )
    return ap.parse_args(argv)


def open_mesh():
    n = ttnn.get_num_devices()
    if n >= 32:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        return ttnn.open_mesh_device(ttnn.MeshShape(1, 32), l1_small_size=24576, trace_region_size=P.TRACE_REGION_SIZE)
    print(f"[demo] only {n} device(s) visible: falling back to a single device (the full model needs the 32-chip mesh)")
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main(argv=None):
    args = parse_args(argv)
    chat = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    processor = P.load_processor()
    enc = P.encode_example(
        processor, prompt=args.prompt, image_url=args.image, max_image_tokens=args.max_image_tokens, chat_kwargs=chat
    )
    mesh = open_mesh()
    try:
        kw = {} if args.batch is None else {"batch": args.batch}
        pipe = P.build_pipeline(mesh, layers=args.layers, **kw)
        e = pipe.encode(enc)
        sampling = None
        if args.sample:
            sampling = P.Sampling.from_generation_config(
                pipe.generation_config, seeds=range(args.seed_offset, args.seed_offset + e.B)
            )
        t0 = time.time()
        res = P.run_image_text_to_text(
            pipe, e, max_new_tokens=args.max_new_tokens, keep_logits=False, sampling=sampling
        )
        dt = time.time() - t0
        texts = processor.tokenizer.batch_decode(P.trim_at_eos(res.tokens, pipe.eos_ids), skip_special_tokens=False)
        for b, t in enumerate(texts):
            print(f"==== sample {b} ({'eos' if res.stopped_on_eos[b] else 'no eos'}) ====\n{t}")
        print(f"[demo] {e.B} sample(s), prompt {e.real_len} tokens, {res.tokens.shape[1]} decode steps, {dt:.1f}s")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
