# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Call 1 -- text -> image on Tenstorrent hardware.

    ./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_text_to_image \
        --prompt "a red apple on a wooden table" --height 256 --width 256 --steps 4 -o apple.png

The chained forward pass lives in `tt/pipeline.py` and is the SAME code the e2e
test runs, so a green test means a working demo.
"""

from __future__ import annotations

import argparse
import time

from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.mesh import open_flux_mesh
from models.demos.flux_2_klein_9b.tt.pipeline import build_pipeline


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLUX.2-klein-9B text-to-image on a 1x8 Tenstorrent mesh")
    p.add_argument("--prompt", default="a red apple on a wooden table", help="the text prompt")
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="run N INDEPENDENT samples in one pass (N distinct prompts from "
        "reference.batch_prompts and N distinct noise draws); --prompt is used when N == 1",
    )
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=4, help="denoise steps (Klein is step-distilled)")
    p.add_argument("--max-sequence-length", type=int, default=128, help="tokenizer padding length")
    p.add_argument("--seed", type=int, default=0, help="seed for the initial latent noise")
    p.add_argument("--layers", type=int, default=None, help="cap the depth of every repeated block")
    p.add_argument("-o", "--out", default="flux2_klein_t2i.png")
    p.add_argument("--compare-hf", action="store_true", help="also run the HF golden and print the PCC")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    R.ensure_flux_imports()

    with open_flux_mesh() as device:
        pipe = build_pipeline(device, layers=args.layers, batch=args.batch)
        prompt = args.prompt if args.batch == 1 else R.batch_prompts(args.batch)
        latents = R.batch_latents(args.batch, args.height, args.width, args.seed)

        start = time.time()
        image = pipe.run_text_to_image(
            prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            max_sequence_length=args.max_sequence_length,
            latents=latents,
        )
        elapsed = time.time() - start

    pils = R.image_processor().postprocess(image, output_type="pil")
    outs = _save(pils, args.out)
    print(f"prompt      : {prompt!r}")
    print(f"image       : {tuple(image.shape)} -> {', '.join(outs)}")
    print(
        f"tt wall time: {elapsed:.1f}s for {args.steps} steps at {args.height}x{args.width}, "
        f"batch {args.batch} ({elapsed / max(args.batch, 1):.1f}s per sample)"
    )

    if args.compare_hf:
        golden = R.hf_text_to_image(
            prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            latents=latents,
            max_sequence_length=args.max_sequence_length,
        )
        per_sample = R.per_sample_pcc(image, golden)
        if len(per_sample) > 1:
            print(f"per-sample PCC={[round(v, 5) for v in per_sample]}")
        print(f"e2e PCC={min(per_sample)}")
        _save(R.image_processor().postprocess(golden, output_type="pil"), args.out.replace(".png", "_hf.png"))
    return 0


def _save(pils, path: str) -> list[str]:
    """One file for a single sample; `<stem>_<i><ext>` for a batch."""
    if len(pils) == 1:
        pils[0].save(path)
        return [path]
    stem, _, ext = path.rpartition(".")
    names = [f"{stem}_{i}.{ext}" for i in range(len(pils))]
    for pil, name in zip(pils, names):
        pil.save(name)
    return names


if __name__ == "__main__":
    raise SystemExit(main())
