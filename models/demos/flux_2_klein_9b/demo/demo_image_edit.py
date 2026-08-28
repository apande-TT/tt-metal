# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Call 3 -- text + reference image(s) -> image on Tenstorrent hardware.

FLUX.2's headline capability: the Klein pipeline accepts a LIST of reference
images, gives each one its own T coordinate on the rope axes, and concatenates
their VAE latents into the denoised stream.  Each reference is encoded by a
different graduated VAE route (the composite `encoder`, its alias
`encoder_stack`, and the block-wise decomposition), and every one of those
latents feeds the final image.

    ./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_image_edit \
        --prompt "make it a watercolour painting" --image a.png --image b.png --steps 2 -o edit.png

With no --image the demo uses three deterministic synthetic references so it runs
without any asset.
"""

from __future__ import annotations

import argparse
import time

from PIL import Image

from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.mesh import open_flux_mesh
from models.demos.flux_2_klein_9b.tt.pipeline import _demo_image, build_pipeline


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLUX.2-klein-9B multi-reference image editing on a 1x8 mesh")
    p.add_argument("--prompt", default="make it a watercolour painting")
    p.add_argument("--image", action="append", default=[], help="reference image path (repeatable)")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--max-sequence-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="edit N INDEPENDENT samples in one pass -- N distinct prompts and N "
        "distinct noise draws against the SAME reference slots, which is what the "
        "reference pipeline does (image=[...] is a list of slots, not of samples)",
    )
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("-o", "--out", default="flux2_klein_edit.png")
    p.add_argument("--compare-hf", action="store_true")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    R.ensure_flux_imports()

    if args.image:
        images = [Image.open(path).convert("RGB") for path in args.image]
    else:
        # three distinct synthetic references, so the multi-reference path is real
        images = [_demo_image(args.height) for _ in range(3)]
        images[1] = images[1].rotate(90)
        images[2] = images[2].transpose(Image.FLIP_LEFT_RIGHT)

    with open_flux_mesh() as device:
        pipe = build_pipeline(device, layers=args.layers, batch=args.batch)
        prompt = args.prompt if args.batch == 1 else R.batch_prompts(args.batch)
        latents = R.batch_latents(args.batch, args.height, args.width, args.seed)
        start = time.time()
        image = pipe.run_image_edit(
            prompt,
            images,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            max_sequence_length=args.max_sequence_length,
            latents=latents,
        )
        elapsed = time.time() - start

    pils = R.image_processor().postprocess(image, output_type="pil")
    outs = []
    for i, pil in enumerate(pils):
        name = args.out if len(pils) == 1 else args.out.replace(".png", f"_{i}.png")
        pil.save(name)
        outs.append(name)
    print(f"prompt      : {prompt!r}")
    print(f"references  : {len(images)} slot(s), shared across the batch")
    print(f"image       : {tuple(image.shape)} -> {', '.join(outs)}")
    print(f"tt wall time: {elapsed:.1f}s for {args.steps} steps at {args.height}x{args.width}, " f"batch {args.batch}")

    if args.compare_hf:
        golden = R.hf_image_edit(
            prompt,
            images,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
