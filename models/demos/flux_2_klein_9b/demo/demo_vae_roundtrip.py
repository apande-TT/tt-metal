# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Call 4 -- image -> image through the checkpoint's own latent codec.

`AutoencoderKLFlux2` encode/decode is a real task head of this repo (it is what
turns pixels into the transformer's latents and back).  This demo runs the FULLY
DECOMPOSED routes -- every VAE sub-block stub at its own position in the network.

    ./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_vae_roundtrip \
        --image photo.png -o recon.png --compare-hf
"""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.mesh import open_flux_mesh
from models.demos.flux_2_klein_9b.tt.pipeline import build_pipeline


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLUX.2-klein-9B VAE round trip on a 1x8 mesh")
    p.add_argument("--image", default=None, help="input image path (default: a synthetic test pattern)")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=1, help="round-trip N INDEPENDENT images in one pass")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("-o", "--out", default="flux2_klein_vae_recon.png")
    p.add_argument("--compare-hf", action="store_true")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    R.ensure_flux_imports()

    if args.image:
        image = [Image.open(args.image).convert("RGB")] * args.batch
    else:
        image = R.batch_images(args.batch, args.size)

    with open_flux_mesh() as device:
        pipe = build_pipeline(device, layers=args.layers, batch=args.batch)
        start = time.time()
        recon = pipe.run_vae_roundtrip(image, height=args.size, width=args.size)
        elapsed = time.time() - start

    pils = R.image_processor().postprocess(recon, output_type="pil")
    outs = []
    for i, pil in enumerate(pils):
        name = args.out if len(pils) == 1 else args.out.replace(".png", f"_{i}.png")
        pil.save(name)
        outs.append(name)
    print(f"input       : {'synthetic test patterns' if not args.image else args.image}")
    print(f"recon       : {tuple(recon.shape)} -> {', '.join(outs)}")
    print(f"tt wall time: {elapsed:.1f}s at {args.size}x{args.size}, batch {args.batch}")

    if args.compare_hf:
        pixel = torch.cat([R.preprocess_image(im, args.size, args.size) for im in image], dim=0)
        golden, _ = R.hf_vae_roundtrip(pixel)
        per_sample = R.per_sample_pcc(recon, golden)
        if len(per_sample) > 1:
            print(f"per-sample PCC={[round(v, 5) for v in per_sample]}")
        print(f"e2e PCC={min(per_sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
