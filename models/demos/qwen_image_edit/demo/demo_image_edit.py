# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 demo: image + instruction -> edited image with Qwen-Image-Edit on TTNN (2x4 mesh).

Default input is the model's published example (docstring example 1: yarn-art-pikachu.png and its
prompt, 50 steps), one sample per seed starting at the README's seed 0. The chained forward is
tt/pipeline.run_image_edit -- the same function the e2e test gates.

    python -m models.demos.qwen_image_edit.demo.demo_image_edit --batch 4
    python -m models.demos.qwen_image_edit.demo.demo_image_edit --image photo.png --prompt "..." --batch 1
    python -m models.demos.qwen_image_edit.demo.demo_image_edit --batch 4 --compare-golden
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from models.demos.qwen_image_edit.tt import inputs as I


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=None, help="samples per call (default: $TT_PERF_BATCH or 32)")
    ap.add_argument("--image", default=None, help="input image path or URL (default: the published example)")
    ap.add_argument("--prompt", default=I.EXAMPLE_PROMPT)
    ap.add_argument("--seed", type=int, default=I.EXAMPLE_SEED, help="seed of sample 0 (sample i uses seed + i)")
    ap.add_argument("--steps", type=int, default=I.EXAMPLE_NUM_INFERENCE_STEPS)
    ap.add_argument("--area", type=int, default=I.DEFAULT_AREA, help="condition-image side (HF default: 1024)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "output"))
    ap.add_argument("--no-trace", action="store_true", help="run the denoising steps eagerly")
    ap.add_argument("--compare-golden", action="store_true", help="PCC vs the HF float32 golden (built if missing)")
    args = ap.parse_args(argv)

    import ttnn
    from models.demos.qwen_image_edit.demo.mesh import close_mesh, open_mesh
    from models.demos.qwen_image_edit.tt.pipeline import build_pipeline, run_image_edit, to_host

    B = args.batch if args.batch is not None else I.batch_size_from_env()
    image = None
    if args.image is not None:
        from diffusers.utils import load_image

        image = load_image(args.image).convert("RGB")
    enc = I.encode_inputs(
        B, area=args.area, prompt=args.prompt, image=image, base_seed=args.seed, num_inference_steps=args.steps
    )
    print(
        f"[demo] {B} sample(s), {enc.width}x{enc.height}, {len(enc.timesteps)} steps, seeds {enc.seeds[0]}..{enc.seeds[-1]}"
    )

    device = open_mesh()
    try:
        t0 = time.time()
        pipe = build_pipeline(device)
        print(f"[demo] pipeline built in {time.time() - t0:.0f} s", flush=True)
        t0 = time.time()
        out = run_image_edit(pipe, enc, use_trace=not args.no_trace)
        ttnn.synchronize_device(device)
        print(f"[demo] {B} image(s) in {time.time() - t0:.0f} s", flush=True)
        images = to_host(out["image"], device).float()
    finally:
        close_mesh(device)

    from PIL import Image

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    enc.images[0].save(outdir / "input.png")
    for i, s in enumerate(enc.seeds):
        arr = (images[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")
        Image.fromarray(arr).save(outdir / f"edit_seed{s:04d}.png")
    print(f"[demo] wrote {B} edited image(s) to {outdir}")

    if args.compare_golden:
        from models.demos.qwen_image_edit.reference.golden import load_golden
        from models.demos.qwen_image_edit.tests.e2e.test_e2e_image_edit import _pcc_rows

        g = load_golden(enc)
        pcc = _pcc_rows(images, g["image"].float())
        for i, s in enumerate(enc.seeds):
            print(f"[demo] seed {s}: image pcc vs HF {pcc[i].item():.6f}")
        print(f"[demo] min image pcc vs HF over {B} samples: {pcc.min().item():.6f}")


if __name__ == "__main__":
    main()
