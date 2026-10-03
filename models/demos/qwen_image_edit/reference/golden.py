# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF golden for the image_edit call: diffusers QwenImageEditPipeline, float32 on CPU.

The reference is the HF pipeline itself, called with the example's arguments (image, prompt,
num_inference_steps, one torch generator per sample). The ONE change is the condition-image target area
(see tt/inputs.py DEFAULT_AREA): HF hard-codes calculate_dimensions(1024 * 1024), and the golden rebinds
that module function to the gate's area. Stage outputs are recorded on the way (prompt embeddings, image
latents, the latents after every step) so a failing sample can be localised; the final image is the
pipeline's own output (output_type="pt", [0, 1]).

It runs in chunks of CHUNK consecutive seeds and caches each chunk, so a batch of 4 reuses the first
chunk of the batch-32 golden and an interrupted build resumes.

    python -m models.demos.qwen_image_edit.reference.golden --batch 32
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import numpy as np
import torch

from models.demos.qwen_image_edit.tt import inputs as I

CHUNK = 4
VERSION = 1


def _content_key(prompt, image, area, steps):
    h = hashlib.sha256()
    h.update(repr((VERSION, prompt, area, steps, I.MODEL_ID)).encode())
    h.update(np.asarray(image).tobytes())
    return h.hexdigest()[:10]


def _chunk_path(key, lo):
    return I._CACHE / f"golden_{key}_seeds{lo:04d}-{lo + CHUNK - 1:04d}.pt"


_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        from diffusers import QwenImageEditPipeline

        _PIPE = QwenImageEditPipeline.from_pretrained(I.MODEL_ID, torch_dtype=torch.float32)
        _PIPE.set_progress_bar_config(disable=False)
    return _PIPE


def _hf_reference_image_edit(prompt, image, seeds, area, steps):
    """Run the HF pipeline for these seeds; returns the recorded stage outputs + final images."""
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit as pqe

    pipe = _pipe()
    orig = pqe.calculate_dimensions
    rec = {"prompt_embeds": [], "prompt_embeds_mask": [], "image_latents": [], "step_latents": []}
    enc_prompt, enc_vae = pipe.encode_prompt, pipe._encode_vae_image

    def _encode_prompt(*a, **k):
        pe, pm = enc_prompt(*a, **k)
        rec["prompt_embeds"].append(pe.detach().clone())
        rec["prompt_embeds_mask"].append(None if pm is None else pm.detach().clone())
        return pe, pm

    def _encode_vae(*a, **k):
        out = enc_vae(*a, **k)
        rec["image_latents"].append(out.detach().clone())
        return out

    def _on_step(_pipe, i, t, kw):
        rec["step_latents"].append(kw["latents"].detach().clone())
        return {}

    w, h = I.calculated_size(image, area)
    resized = pipe.image_processor.resize(image, h, w)
    pqe.calculate_dimensions = lambda target_area, ratio: orig(area * area, ratio)
    pipe.encode_prompt, pipe._encode_vae_image = _encode_prompt, _encode_vae
    try:
        with torch.no_grad():
            out = pipe(
                image=[resized] * len(seeds),
                prompt=[prompt] * len(seeds),
                negative_prompt=I.EXAMPLE_NEGATIVE_PROMPT,
                num_inference_steps=steps,
                generator=[torch.Generator(device="cpu").manual_seed(s) for s in seeds],
                output_type="pt",
                callback_on_step_end=_on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
    finally:
        pqe.calculate_dimensions = orig
        pipe.encode_prompt, pipe._encode_vae_image = enc_prompt, enc_vae
    return {
        "seeds": list(seeds),
        "prompt_embeds": rec["prompt_embeds"][0],
        "prompt_embeds_mask": rec["prompt_embeds_mask"][0],
        "image_latents": rec["image_latents"][0],
        "step_latents": torch.stack(rec["step_latents"]),  # [steps, B, S_img, 64]
        "image": out.images.to(torch.float32),  # [B, 3, H, W] in [0, 1]
    }


def load_golden(enc: I.EncodedInputs, build=True):
    """Golden for exactly the samples of `enc` (same prompt, image, area and seeds), built if missing."""
    prompt, image = enc.prompts[0], enc.images[0]
    assert all(p == prompt for p in enc.prompts), "the golden is keyed on one content input"
    area = int(round((enc.width * enc.height) ** 0.5))
    key = _content_key(prompt, image, area, enc.num_inference_steps)
    I._CACHE.mkdir(parents=True, exist_ok=True)
    parts = {}
    for s in enc.seeds:
        lo = (s // CHUNK) * CHUNK
        if lo in parts:
            continue
        path = _chunk_path(key, lo)
        if not path.exists():
            if not build:
                raise FileNotFoundError(path)
            t0 = time.time()
            print(f"[golden] building seeds {lo}..{lo + CHUNK - 1} -> {path.name}", flush=True)
            g = _hf_reference_image_edit(prompt, image, list(range(lo, lo + CHUNK)), area, enc.num_inference_steps)
            g["prompt"] = prompt
            torch.save(g, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)
            print(f"[golden] seeds {lo}..{lo + CHUNK - 1} done in {time.time() - t0:.0f} s", flush=True)
        g = torch.load(path, weights_only=False)
        assert g["prompt"] == prompt and g["seeds"] == list(range(lo, lo + CHUNK)), f"stale golden {path}"
        parts[lo] = g
    out = {"seeds": list(enc.seeds)}
    for name in ("prompt_embeds", "image_latents", "image"):
        out[name] = torch.stack([parts[(s // CHUNK) * CHUNK][name][s % CHUNK] for s in enc.seeds])
    out["step_latents"] = torch.stack(
        [parts[(s // CHUNK) * CHUNK]["step_latents"][:, s % CHUNK] for s in enc.seeds], dim=1
    )
    out["prompt_embeds_mask"] = parts[(enc.seeds[0] // CHUNK) * CHUNK]["prompt_embeds_mask"]
    return out


def check_noise_matches_hf(enc: I.EncodedInputs):
    """The pipeline's initial noise == HF prepare_latents with the same generators (exact)."""
    pipe = _pipe()
    gens = [torch.Generator(device="cpu").manual_seed(s) for s in enc.seeds]
    h_lat, w_lat = enc.latent_hw
    lat, _ = pipe.prepare_latents(None, enc.batch, 16, h_lat * 8, w_lat * 8, torch.float32, "cpu", gens)
    assert torch.equal(lat, enc.latents), "initial noise differs from HF prepare_latents"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--area", type=int, default=I.DEFAULT_AREA)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    enc = I.encode_inputs(args.batch, area=args.area)
    check_noise_matches_hf(enc)
    g = load_golden(enc)
    print({k: tuple(v.shape) for k, v in g.items() if isinstance(v, torch.Tensor)})


if __name__ == "__main__":
    main()
