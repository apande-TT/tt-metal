# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the image_edit call, encoded exactly as HF QwenImageEditPipeline.__call__ encodes them.

The CONTENT is the model's own published example, used verbatim for every sample; the samples differ
only in the seed of the generator the example passes (sample i uses EXAMPLE_SEED + i, so sample 0 is
the published run). Everything here is host-side input encoding (processor, image resize, the seeded
initial noise, the scheduler's sigma table); the model math runs on device in tt/pipeline.py.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen-Image-Edit"

# ---- the published example (one block; each value carries its source) --------------------------------
# source: diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline docstring example 1
EXAMPLE_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
)
# source: diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline docstring example 1
EXAMPLE_PROMPT = "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant colors"
# source: diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline docstring example 1
EXAMPLE_NUM_INFERENCE_STEPS = 50
# source: Qwen/Qwen-Image-Edit README.md example 1 (generator=torch.manual_seed(0))
EXAMPLE_SEED = 0
# The docstring example passes no negative_prompt, so HF runs it without true CFG (do_true_cfg needs a
# negative prompt); true_cfg_scale keeps the pipeline default and is inert.
EXAMPLE_NEGATIVE_PROMPT = None
INPUTS_PROVENANCE = (
    "32 x the QwenImageEditPipeline docstring example 1 (yarn-art-pikachu.png + its prompt, 50 steps), "
    "seeds 0..31 (seed 0 from the Qwen-Image-Edit README example 1)"
)

# The pipeline sizes the condition image to a 1024 x 1024 pixel area (calculate_dimensions(1024 * 1024)).
# The golden runs this HF pipeline in float32 on CPU, ~10 s per sample-forward at a 256 x 256 area; at
# 1024 x 1024 (~9.5k joint tokens) it is ~140 h for 32 samples x 50 steps. So the gate runs the same
# pipeline with that ONE target area set to DEFAULT_AREA on both sides. demo --area overrides it.
PIPELINE_AREA = 1024
DEFAULT_AREA = 256

PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the image. Generate a new "
    "image that meets the user's requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP = 64  # QwenImageEditPipeline.prompt_template_encode_start_idx

_CACHE = Path(os.environ.get("QIE_CACHE_DIR", Path(__file__).resolve().parents[1] / "reference" / "_golden"))


def model_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.jinja"])


def load_example_image(url=EXAMPLE_IMAGE_URL):
    """The example's image, downloaded once and kept beside the golden cache."""
    from PIL import Image

    _CACHE.mkdir(parents=True, exist_ok=True)
    local = _CACHE / Path(url).name
    if not local.exists():
        from diffusers.utils import load_image

        load_image(url).save(local)
    return Image.open(local).convert("RGB")


def seeds_for(batch, base=EXAMPLE_SEED):
    return [base + i for i in range(batch)]


def batch_size_from_env(default=32):
    """The batch the harness asks for ($TT_PERF_BATCH), else `default`."""
    from models.experimental.perf_automation.agent.perf_adapter import BATCH_ENV

    v = os.environ.get(BATCH_ENV, "")
    return int(v) if v.strip() else int(default)


def calculated_size(image, area):
    """(width, height) the pipeline derives for the condition image at a target pixel area."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions

    w, h, _ = calculate_dimensions(area * area, image.size[0] / image.size[1])
    return int(w), int(h)


@dataclass
class EncodedInputs:
    """Host tensors for one batched call (everything the pipeline uploads before its forward)."""

    batch: int
    seeds: list
    prompts: list
    images: list  # PIL, resized to (width, height)
    width: int
    height: int
    num_inference_steps: int
    vl: dict  # Qwen2VLProcessor output: input_ids, attention_mask, pixel_values, image_grid_thw
    vae_image: torch.Tensor  # [B, 3, 1, H, W] in [-1, 1] (VaeImageProcessor.preprocess)
    latents: torch.Tensor  # [B, S_img, 64] packed initial noise, one generator per sample
    timesteps: torch.Tensor  # [N] scheduler timesteps (sigma * 1000)
    sigmas: torch.Tensor  # [N + 1] scheduler sigmas (terminal 0 appended)
    latent_hw: tuple  # (h, w) of the unpacked latent
    img_shapes: list = field(default_factory=list)

    def key(self):
        h = hashlib.sha256()
        h.update(repr((self.prompts, self.seeds, self.width, self.height, self.num_inference_steps)).encode())
        buf = io.BytesIO()
        np.save(buf, np.asarray(self.images[0]))
        h.update(buf.getvalue())
        return h.hexdigest()[:10]


def encode_inputs(
    batch,
    area=DEFAULT_AREA,
    prompt=EXAMPLE_PROMPT,
    image=None,
    base_seed=EXAMPLE_SEED,
    num_inference_steps=EXAMPLE_NUM_INFERENCE_STEPS,
):
    """HF QwenImageEditPipeline.__call__ steps 3-5 on the host: resize, VL processor, VAE preprocess,
    seeded noise (randn_tensor with one generator per sample), FlowMatch sigma schedule."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
        QwenImageEditPipeline,
        calculate_shift,
        retrieve_timesteps,
    )
    from diffusers.utils.torch_utils import randn_tensor
    from transformers import Qwen2VLProcessor

    root = model_path()
    image = load_example_image() if image is None else image
    width, height = calculated_size(image, area)
    vae_scale = 8  # 2 ** len(vae.temperal_downsample)
    multiple = vae_scale * 2
    width, height = width // multiple * multiple, height // multiple * multiple

    ip = VaeImageProcessor(vae_scale_factor=vae_scale * 2)
    resized = ip.resize(image, height, width)
    images = [resized] * batch
    prompts = [prompt] * batch
    seeds = seeds_for(batch, base_seed)

    processor = Qwen2VLProcessor.from_pretrained(root, subfolder="processor")
    vl = processor(text=[PROMPT_TEMPLATE.format(p) for p in prompts], images=images, padding=True, return_tensors="pt")
    vl = {k: vl[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    vae_image = ip.preprocess(images, height, width).unsqueeze(2).to(torch.float32)

    h_lat, w_lat = 2 * (height // multiple), 2 * (width // multiple)
    gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    noise = randn_tensor((batch, 1, 16, h_lat, w_lat), generator=gens, device=torch.device("cpu"), dtype=torch.float32)
    latents = QwenImageEditPipeline._pack_latents(noise, batch, 16, h_lat, w_lat)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    sig = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        latents.shape[1],
        sched.config.get("base_image_seq_len", 256),
        sched.config.get("max_image_seq_len", 4096),
        sched.config.get("base_shift", 0.5),
        sched.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(sched, num_inference_steps, torch.device("cpu"), sigmas=sig, mu=mu)
    img_shapes = [(1, h_lat // 2, w_lat // 2), (1, height // multiple, width // multiple)]
    return EncodedInputs(
        batch=batch,
        seeds=seeds,
        prompts=prompts,
        images=images,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        vl=vl,
        vae_image=vae_image,
        latents=latents,
        timesteps=timesteps.to(torch.float32),
        sigmas=sched.sigmas.to(torch.float32),
        latent_hw=(h_lat, w_lat),
        img_shapes=img_shapes,
    )
