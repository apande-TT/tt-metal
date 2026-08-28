# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The BATCHED e2e gates for the three heads that run the VAE DECODER.

They are a separate module because they need a separate DEVICE, not because they are a
separate task: the batched VAE's convolutions need a larger `l1_small_size` halo, and
`l1_small_size` is fixed when the mesh is opened.  Raising it for every head is not
free -- it pushes the L1-resident activations of the heads that need no halo into the
circular-buffer region, and doing that globally broke the text head at B=32 even though
it has no VAE at all.  So the halo is scoped to the heads that need it.

The batch here is 16, not 32, and the reason is measured rather than assumed -- see
`BATCH_VAE` in `test_e2e_pipeline.py`.  Everything else is identical to the B=1 gates:
real inputs, the same chained pipeline, per-sample PCC against each sample's own
golden, and a pairwise-distinctness check.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.mesh import open_flux_mesh
from models.demos.flux_2_klein_9b.tests.e2e.test_e2e_pipeline import (
    BATCH_VAE,
    EDIT_BATCH,
    PCC_TARGET,
    T2I_BATCH,
    VAE_L1_SMALL,
    VAE_SIZE,
    _edit_references,
    _report_batch,
)
from models.demos.flux_2_klein_9b.tt.pipeline import build_pipeline


@pytest.fixture(scope="module")
def pipe():
    R.ensure_flux_imports()
    with open_flux_mesh(l1_small_size=VAE_L1_SMALL) as device:
        yield build_pipeline(device, layers=_env_layers(), batch=BATCH_VAE)
    R.release()


def _env_layers():
    value = os.environ.get("TT_PERF_LAYERS") or os.environ.get("FLUX2_E2E_LAYERS")
    return int(value) if value else None


@pytest.mark.timeout(5400)  # pytest.ini caps at 300s; the B=32 CPU golden alone is ~600s
def test_call_1_text_to_image_batch(pipe):
    """`BATCH_VAE` distinct prompts -> that many distinct images, one denoise program per step."""
    prompts = R.batch_prompts(BATCH_VAE)
    latents = R.batch_latents(BATCH_VAE, T2I_BATCH["height"], T2I_BATCH["width"], T2I_BATCH["seed"])

    got = pipe.run_text_to_image(prompts, latents=latents, **T2I_BATCH)
    golden = R.hf_text_to_image(
        prompts,
        height=T2I_BATCH["height"],
        width=T2I_BATCH["width"],
        num_inference_steps=T2I_BATCH["num_inference_steps"],
        latents=latents,
        max_sequence_length=T2I_BATCH["max_sequence_length"],
    )
    assert tuple(got.shape) == tuple(golden.shape), (got.shape, golden.shape)
    assert int(got.shape[0]) == BATCH_VAE, f"expected {BATCH_VAE} samples, got {got.shape[0]}"

    per_sample = R.per_sample_pcc(got, golden)
    worst_pair = R.assert_samples_are_distinct(got)
    worst = _report_batch("call_1_text_to_image", per_sample, worst_pair)
    assert (
        worst >= PCC_TARGET
    ), f"Call 1 batch{BATCH_VAE}: sample {per_sample.index(worst)} scored {worst} < {PCC_TARGET}"


@pytest.mark.timeout(5400)  # pytest.ini caps at 300s; the B=32 CPU golden alone is ~600s
def test_call_3_image_edit_batch(pipe):
    """`BATCH_VAE` distinct prompts + the SAME three reference slots.

    Sharing the references across the batch is what the reference pipeline itself
    does -- `Flux2KleinPipeline.__call__` treats `image=[...]` as a list of reference
    SLOTS applied to every prompt, with no per-sample reference support -- so the `BATCH_VAE`
    samples are independent through their prompts and their noise, and the gate is
    faithful to Source A rather than to a shape we invented.
    """
    prompts = R.batch_prompts(BATCH_VAE)
    images = _edit_references(EDIT_BATCH["height"])
    latents = R.batch_latents(BATCH_VAE, EDIT_BATCH["height"], EDIT_BATCH["width"], EDIT_BATCH["seed"])

    got = pipe.run_image_edit(prompts, images, latents=latents, **EDIT_BATCH)
    golden = R.hf_image_edit(
        prompts,
        images,
        height=EDIT_BATCH["height"],
        width=EDIT_BATCH["width"],
        num_inference_steps=EDIT_BATCH["num_inference_steps"],
        latents=latents,
        max_sequence_length=EDIT_BATCH["max_sequence_length"],
    )
    assert tuple(got.shape) == tuple(golden.shape), (got.shape, golden.shape)
    assert int(got.shape[0]) == BATCH_VAE

    per_sample = R.per_sample_pcc(got, golden)
    worst_pair = R.assert_samples_are_distinct(got)
    worst = _report_batch("call_3_image_edit", per_sample, worst_pair)
    assert (
        worst >= PCC_TARGET
    ), f"Call 3 batch{BATCH_VAE}: sample {per_sample.index(worst)} scored {worst} < {PCC_TARGET}"


@pytest.mark.timeout(5400)  # pytest.ini caps at 300s; the B=32 CPU golden alone is ~600s
def test_call_4_vae_roundtrip_batch(pipe):
    """32 DISTINCT images through the decomposed codec in one pass."""
    images = R.batch_images(BATCH_VAE, VAE_SIZE)
    got = pipe.run_vae_roundtrip(images, height=VAE_SIZE, width=VAE_SIZE)

    pixel = torch.cat([R.preprocess_image(im, VAE_SIZE, VAE_SIZE) for im in images], dim=0)
    golden, _ = R.hf_vae_roundtrip(pixel)
    assert tuple(got.shape) == tuple(golden.shape), (got.shape, golden.shape)
    assert int(got.shape[0]) == BATCH_VAE

    per_sample = R.per_sample_pcc(got, golden)
    worst_pair = R.assert_samples_are_distinct(got)
    worst = _report_batch("call_4_vae_roundtrip", per_sample, worst_pair)
    assert (
        worst >= PCC_TARGET
    ), f"Call 4 batch{BATCH_VAE}: sample {per_sample.index(worst)} scored {worst} < {PCC_TARGET}"
