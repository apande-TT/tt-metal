# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The on-device latent plumbing between the transformer and the VAE.

`Flux2KleinPipeline` ends each generation with `unpack_latents_with_ids` ->
BatchNorm de-normalise -> `_unpatchify_latents`, and starts each reference image
with the inverse.  In this pipeline all of that runs ON DEVICE so the heads stay
host-op-free, built out of a 0/1 selection matmul, a ROW_MAJOR view reshape,
`ttnn.upsample` and a 0/1 sub-pixel mask (a `ttnn.reshape` that splits a TILED axis
does not compile in this build).

This test is what makes that rewrite trustworthy: both directions are compared
against `tt/latents.py`, which `test_layout_parity.py` has already shown to be
bit-identical to the reference pipeline's own staticmethods.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.flux_2_klein_9b import host_inputs as L
from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.mesh import open_flux_mesh
from models.demos.flux_2_klein_9b.tt import pipeline as P

PCC_TARGET = 0.999


@pytest.fixture(scope="module")
def pipe():
    R.ensure_flux_imports()
    with open_flux_mesh() as device:
        yield P.build_pipeline(device, model={"vae": R.load_vae()})


@pytest.mark.parametrize("batch", [1, R.BATCH])
@pytest.mark.parametrize("size", [256, 512])
def test_unpack_denormalise_unpatchify_on_device(pipe, size, batch):
    h, w = L.latent_grid(size, size)
    gen = torch.Generator("cpu").manual_seed(0)
    packed = torch.randn(batch, h * w, 128, generator=gen).to(torch.bfloat16)

    mean, std = L.bn_stats(R.load_vae())
    ref = L.unpack_latents_with_ids(packed.float(), L.latent_ids(batch, h, w), h, w)
    ref = L.unpatchify_latents(ref * std + mean)

    got = P._to_torch(pipe._latents_to_nchw(P._replicate(packed, pipe.device), h, w), pipe.device).float()
    assert tuple(got.shape) == tuple(ref.shape)
    per_sample = R.per_sample_pcc(got, ref)
    value = min(per_sample)
    print(f"unpatchify PCC={value} ({size}px, grid {h}x{w}, batch {batch}) worst of {len(per_sample)}")
    assert value >= PCC_TARGET


@pytest.mark.parametrize("batch", [1, R.BATCH])
@pytest.mark.parametrize("size", [256, 512])
def test_patchify_normalise_pack_on_device(pipe, size, batch):
    h, w = L.latent_grid(size, size)
    gen = torch.Generator("cpu").manual_seed(1)
    nchw = torch.randn(batch, 32, 2 * h, 2 * w, generator=gen).to(torch.bfloat16)

    mean, std = L.bn_stats(R.load_vae())
    ref = L.pack_latents((L.patchify_latents(nchw.float()) - mean) / std)

    got = P._to_torch(pipe._nchw_to_latents(P._replicate(nchw, pipe.device)), pipe.device).float()
    assert tuple(got.shape) == tuple(ref.shape)
    per_sample = R.per_sample_pcc(got, ref)
    value = min(per_sample)
    print(f"patchify PCC={value} ({size}px, grid {h}x{w}, batch {batch}) worst of {len(per_sample)}")
    assert value >= PCC_TARGET


def test_euler_step_matches_the_scheduler(pipe):
    """`stochastic_sampling: false` => `sample + dt * model_output`, upcast then back."""
    gen = torch.Generator("cpu").manual_seed(2)
    sample = torch.randn(R.BATCH, 256, 128, generator=gen).to(torch.bfloat16)
    pred = torch.randn(R.BATCH, 256, 128, generator=gen).to(torch.bfloat16)

    scheduler = R.load_scheduler()
    _, sigmas = L.schedule(scheduler, 4, image_seq_len=256)
    dt = L.euler_deltas(sigmas)[0]
    ref = (sample.float() + dt * pred.float()).to(torch.bfloat16).float()

    got = P._to_torch(
        pipe._euler_step(P._replicate(sample, pipe.device), P._replicate(pred, pipe.device), dt),
        pipe.device,
    ).float()
    value = min(R.per_sample_pcc(got, ref))
    print(f"euler PCC={value} (dt={dt}, batch {R.BATCH}, worst row)")
    assert value >= 0.9999


def test_row_major_ids_assertion_is_live(expect_error):
    """The device unpack is a reshape only because the kept tokens carry the
    row-major grid ids; the guard must actually fire if that stops being true."""
    good = L.latent_ids(R.BATCH, 16, 16)
    P.Flux2KleinTtPipeline._assert_row_major_ids(good, 16, 16)
    with expect_error(AssertionError, "latent ids are not the row-major grid"):
        P.Flux2KleinTtPipeline._assert_row_major_ids(good.flip(1), 16, 16)
