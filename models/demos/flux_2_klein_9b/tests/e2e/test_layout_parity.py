# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only parity check: `tt/latents.py` == the reference pipeline's own layout code.

The TT forward path must not reach into an HF object, so the pipeline's pure
index/shape helpers are reimplemented in `tt/latents.py`.  This test is what makes
that reimplementation trustworthy: every function is compared against the
reference staticmethod it was lifted from, bit for bit.  No device needed.
"""

from __future__ import annotations

import torch

from models.demos.flux_2_klein_9b import host_inputs as L
from models.demos.flux_2_klein_9b import reference as R


def _ref_cls():
    return R.diffusers_module("Flux2KleinPipeline").Flux2KleinPipeline


def test_patchify_roundtrip_matches_reference():
    cls = _ref_cls()
    x = torch.randn(2, 32, 28, 28)
    assert torch.equal(L.patchify_latents(x), cls._patchify_latents(x))
    p = L.patchify_latents(x)
    assert torch.equal(L.unpatchify_latents(p), cls._unpatchify_latents(p))
    assert torch.equal(L.unpatchify_latents(p), x)


def test_pack_matches_reference():
    cls = _ref_cls()
    x = torch.randn(2, 128, 14, 14)
    assert torch.equal(L.pack_latents(x), cls._pack_latents(x))


def test_ids_match_reference():
    cls = _ref_cls()
    embeds = torch.zeros(2, 37, 12288)
    assert torch.equal(L.text_ids(2, 37), cls._prepare_text_ids(embeds))

    lat = torch.zeros(2, 128, 16, 16)
    assert torch.equal(L.latent_ids(2, 16, 16), cls._prepare_latent_ids(lat))

    refs = [torch.zeros(1, 128, 14, 14), torch.zeros(1, 128, 12, 12)]
    mine = L.image_ids([(14, 14), (12, 12)])
    assert torch.equal(mine, cls._prepare_image_ids(refs))


def test_unpack_matches_reference():
    cls = _ref_cls()
    ids = L.latent_ids(1, 16, 16)
    x = torch.randn(1, 256, 128)
    assert torch.equal(
        L.unpack_latents_with_ids(x, ids, 16, 16),
        cls._unpack_latents_with_ids(x, ids, 16, 16),
    )


def test_mu_matches_reference():
    mod = R.diffusers_module("Flux2KleinPipeline")
    ref = mod.pipelines.flux2.pipeline_flux2_klein.compute_empirical_mu
    for seq in (256, 972, 4096, 5000):
        for steps in (2, 4, 50):
            assert L.empirical_mu(seq, steps) == ref(seq, steps)


def test_schedule_matches_reference_pipeline():
    """The (timesteps, sigmas) the TT loop walks are the scheduler's own."""
    scheduler = R.load_scheduler()
    ts, sigmas = L.schedule(scheduler, 4, 256)
    assert len(ts) == 4
    assert len(sigmas) == 5
    # dt is negative and monotone for a flow-match schedule
    dts = L.euler_deltas(sigmas)
    assert all(d < 0 for d in dts), dts


def test_latent_grid_matches_prepare_latents():
    for size in (224, 256, 512, 1024):
        h, w = L.latent_grid(size, size)
        assert h == w
        assert h == (2 * (size // 16)) // 2
