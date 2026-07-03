# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for ACE-Step host guidance (APG/ADG/CFG)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from models.tt_dit.pipelines.acestep.apg_guidance import (
    AceStepGuidanceConfig,
    MomentumBuffer,
    adg_forward,
    adg_w_norm_forward,
    adg_wo_clip_forward,
    apg_forward,
    apply_acestep_guidance,
    cfg_forward,
    should_apply_cfg_interval,
)
from models.tt_dit.pipelines.cfg import (
    combine_cfg_host,
    double_batch_for_cfg,
    split_cfg_combiner_batch,
    split_hf_cfg_batch,
)

_HF_APG_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--ACE-Step--acestep-v15-base/snapshots"
    / "e432212fec32b8965a14ffa57ae653438d6abd14/apg_guidance.py"
)


def _load_hf_apg_reference():
    if not _HF_APG_PATH.is_file():
        pytest.skip(f"HF apg_guidance.py not found at {_HF_APG_PATH}")
    spec = importlib.util.spec_from_file_location("hf_apg_guidance_ref", _HF_APG_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def velocity_pair():
    """ACE-Step latent velocity shape [N, T, d]."""
    g = torch.Generator().manual_seed(42)
    n, t, d = 1, 50, 64
    pred_cond = torch.randn(n, t, d, generator=g, dtype=torch.float32)
    pred_uncond = torch.randn(n, t, d, generator=g, dtype=torch.float32)
    latents = torch.randn(n, t, d, generator=g, dtype=torch.float32)
    return latents, pred_cond, pred_uncond


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("guidance_scale", [1.0, 3.5, 7.0])
def test_apg_forward_matches_hf_reference(velocity_pair, guidance_scale):
    hf = _load_hf_apg_reference()
    latents, pred_cond, pred_uncond = velocity_pair
    del latents

    buf_ours = MomentumBuffer()
    buf_hf = hf.MomentumBuffer()
    ours = apg_forward(
        pred_cond,
        pred_uncond,
        guidance_scale=guidance_scale,
        momentum_buffer=buf_ours,
        dims=[1],
    )
    theirs = hf.apg_forward(
        pred_cond,
        pred_uncond,
        guidance_scale=guidance_scale,
        momentum_buffer=buf_hf,
        dims=[1],
    )
    assert _max_abs_diff(ours, theirs) < 1e-5


def test_adg_forward_matches_hf_reference(velocity_pair):
    hf = _load_hf_apg_reference()
    latents, pred_cond, pred_uncond = velocity_pair
    guidance_scale = 7.0
    sigma = 0.75

    ours = adg_forward(latents, pred_cond, pred_uncond, sigma=sigma, guidance_scale=guidance_scale)
    theirs = hf.adg_forward(latents, pred_cond, pred_uncond, sigma=sigma, guidance_scale=guidance_scale)
    assert _max_abs_diff(ours, theirs) < 1e-3


def test_adg_variants_match_hf_reference(velocity_pair):
    hf = _load_hf_apg_reference()
    latents, pred_cond, pred_uncond = velocity_pair
    guidance_scale = 5.0
    sigma = 0.5

    ours_norm = adg_w_norm_forward(latents, pred_cond, pred_uncond, sigma, guidance_scale)
    theirs_norm = hf.adg_w_norm_forward(latents, pred_cond, pred_uncond, sigma, guidance_scale)
    assert _max_abs_diff(ours_norm, theirs_norm) < 1e-3

    ours_noclip = adg_wo_clip_forward(latents, pred_cond, pred_uncond, sigma, guidance_scale)
    theirs_noclip = hf.adg_wo_clip_forward(latents, pred_cond, pred_uncond, sigma, guidance_scale)
    assert _max_abs_diff(ours_noclip, theirs_noclip) < 1e-3


def test_cfg_forward_matches_hf_reference(velocity_pair):
    hf = _load_hf_apg_reference()
    _, pred_cond, pred_uncond = velocity_pair
    cfg_strength = 4.0

    ours = cfg_forward(pred_cond, pred_uncond, cfg_strength)
    theirs = hf.cfg_forward(pred_cond, pred_uncond, cfg_strength)
    assert _max_abs_diff(ours, theirs) < 1e-6


def test_combine_cfg_host_matches_cfg_forward(velocity_pair):
    _, pred_cond, pred_uncond = velocity_pair
    cfg_scale = 6.0
    # ttnn.lerp(uncond, cond, t) == uncond + t*(cond-uncond)
    assert (
        _max_abs_diff(
            combine_cfg_host(pred_cond, pred_uncond, cfg_scale), cfg_forward(pred_cond, pred_uncond, cfg_scale)
        )
        < 1e-6
    )


def test_batch_split_orderings():
    batched = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    cond, uncond = split_hf_cfg_batch(batched, batch_size=2)
    assert cond.tolist() == [[1.0], [2.0]]
    assert uncond.tolist() == [[3.0], [4.0]]

    uncond_c, cond_c = split_cfg_combiner_batch(batched, batch_size=2)
    assert uncond_c.tolist() == [[3.0], [4.0]]
    assert cond_c.tolist() == [[1.0], [2.0]]


def test_double_batch_for_cfg():
    x = torch.arange(6, dtype=torch.float32).view(2, 3)
    doubled = double_batch_for_cfg(x)
    assert doubled.shape == (4, 3)
    assert torch.equal(doubled[:2], x)
    assert torch.equal(doubled[2:], x)


def test_should_apply_cfg_interval():
    assert should_apply_cfg_interval(0.5, 0.0, 1.0)
    assert not should_apply_cfg_interval(1.1, 0.0, 1.0)
    assert not should_apply_cfg_interval(-0.1, 0.0, 1.0)
    assert should_apply_cfg_interval(0.25, 0.2, 0.8)
    assert not should_apply_cfg_interval(0.1, 0.2, 0.8)


def test_apply_acestep_guidance_apg_default(velocity_pair):
    latents, pred_cond, pred_uncond = velocity_pair
    config = AceStepGuidanceConfig(guidance_scale=7.0, use_adg=False)
    buf_guided = MomentumBuffer()
    buf_expected = MomentumBuffer()

    guided = apply_acestep_guidance(
        latents=latents,
        pred_cond=pred_cond,
        pred_uncond=pred_uncond,
        timestep=0.5,
        config=config,
        momentum_buffer=buf_guided,
    )
    expected = apg_forward(pred_cond, pred_uncond, guidance_scale=7.0, momentum_buffer=buf_expected, dims=[1])
    assert _max_abs_diff(guided, expected) < 1e-6


def test_apply_acestep_guidance_adg(velocity_pair):
    latents, pred_cond, pred_uncond = velocity_pair
    config = AceStepGuidanceConfig(guidance_scale=7.0, use_adg=True, adg_angle_clip=3.14 / 6)

    guided = apply_acestep_guidance(
        latents=latents,
        pred_cond=pred_cond,
        pred_uncond=pred_uncond,
        timestep=0.6,
        config=config,
    )
    expected = adg_forward(latents, pred_cond, pred_uncond, sigma=0.6, guidance_scale=7.0, angle_clip=3.14 / 6)
    assert _max_abs_diff(guided, expected) < 1e-5


def test_apply_acestep_guidance_outside_interval_returns_cond(velocity_pair):
    latents, pred_cond, pred_uncond = velocity_pair
    config = AceStepGuidanceConfig(cfg_interval_start=0.2, cfg_interval_end=0.8)

    guided = apply_acestep_guidance(
        latents=latents,
        pred_cond=pred_cond,
        pred_uncond=pred_uncond,
        timestep=0.1,
        config=config,
    )
    assert torch.equal(guided, pred_cond)


def test_momentum_buffer_stateful_across_steps(velocity_pair):
    _, pred_cond, pred_uncond = velocity_pair
    buf = MomentumBuffer(momentum=-0.75)

    first = apg_forward(pred_cond, pred_uncond, guidance_scale=7.0, momentum_buffer=buf, dims=[1])
    second = apg_forward(pred_cond * 0.5, pred_uncond * 0.5, guidance_scale=7.0, momentum_buffer=buf, dims=[1])

    buf_reset = MomentumBuffer(momentum=-0.75)
    apg_forward(pred_cond, pred_uncond, guidance_scale=7.0, momentum_buffer=buf_reset, dims=[1])
    second_reset = apg_forward(
        pred_cond * 0.5, pred_uncond * 0.5, guidance_scale=7.0, momentum_buffer=buf_reset, dims=[1]
    )

    assert not torch.equal(first, second)
    assert _max_abs_diff(second, second_reset) < 1e-5
