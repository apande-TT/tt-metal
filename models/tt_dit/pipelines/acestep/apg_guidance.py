# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side velocity guidance for ACE-Step v1.5 flow-matching denoising.

Ported from the HuggingFace reference
``ACE-Step/acestep-v15-base`` ``apg_guidance.py`` (production default: APG with
``dims=[1]`` on ``[N, T, d]`` velocity tensors).

Pipeline integration (Phase 3 device wiring, not implemented here):

1. When ``guidance_scale > 1``, double the DiT batch on device (cond first, null
   second — matches HF ``generate_audio``, *not* ``CFGCombiner``'s uncond-first
   convention; use ``split_hf_cfg_batch`` from ``cfg.py``).
2. Run one batched DiT forward per ODE step; pull velocities to host.
3. If ``cfg_interval_start <= t_curr <= cfg_interval_end``, call
   ``apply_acestep_guidance`` (APG by default, ADG when ``use_adg=True``).
4. Euler update on host: ``xt = xt - vt * (t_curr - t_prev)``.

See ``models/tt_dit/pipelines/cfg.py`` for host CFG helpers and batch-order notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "MomentumBuffer",
    "AceStepGuidanceConfig",
    "project",
    "apg_forward",
    "cfg_forward",
    "adg_forward",
    "adg_w_norm_forward",
    "adg_wo_clip_forward",
    "apply_acestep_guidance",
    "should_apply_cfg_interval",
]


class MomentumBuffer:
    """Exponential momentum buffer for APG diff smoothing (HF default momentum=-0.75)."""

    def __init__(self, momentum: float = -0.75) -> None:
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor) -> None:
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

    def reset(self) -> None:
        self.running_average = 0


def project(
    v0: torch.Tensor,
    v1: torch.Tensor,
    dims: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose ``v0`` into parallel and orthogonal parts relative to normalized ``v1``."""
    if dims is None:
        dims = [-1]
    dtype = v0.dtype
    device_type = v0.device.type
    if device_type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()

    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=list(dims))
    v0_parallel = (v0 * v1).sum(dim=list(dims), keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(device_type)


def apg_forward(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer | None = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """Adaptive Projected Guidance (APG) on velocity predictions.

    ACE-Step production uses ``dims=[1]`` for ``[N, T, d]`` tensors.
    """
    if dims is None:
        dims = [-1]
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=list(dims), keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond, dims=dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + (guidance_scale - 1) * normalized_update


def cfg_forward(cond_output: torch.Tensor, uncond_output: torch.Tensor, cfg_strength: float) -> torch.Tensor:
    """Plain classifier-free guidance (no APG projection)."""
    return uncond_output + cfg_strength * (cond_output - uncond_output)


def call_cos_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two tensors along dim=1."""
    tensor1 = tensor1 / torch.linalg.norm(tensor1, dim=1, keepdim=True)
    tensor2 = tensor2 / torch.linalg.norm(tensor2, dim=1, keepdim=True)
    return torch.sum(tensor1 * tensor2, dim=1, keepdim=True)


def compute_perpendicular_component(
    latent_diff: torch.Tensor,
    latent_hat_uncond: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose ``latent_diff`` into parallel and perpendicular parts vs ``latent_hat_uncond``."""
    n, t, c = latent_diff.shape
    latent_diff_flat = latent_diff.view(n * t, c).float()
    latent_hat_flat = latent_hat_uncond.view(n * t, c).float()

    if latent_diff_flat.size() != latent_hat_flat.size():
        msg = "latent_diff and latent_hat_uncond must have the same shape [n, d]."
        raise ValueError(msg)

    dot_product = torch.sum(latent_diff_flat * latent_hat_flat, dim=1, keepdim=True)
    norm_square = torch.sum(latent_hat_flat * latent_hat_flat, dim=1, keepdim=True)
    projection = (dot_product / (norm_square + 1e-8)) * latent_hat_flat
    perpendicular_component = latent_diff_flat - projection

    return projection.view(n, t, c), perpendicular_component.reshape(n, t, c)


def adg_forward(
    latents: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    sigma: torch.Tensor | float,
    guidance_scale: float,
    angle_clip: float = 3.14 / 6,
    apply_norm: bool = False,
    apply_clip: bool = True,
) -> torch.Tensor:
    """Angle-based Dynamic Guidance (ADG) for flow-matching velocity fields."""
    n = noise_pred_cond.shape[0]
    noise_pred_text = noise_pred_cond
    n, t, c = noise_pred_text.shape

    if isinstance(sigma, (int, float)):
        sigma_t = torch.tensor(sigma, device=latents.device, dtype=latents.dtype)
        sigma_t = sigma_t.view(1, 1, 1).expand(n, 1, 1)
    elif torch.is_tensor(sigma):
        sigma_t = sigma
        if sigma_t.numel() == 1:
            sigma_t = sigma_t.view(1, 1, 1).expand(n, 1, 1)
        elif sigma_t.numel() == n:
            sigma_t = sigma_t.view(n, 1, 1)
        else:
            msg = f"sigma has incompatible shape. Expected scalar or size {n}, got {sigma_t.shape}"
            raise ValueError(msg)
    else:
        msg = f"sigma must be a number or tensor, got {type(sigma)}"
        raise TypeError(msg)

    weight = guidance_scale - 1
    weight = weight * (weight > 0) + 1e-3

    latent_hat_text = latents - sigma_t * noise_pred_text
    latent_hat_uncond = latents - sigma_t * noise_pred_uncond
    latent_diff = latent_hat_text - latent_hat_uncond

    latent_theta = torch.acos(
        call_cos_tensor(
            latent_hat_text.view(-1, c).to(float),
            latent_hat_uncond.reshape(-1, c).contiguous().to(float),
        )
    )
    latent_theta_new = (
        torch.clip(weight * latent_theta, -angle_clip, angle_clip) if apply_clip else weight * latent_theta
    )
    _proj, perp = compute_perpendicular_component(latent_diff, latent_hat_uncond)
    latent_v_new = torch.cos(latent_theta_new) * latent_hat_text

    latent_p_new = perp * torch.sin(latent_theta_new) / torch.sin(latent_theta) * (
        torch.sin(latent_theta) > 1e-3
    ) + perp * weight * (torch.sin(latent_theta) <= 1e-3)
    latent_new = latent_v_new + latent_p_new
    if apply_norm:
        latent_new = (
            latent_new
            * torch.linalg.norm(latent_hat_text, dim=1, keepdim=True)
            / torch.linalg.norm(latent_new, dim=1, keepdim=True)
        )

    noise_pred = (latents - latent_new) / sigma_t
    return noise_pred.reshape(n, t, c).to(latents.dtype)


def adg_w_norm_forward(
    latents: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    sigma: float,
    guidance_scale: float,
    angle_clip: float = 3.14 / 3,
) -> torch.Tensor:
    return adg_forward(
        latents,
        noise_pred_cond,
        noise_pred_uncond,
        sigma,
        guidance_scale,
        angle_clip=angle_clip,
        apply_norm=True,
        apply_clip=True,
    )


def adg_wo_clip_forward(
    latents: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    sigma: float,
    guidance_scale: float,
) -> torch.Tensor:
    return adg_forward(
        latents,
        noise_pred_cond,
        noise_pred_uncond,
        sigma,
        guidance_scale,
        apply_norm=False,
        apply_clip=False,
    )


@dataclass(frozen=True)
class AceStepGuidanceConfig:
    """Production ACE-Step guidance knobs (HF ``generate_audio`` defaults)."""

    guidance_scale: float = 7.0
    use_adg: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    apg_eta: float = 0.0
    apg_norm_threshold: float = 2.5
    apg_momentum: float = -0.75
    adg_angle_clip: float = 3.14 / 6


def should_apply_cfg_interval(timestep: float, interval_start: float, interval_end: float) -> bool:
    """Return whether CFG/APG should run at ``timestep`` (HF ``cfg_interval_*`` gate)."""
    return interval_start <= timestep <= interval_end


def apply_acestep_guidance(
    *,
    latents: torch.Tensor,
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    timestep: float,
    config: AceStepGuidanceConfig,
    momentum_buffer: MomentumBuffer | None = None,
) -> torch.Tensor:
    """Apply APG or ADG to a single-batch velocity pair (host-side denoise step)."""
    if not should_apply_cfg_interval(timestep, config.cfg_interval_start, config.cfg_interval_end):
        return pred_cond

    if config.use_adg:
        return adg_forward(
            latents=latents,
            noise_pred_cond=pred_cond,
            noise_pred_uncond=pred_uncond,
            sigma=timestep,
            guidance_scale=config.guidance_scale,
            angle_clip=config.adg_angle_clip,
        )

    if momentum_buffer is None:
        momentum_buffer = MomentumBuffer(momentum=config.apg_momentum)

    return apg_forward(
        pred_cond=pred_cond,
        pred_uncond=pred_uncond,
        guidance_scale=config.guidance_scale,
        momentum_buffer=momentum_buffer,
        eta=config.apg_eta,
        norm_threshold=config.apg_norm_threshold,
        dims=[1],
    )
