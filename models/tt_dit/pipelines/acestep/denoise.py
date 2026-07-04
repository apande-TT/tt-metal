# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 3: flow-matching ODE denoise with optional CFG + APG (host guidance math)."""

from __future__ import annotations

from typing import Callable, Protocol

import torch

from models.demos.hf_eager.acestep_v15_base.tt.common import ode_timesteps, prepare_noise
from models.tt_dit.pipelines.acestep.apg_guidance import AceStepGuidanceConfig, MomentumBuffer, apply_acestep_guidance
from models.tt_dit.pipelines.cfg import merge_encoder_for_cfg


class DecoderForward(Protocol):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        *,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        ...


def _double_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x], dim=0)


def prepare_cfg_tensors(
    *,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
    null_condition_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """HF ``generate_audio`` batch layout: cond rows first, null rows second."""
    enc = merge_encoder_for_cfg(encoder_hidden_states, null_condition_emb)
    enc_mask = _double_batch(encoder_attention_mask)
    ctx = _double_batch(context_latents)
    attn = _double_batch(attention_mask)
    return enc, enc_mask, ctx, attn


def run_flow_matching_ode(
    *,
    decoder: DecoderForward,
    context_latents: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    infer_steps: int,
    seed: int | None,
    guidance_config: AceStepGuidanceConfig | None = None,
    null_condition_emb: torch.Tensor | None = None,
    shift: float = 1.0,
    on_step_start: Callable[[int, float, float], None] | None = None,
    on_step_end: Callable[[int], None] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Run Call C ODE loop; optional sequential CFG + host APG when ``guidance_scale > 1``.

    TT DiT on p150 1×1 uses two single-batch decoder forwards per step (cond, then
    null) because the decoder does not yet support HF-style 2× batched CFG.
    """
    bsz = context_latents.shape[0]
    use_cfg = guidance_config is not None and guidance_config.guidance_scale > 1.0
    if use_cfg and null_condition_emb is None:
        raise ValueError("null_condition_emb required when guidance_scale > 1")

    noise = prepare_noise(context_latents, seed)
    t = ode_timesteps(infer_steps, shift=shift)
    xt = noise
    per_step_vt: list[torch.Tensor] = []
    per_step_xt: list[torch.Tensor] = []
    momentum_buffer = MomentumBuffer(momentum=guidance_config.apg_momentum) if use_cfg else None

    for i in range(infer_steps):
        t_curr, t_prev = t[i], t[i + 1]
        if on_step_start is not None:
            on_step_start(i, float(t_curr), float(t_prev))

        t_curr_tensor = t_curr * torch.ones((bsz,), dtype=torch.float32)

        if use_cfg:
            assert guidance_config is not None
            assert null_condition_emb is not None
            null_enc = null_condition_emb.expand_as(encoder_hidden_states)
            vt_cond = decoder(
                xt,
                timestep=t_curr_tensor,
                timestep_r=t_curr_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )
            vt_uncond = decoder(
                xt,
                timestep=t_curr_tensor,
                timestep_r=t_curr_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=null_enc,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )
            vt = apply_acestep_guidance(
                latents=xt,
                pred_cond=vt_cond.to(torch.float32),
                pred_uncond=vt_uncond.to(torch.float32),
                timestep=float(t_curr),
                config=guidance_config,
                momentum_buffer=momentum_buffer,
            )
        else:
            vt = decoder(
                xt,
                timestep=t_curr_tensor,
                timestep_r=t_curr_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )
            vt = vt.to(torch.float32)

        per_step_vt.append(vt)
        dt = t_curr - t_prev
        xt = xt - vt * dt
        per_step_xt.append(xt)

        if on_step_end is not None:
            on_step_end(i)

    return xt, per_step_vt, per_step_xt, noise
