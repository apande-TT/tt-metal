# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step v1.5 flow-matching audio pipeline (tt_dit production entry)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import (
    assemble_context_latents,
    build_inputs,
    load_hf_model,
    ode_timesteps,
    prepare_noise,
    tokenize_preprocess,
)
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence


_DEFAULT_CHECKPOINT = "ACE-Step/acestep-v15-base"


@dataclass(frozen=True, kw_only=True)
class AceStepPipelineConfig:
    """Minimal config for ACE-Step host orchestration on a 1x1 mesh (p150)."""

    checkpoint_name: str = _DEFAULT_CHECKPOINT
    cfg_enabled: bool = True
    num_inference_steps: int = 4
    sample_rate: int = 48000
    pool_window_size: int = 5


class AceStepPipeline(PipelineAPIMixin):
    """Host-side ACE-Step v1.5 pipeline.

    v0 scope: text+lyric+timbre conditioning (captured inputs) → flow-matching
    DiT denoising → ``target_latents`` only (no VAE waveform yet).

    Delegates subsystem math to the graduated hf_eager TT stubs via
    ``AceStepPipelineTT``; this module adds tt_dit config/device wiring and
    profiler section events.
    """

    def __init__(
        self,
        *,
        device: ttnn.Device | ttnn.MeshDevice,
        config: AceStepPipelineConfig,
    ) -> None:
        self._device = device
        self._config = config
        self._hf_model = load_hf_model()
        self._inner = AceStepPipelineTT(device, self._hf_model)

    @classmethod
    def create_pipeline(
        cls,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        *,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        cfg_enabled: bool = True,
        num_inference_steps: int = 4,
        sample_rate: int = 48000,
        pool_window_size: int = 5,
    ) -> AceStepPipeline:
        config = AceStepPipelineConfig(
            checkpoint_name=checkpoint_name,
            cfg_enabled=cfg_enabled,
            num_inference_steps=num_inference_steps,
            sample_rate=sample_rate,
            pool_window_size=pool_window_size,
        )
        return cls(device=mesh_device, config=config)

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: AbcSequence[str],
        negative_prompts: AbcSequence[str] | None = None,
        num_inference_steps: int | None = None,
        seed: int = 0,
        traced: bool = True,
        on_event: PipelineEventCallback | None = None,
    ) -> torch.Tensor:
        del traced  # v0: hf_eager stubs do not expose tt_dit tracing yet.
        on_event = on_event if on_event is not None else null_callback

        if negative_prompts is not None and self._config.cfg_enabled:
            logger.warning("ACE-Step v0: negative_prompts ignored (CFG not wired yet)")

        if prompts:
            logger.debug("ACE-Step v0: prompts ignored; using captured inputs from hf_eager build_inputs()")

        infer_steps = num_inference_steps if num_inference_steps is not None else self._config.num_inference_steps
        inputs = build_inputs(seed=seed if seed else None)
        pool_window_size = self._config.pool_window_size

        src_latents = inputs["src_latents"]
        silence_latent = inputs["silence_latent"]
        attention_mask = inputs["attention_mask"]
        chunk_masks = inputs["chunk_masks"]
        is_covers = inputs["is_covers"]
        bsz = src_latents.shape[0]

        on_event(SectionStart("total"))

        on_event(SectionStart("encoder"))
        encoder_hidden_states, encoder_attention_mask = self._inner.condition_encoder(
            text_hidden_states=inputs["text_hidden_states"],
            text_attention_mask=inputs["text_attention_mask"],
            lyric_hidden_states=inputs["lyric_hidden_states"],
            lyric_attention_mask=inputs["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=inputs["refer_audio_order_mask"],
        )
        on_event(SectionEnd("encoder"))

        on_event(SectionStart("tokenizer"))
        x_patched, _ = tokenize_preprocess(src_latents, silence_latent, attention_mask, pool_window_size)
        quantized, _indices = self._inner.audio_tokenizer(x_patched)
        on_event(SectionEnd("tokenizer"))

        on_event(SectionStart("detokenizer"))
        lm_hints_25hz = self._inner.detokenizer(quantized)
        context_latents = assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers)
        on_event(SectionEnd("detokenizer"))

        noise = prepare_noise(context_latents, seed)
        t = ode_timesteps(infer_steps)
        xt = noise

        on_event(SectionStart("denoising"))
        for i in range(infer_steps):
            on_event(SectionStart(f"denoising_step_{i}"))
            t_curr, t_prev = t[i], t[i + 1]
            t_curr_tensor = t_curr * torch.ones((bsz,), dtype=torch.float32)
            vt = self._inner.decoder(
                hidden_states=xt,
                timestep=t_curr_tensor,
                timestep_r=t_curr_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )
            dt = t_curr - t_prev
            xt = xt - vt.to(torch.float32) * dt
            on_event(SectionEnd(f"denoising_step_{i}"))
        on_event(SectionEnd("denoising"))

        on_event(SectionEnd("total"))

        return xt
