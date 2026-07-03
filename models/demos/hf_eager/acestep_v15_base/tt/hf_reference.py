# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""HF golden reference for the ACE-Step v1.5 e2e task head.

This is a faithful re-implementation of AceStepConditionGenerationModel.generate_audio
run with the REAL HF torch submodules (model.encoder / model.tokenize /
model.detokenize / model.decoder), fixed to the e2e gate settings
(use_cache=False, guidance_scale=1.0). It returns target_latents plus every
stage intermediate so the e2e test can PCC each joint. The TT pipeline
(tt/pipeline.py) mirrors this exact chain with the graduated TTNN stubs."""
from __future__ import annotations

import torch

from .common import assemble_context_latents, ode_timesteps, prepare_noise, tokenize_preprocess


@torch.no_grad()
def hf_generate_reference(model, inputs, infer_steps=2, seed=1234, pool_window_size=5):
    src_latents = inputs["src_latents"]
    silence_latent = inputs["silence_latent"]
    attention_mask = inputs["attention_mask"]
    chunk_masks = inputs["chunk_masks"]
    is_covers = inputs["is_covers"]
    bsz = src_latents.shape[0]

    # --- prepare_condition ---
    encoder_hidden_states, encoder_attention_mask = model.encoder(
        text_hidden_states=inputs["text_hidden_states"],
        text_attention_mask=inputs["text_attention_mask"],
        lyric_hidden_states=inputs["lyric_hidden_states"],
        lyric_attention_mask=inputs["lyric_attention_mask"],
        refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
        refer_audio_order_mask=inputs["refer_audio_order_mask"],
    )

    # tokenize (host preprocess mirrors model.tokenize) then the real tokenizer module
    x_patched, _ = tokenize_preprocess(src_latents, silence_latent, attention_mask, pool_window_size)
    quantized, indices = model.tokenizer(x_patched)
    lm_hints_25hz = model.detokenize(quantized)
    context_latents = assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers)

    # --- flow-matching ODE loop (guidance_scale=1.0 -> no CFG; use_cache=False) ---
    noise = prepare_noise(context_latents, seed)
    t = ode_timesteps(infer_steps)
    xt = noise
    per_step_vt = []
    per_step_xt = []
    for i in range(infer_steps):
        t_curr, t_prev = t[i], t[i + 1]
        t_curr_tensor = t_curr * torch.ones((bsz,), dtype=xt.dtype)
        decoder_outputs = model.decoder(
            hidden_states=xt,
            timestep=t_curr_tensor,
            timestep_r=t_curr_tensor,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
        )
        vt = decoder_outputs[0]
        per_step_vt.append(vt)
        dt = t_curr - t_prev
        xt = xt - vt * dt
        # Generated denoising state after step i (x_1..x_N; x_N == target_latents).
        per_step_xt.append(xt)

    return {
        "target_latents": xt,
        "per_step_vt": per_step_vt,
        "per_step_xt": per_step_xt,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "x_patched": x_patched,
        "quantized": quantized,
        "lm_hints_25hz": lm_hints_25hz,
        "context_latents": context_latents,
        "noise": noise,
    }
