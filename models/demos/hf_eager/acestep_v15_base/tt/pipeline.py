# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The ONE shared ACE-Step v1.5 end-to-end TTNN pipeline.

Both demo/demo_generate_audio.py and tests/e2e/test_e2e_generate_audio.py import
and call this module, so a passing test guarantees a working demo (identical
wiring). It chains the four graduated subsystems exactly as
AceStepConditionGenerationModel.generate_audio does (guidance_scale=1.0,
use_cache=False, is_covers per input):

    Call A  ConditionEncoderTT  -> encoder_hidden_states, encoder_attention_mask
    Call B  AudioTokenizerTT    -> quantized  (fed the real src_latents patches)
    Call D  DetokenizerTT       -> lm_hints_25hz (fed Call B's REAL output)
    context_latents = assemble(lm_hints, src_latents, chunk_masks, is_covers)
    Call C  DecoderTT (x infer_steps ODE loop) -> target_latents

Every joint is fed the previous TT stage's REAL output; no golden/captured
tensor is ever injected mid-chain.
"""
from __future__ import annotations

import torch

from .common import assemble_context_latents, ode_timesteps, prepare_noise, resolve, to_torch, tokenize_preprocess
from .subsystem_audio_tokenizer import AudioTokenizerTT
from .subsystem_condition_encoder import ConditionEncoderTT
from .subsystem_decoder import DecoderTT
from .subsystem_detokenizer import DetokenizerTT
from .traced_audio_path import TracedAudioPath2CQ
from .traced_condition_encoder import TracedConditionEncoder2CQ
from .traced_decoder import TracedDecoder2CQ
from .traced_pipeline import bind_decoder_constants, run_encoder_audio_overlap


class AceStepPipelineTT:
    """Builds all four subsystem builders once, then runs generate() repeatedly."""

    def __init__(self, device, hf_model):
        self.device = device
        self.hf_model = hf_model
        self.condition_encoder = ConditionEncoderTT(device, hf_model)  # Call A (4 stubs)
        self.audio_tokenizer = AudioTokenizerTT(device, hf_model)  # Call B (4 stubs)
        self.detokenizer = DetokenizerTT(device, hf_model)  # Call D (1 stub)
        self.decoder = DecoderTT(device, hf_model)  # Call C (4 stubs)
        self._traced_condition_encoder: TracedConditionEncoder2CQ | None = None
        self._traced_audio_path: TracedAudioPath2CQ | None = None
        self._traced_decoder: TracedDecoder2CQ | None = None

    @torch.no_grad()
    def generate(
        self,
        inputs,
        infer_steps=2,
        seed=1234,
        pool_window_size=5,
        *,
        shift: float = 1.0,
        traced: bool = False,
        use_2cq: bool | None = None,
        on_event=None,
        guidance_config=None,
    ):
        src_latents = inputs["src_latents"]
        silence_latent = inputs["silence_latent"]
        attention_mask = inputs["attention_mask"]
        chunk_masks = inputs["chunk_masks"]
        is_covers = inputs["is_covers"]
        lm_quantized = inputs.get("lm_quantized")
        bsz = src_latents.shape[0]

        def _evt(name, start: bool) -> None:
            if on_event is None:
                return
            from models.tt_dit.pipelines.events import SectionEnd, SectionStart

            on_event(SectionStart(name) if start else SectionEnd(name))

        _evt("total", True)

        traced_condition_encoder = None
        traced_audio_path = None
        use_2cq_resolved = use_2cq if use_2cq is not None else traced

        x_patched, _ = tokenize_preprocess(src_latents, silence_latent, attention_mask, pool_window_size)

        if traced and lm_quantized is not None:
            raise NotImplementedError("LM planner with traced=True is Phase 8; use traced=False")

        use_lm_quantized = lm_quantized is not None

        if traced:
            if self._traced_condition_encoder is None:
                self._traced_condition_encoder = TracedConditionEncoder2CQ(
                    self.condition_encoder,
                    use_2cq=use_2cq,
                )
            traced_condition_encoder = self._traced_condition_encoder
            if not traced_condition_encoder.is_captured:
                traced_condition_encoder.capture(
                    text_hidden_states=inputs["text_hidden_states"],
                    text_attention_mask=inputs["text_attention_mask"],
                    lyric_hidden_states=inputs["lyric_hidden_states"],
                    lyric_attention_mask=inputs["lyric_attention_mask"],
                    refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
                    refer_audio_order_mask=inputs["refer_audio_order_mask"],
                )

            if self._traced_audio_path is None:
                self._traced_audio_path = TracedAudioPath2CQ(
                    self.audio_tokenizer,
                    self.detokenizer,
                    use_2cq=use_2cq,
                )
            traced_audio_path = self._traced_audio_path
            if not traced_audio_path.is_captured:
                traced_audio_path.capture(x_patched=x_patched)

        overlap_prefill = (
            traced
            and use_2cq_resolved
            and traced_condition_encoder is not None
            and traced_audio_path is not None
            and traced_condition_encoder.use_2cq
            and traced_audio_path.use_2cq
        )

        encoder_hidden_states_tt = None
        if overlap_prefill:
            _evt("encoder", True)
            (
                encoder_hidden_states_tt,
                encoder_attention_mask,
                lm_hints_25hz,
                quantized,
            ) = run_encoder_audio_overlap(
                traced_condition_encoder,
                traced_audio_path,
                text_hidden_states=inputs["text_hidden_states"],
                text_attention_mask=inputs["text_attention_mask"],
                lyric_hidden_states=inputs["lyric_hidden_states"],
                lyric_attention_mask=inputs["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=inputs["refer_audio_order_mask"],
                x_patched=x_patched,
            )
            _evt("encoder", False)
            _evt("tokenizer", True)
            _evt("tokenizer", False)
            _evt("detokenizer", True)
            context_latents = assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers)
            _evt("detokenizer", False)
            encoder_hidden_states = to_torch(encoder_hidden_states_tt, self.device).to(torch.float32)
        elif traced_condition_encoder is not None:
            _evt("encoder", True)
            encoder_hidden_states, encoder_attention_mask = traced_condition_encoder(
                text_hidden_states=inputs["text_hidden_states"],
                text_attention_mask=inputs["text_attention_mask"],
                lyric_hidden_states=inputs["lyric_hidden_states"],
                lyric_attention_mask=inputs["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=inputs["refer_audio_order_mask"],
            )
            _evt("encoder", False)

            _evt("tokenizer", True)
            lm_hints_25hz, quantized = traced_audio_path(x_patched)
            _evt("tokenizer", False)

            _evt("detokenizer", True)
            context_latents = assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers)
            _evt("detokenizer", False)
        else:
            _evt("encoder", True)
            encoder_hidden_states, encoder_attention_mask = self.condition_encoder(
                text_hidden_states=inputs["text_hidden_states"],
                text_attention_mask=inputs["text_attention_mask"],
                lyric_hidden_states=inputs["lyric_hidden_states"],
                lyric_attention_mask=inputs["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=inputs["refer_audio_order_mask"],
            )
            _evt("encoder", False)

            if use_lm_quantized:
                _evt("tokenizer", True)
                _evt("tokenizer", False)
                _evt("detokenizer", True)
                quantized = lm_quantized
                lm_hints_25hz = self.detokenizer(quantized)
                context_latents = assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers)
                _evt("detokenizer", False)
            else:
                _evt("tokenizer", True)
                quantized, _indices = self.audio_tokenizer(x_patched)
                _evt("tokenizer", False)

                _evt("detokenizer", True)
                lm_hints_25hz = self.detokenizer(quantized)
                context_latents = assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers)
                _evt("detokenizer", False)

        # --- Call C: flow-matching ODE loop over the DiT decoder ---
        use_cfg = guidance_config is not None and guidance_config.guidance_scale > 1.0
        if traced and use_cfg:
            raise NotImplementedError("CFG/APG with traced=True is Phase 6; use traced=False for Phase 3.")

        null_condition_emb = None
        if use_cfg:
            null_condition_emb = resolve(self.hf_model, "null_condition_emb").detach().float()

        if traced:
            noise = prepare_noise(context_latents, seed)
            t = ode_timesteps(infer_steps, shift=shift)
            xt = noise
            per_step_vt = []
            per_step_xt = []

            traced_decoder = None
            if self._traced_decoder is None:
                self._traced_decoder = TracedDecoder2CQ(self.decoder, use_2cq=use_2cq)
            traced_decoder = self._traced_decoder
            if not traced_decoder.is_captured:
                t0 = t[0] * torch.ones((bsz,), dtype=torch.float32)
                capture_kwargs = {
                    "context_latents": context_latents,
                    "sample_hidden_states": xt,
                    "sample_timestep": t0,
                }
                if encoder_hidden_states_tt is not None:
                    capture_kwargs["encoder_hidden_states_tt"] = encoder_hidden_states_tt
                else:
                    capture_kwargs["encoder_hidden_states"] = encoder_hidden_states
                traced_decoder.capture(**capture_kwargs)
            elif encoder_hidden_states_tt is not None:
                bind_decoder_constants(
                    traced_decoder,
                    encoder_hidden_states_tt=encoder_hidden_states_tt,
                    context_latents=context_latents,
                )

            _evt("denoising", True)
            for i in range(infer_steps):
                t_curr, t_prev = t[i], t[i + 1]
                t_curr_tensor = t_curr * torch.ones((bsz,), dtype=torch.float32)
                _evt(f"denoising_step_{i}", True)
                vt = traced_decoder(
                    xt,
                    timestep=t_curr_tensor,
                    timestep_r=t_curr_tensor,
                )
                per_step_vt.append(vt)
                dt = t_curr - t_prev
                xt = xt - vt * dt
                _evt(f"denoising_step_{i}", False)
                per_step_xt.append(xt)
            _evt("denoising", False)
        else:
            from models.tt_dit.pipelines.acestep.denoise import run_flow_matching_ode

            def _decoder_forward(
                hidden_states,
                *,
                timestep,
                timestep_r,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                context_latents,
            ):
                out = self.decoder(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    timestep_r=timestep_r,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                )
                return out

            def _on_step_start(i, _t_curr, _t_prev):
                _evt(f"denoising_step_{i}", True)

            def _on_step_end(i):
                _evt(f"denoising_step_{i}", False)

            _evt("denoising", True)
            xt, per_step_vt, per_step_xt, noise = run_flow_matching_ode(
                decoder=_decoder_forward,
                context_latents=context_latents,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                infer_steps=infer_steps,
                seed=seed,
                guidance_config=guidance_config,
                null_condition_emb=null_condition_emb,
                shift=shift,
                on_step_start=_on_step_start,
                on_step_end=_on_step_end,
            )
            _evt("denoising", False)

        _evt("total", False)

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


def build_pipeline(device, hf_model):
    return AceStepPipelineTT(device, hf_model)
