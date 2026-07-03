# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-stage 2-CQ overlap for ACE-Step traced prefill stages (encoder → audio path).

While the condition-encoder trace runs on CQ_OPS (0), the audio-path input H2D
prefetch runs on CQ_IO (1), following the ViT / tt_cnn MultiCQTracedModelPipelinedIO
event pattern.  Encoder output stays device-resident and is bound into the decoder
trace constant slot via D2D copy (no encoder D2H → decoder H2D round trip).
"""
from __future__ import annotations

import torch

import ttnn

from .common import from_torch
from .traced_audio_path import TracedAudioPath2CQ
from .traced_condition_encoder import TracedConditionEncoder2CQ
from .traced_decoder import TracedDecoder2CQ


def run_encoder_audio_overlap(
    encoder: TracedConditionEncoder2CQ,
    audio: TracedAudioPath2CQ,
    *,
    text_hidden_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    lyric_hidden_states: torch.Tensor,
    lyric_attention_mask: torch.Tensor,
    refer_audio_acoustic_hidden_states_packed: torch.Tensor,
    refer_audio_order_mask: torch.Tensor,
    x_patched: torch.Tensor,
) -> tuple[ttnn.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run encoder then audio path with cross-stage CQ overlap and device-resident encoder out.

    Returns:
        (encoder_hidden_states_tt, encoder_attention_mask, lm_hints_25hz, quantized)
        ``encoder_hidden_states_tt`` is a device-resident tensor suitable for
        ``TracedDecoder2CQ.bind_encoder_hidden_states``.
    """
    encoder_attention_mask = encoder.run_prefill_and_trace(
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        lyric_hidden_states=lyric_hidden_states,
        lyric_attention_mask=lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask=refer_audio_order_mask,
    )

    # Overlap: stream audio-path input on CQ_IO while encoder trace completes on CQ_OPS.
    if audio.use_2cq:
        audio.prefetch_input(x_patched)

    encoder_hidden_states_tt = encoder.finish_trace_to_device()

    lm_hints_25hz, quantized = audio.run_trace_and_read()
    return encoder_hidden_states_tt, encoder_attention_mask, lm_hints_25hz, quantized


def bind_decoder_constants(
    decoder: TracedDecoder2CQ,
    *,
    encoder_hidden_states_tt: ttnn.Tensor,
    context_latents: torch.Tensor,
) -> None:
    """Bind per-generate constants into the decoder trace without an encoder D2H hop."""
    decoder.bind_encoder_hidden_states(encoder_hidden_states_tt)
    decoder.bind_context_latents(from_torch(context_latents, decoder.device))
