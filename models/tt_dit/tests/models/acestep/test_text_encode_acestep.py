# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""CPU golden: live ACE-Step text conditioning vs independent HF reference."""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.acestep_v15_base.tt.common import GATE_CONFIG, build_inputs
from models.tt_dit.pipelines.acestep.text_encode import (
    encode_text_conditioning,
    encode_text_conditioning_hf_reference,
    format_prompt_and_lyrics,
    have_text_encoder_weights,
)

PCC_TARGET = 0.99

FIXED_CASES = [
    pytest.param(
        "upbeat electronic dance track with energetic drums",
        "[verse]\nFeel the beat tonight\n[chorus]\nDance until the morning light\n",
        id="dance",
    ),
    pytest.param(
        "calm ambient piano soundscape, soft and meditative",
        "[verse]\nQuiet stars above\n[chorus]\nPeaceful night of love\n",
        id="ambient",
    ),
]


@pytest.mark.parametrize("prompt,lyrics", FIXED_CASES)
def test_format_prompt_and_lyrics_templates(prompt: str, lyrics: str) -> None:
    text_str, lyric_str = format_prompt_and_lyrics(prompt, lyrics, audio_duration=2.0)
    assert "# Instruction" in text_str
    assert "# Caption" in text_str
    assert prompt in text_str
    assert "- duration: 2 seconds" in text_str
    assert "# Languages" in lyric_str
    assert "# Lyric" in lyric_str
    assert lyrics in lyric_str


@pytest.mark.skipif(not have_text_encoder_weights(), reason="Qwen3-Embedding-0.6B weights not on disk")
@pytest.mark.parametrize("prompt,lyrics", FIXED_CASES)
def test_encode_text_conditioning_matches_hf_reference(prompt: str, lyrics: str) -> None:
    audio_duration = GATE_CONFIG["seq_len_latent"] / 25.0
    actual = encode_text_conditioning(
        prompts=prompt,
        lyrics=lyrics,
        batch_size=1,
        dtype=torch.float32,
        audio_duration=audio_duration,
    )
    golden = encode_text_conditioning_hf_reference(
        prompts=prompt,
        lyrics=lyrics,
        batch_size=1,
        dtype=torch.float32,
        audio_duration=audio_duration,
    )

    for key in ("text_hidden_states", "lyric_hidden_states", "text_attention_mask", "lyric_attention_mask"):
        ok, value = comp_pcc(golden[key], actual[key], PCC_TARGET)
        print(f"TEXT_ENCODE_PCC {key}: {value:.6f}", flush=True)
        assert ok, f"{key} PCC {value:.6f} < {PCC_TARGET}"


@pytest.mark.skipif(not have_text_encoder_weights(), reason="Qwen3-Embedding-0.6B weights not on disk")
def test_build_inputs_live_text_replaces_captures() -> None:
    captured = build_inputs(use_captured=True)
    live = build_inputs(
        use_captured=False,
        prompts="unique live prompt for phase 2a gate test",
        lyrics="[verse]\nLive lyric line one\n",
    )

    for key in ("refer_audio_acoustic_hidden_states_packed", "src_latents", "chunk_masks"):
        assert torch.equal(captured[key], live[key]), f"{key} should still come from captures in Phase 2A"

    assert not torch.equal(captured["text_hidden_states"], live["text_hidden_states"])
    assert not torch.equal(captured["lyric_hidden_states"], live["lyric_hidden_states"])
    assert live["text_hidden_states"].shape[-1] == GATE_CONFIG["text_hidden_dim"]
    assert live["lyric_hidden_states"].shape[-1] == GATE_CONFIG["text_hidden_dim"]
    assert live["text_hidden_states"].shape[0] == GATE_CONFIG["batch"]
    assert live["lyric_hidden_states"].shape[0] == GATE_CONFIG["batch"]
