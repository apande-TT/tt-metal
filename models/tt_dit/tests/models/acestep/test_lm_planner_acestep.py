# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 7A CPU golden: LM planner audio_codes → lm_hints_25Hz."""

from __future__ import annotations

import pytest
import torch

from models.demos.hf_eager.acestep_v15_base.tt.common import GATE_CONFIG, load_hf_model
from models.tt_dit.pipelines.acestep.lm_planner import (
    audio_codes_to_lm_hints_25hz,
    audio_codes_to_lm_quantized,
    have_lm_planner_weights,
    parse_audio_code_string,
    parse_lm_output,
    resolve_lm_planner_path,
)

FIXED_CODE_STRING = "".join(f"<|audio_code_{100 + i * 10}|>" for i in range(10))
FIXED_COT = (
    "<think>\n"
    "bpm: 90\n"
    "duration: 10\n"
    "caption: smooth jazz lounge\n"
    "language: en\n"
    "</think>\n" + FIXED_CODE_STRING
)


def test_parse_audio_code_string_extracts_indices() -> None:
    codes = parse_audio_code_string(FIXED_CODE_STRING)
    assert codes == [100 + i * 10 for i in range(10)]


def test_parse_lm_output_reads_metadata_and_codes() -> None:
    metadata, codes = parse_lm_output(FIXED_COT)
    assert metadata.get("bpm") == 90
    assert metadata.get("duration") == 10
    assert "smooth jazz lounge" in str(metadata.get("caption", ""))
    assert parse_audio_code_string(codes) == [100 + i * 10 for i in range(10)]


def test_resolve_lm_planner_path_finds_gtobar_default() -> None:
    if not have_lm_planner_weights(model="1.7B"):
        pytest.skip("acestep-5Hz-lm-1.7B weights not on disk")
    path = resolve_lm_planner_path(model="1.7B")
    assert path.endswith("acestep-5Hz-lm-1.7B")


@pytest.mark.skipif(not have_lm_planner_weights(model="1.7B"), reason="LM planner weights missing")
def test_audio_codes_to_lm_hints_shape_matches_src_latents() -> None:
    hf_model = load_hf_model()
    latent_length = GATE_CONFIG["seq_len_latent"]
    hints = audio_codes_to_lm_hints_25hz(
        hf_model,
        FIXED_CODE_STRING,
        target_latent_length=latent_length,
        dtype=torch.float32,
    )
    assert hints.shape == (1, latent_length, GATE_CONFIG["audio_acoustic_hidden_dim"])


@pytest.mark.skipif(not have_lm_planner_weights(model="1.7B"), reason="LM planner weights missing")
def test_audio_codes_quantized_matches_hf_prepare_condition_path() -> None:
    hf_model = load_hf_model()
    quantized = audio_codes_to_lm_quantized(hf_model, FIXED_CODE_STRING, dtype=torch.float32)
    indices = torch.tensor([[100 + i * 10] for i in range(10)], dtype=torch.long).unsqueeze(0)
    expected = hf_model.tokenizer.quantizer.get_output_from_indices(indices).float()
    ok = torch.allclose(quantized, expected, rtol=0, atol=0)
    assert ok


@pytest.mark.skipif(not have_lm_planner_weights(model="1.7B"), reason="LM planner weights missing")
@pytest.mark.slow
def test_generate_audio_codes_host_smoke() -> None:
    from models.tt_dit.pipelines.acestep.lm_planner import generate_audio_codes_host

    code_str, metadata = generate_audio_codes_host(
        caption="calm ambient piano, soft pads",
        lyrics="[verse]\nQuiet night\n",
        audio_duration=2.0,
        model="1.7B",
        seed=42,
        temperature=0.7,
    )
    codes = parse_audio_code_string(code_str)
    assert len(codes) == 10  # 2s * 5 Hz
    assert metadata or code_str
