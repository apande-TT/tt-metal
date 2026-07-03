# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 2A/2B integration: live prompt/lyrics/reference → pipeline inputs."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import GATE_CONFIG, load_hf_model
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import DEFAULT_SAMPLE_RATE, encode_reference_audio
from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline
from models.tt_dit.pipelines.acestep.text_encode import have_text_encoder_weights

FIXTURE_DIR = Path(__file__).resolve().parents[4] / "demos" / "hf_eager" / "acestep_v15_base" / "tests" / "fixtures"
FIXTURE_WAV = FIXTURE_DIR / "ref_cover_2s.wav"
SEED = GATE_CONFIG["seed"]
LIVE_PROMPT = "unique live prompt for pipeline integration gate"
LIVE_LYRICS = "[verse]\nLive lyric for e2e gate\n[chorus]\nTest chorus line\n"


def _write_deterministic_wav(path: Path, duration_sec: float = 2.0, seed: int = 42) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(DEFAULT_SAMPLE_RATE * duration_sec)
    t = torch.linspace(0, duration_sec, n)
    g = torch.Generator().manual_seed(seed)
    freq_l = 220.0 + torch.rand(1, generator=g).item() * 80.0
    freq_r = 330.0 + torch.rand(1, generator=g).item() * 80.0
    left = (0.35 * torch.sin(2 * math.pi * freq_l * t)).unsqueeze(0)
    right = (0.35 * torch.sin(2 * math.pi * freq_r * t)).unsqueeze(0)
    stereo = torch.cat([left, right], dim=0)

    try:
        import torchaudio

        torchaudio.save(str(path), stereo, DEFAULT_SAMPLE_RATE)
        return
    except Exception:
        pass

    from scipy.io import wavfile

    pcm = (stereo.T.numpy().clip(-1.0, 1.0) * 32767.0).astype("int16")
    wavfile.write(str(path), DEFAULT_SAMPLE_RATE, pcm)


@pytest.fixture(scope="session")
def fixture_wav() -> Path:
    if not FIXTURE_WAV.is_file():
        _write_deterministic_wav(FIXTURE_WAV)
    return FIXTURE_WAV


@pytest.fixture(scope="session")
def hf_model():
    return load_hf_model()


@pytest.mark.skipif(not have_text_encoder_weights(), reason="Qwen3-Embedding-0.6B weights not on disk")
def test_prepare_inputs_live_text_replaces_captures(hf_model) -> None:
    captured = AceStepPipeline._prepare_inputs(
        prompts=None,
        lyrics=None,
        reference_audio=None,
        seed=SEED,
        hf_model=hf_model,
    )
    live = AceStepPipeline._prepare_inputs(
        prompts=[LIVE_PROMPT],
        lyrics=LIVE_LYRICS,
        reference_audio=None,
        seed=SEED,
        hf_model=hf_model,
    )

    for key in ("refer_audio_acoustic_hidden_states_packed", "src_latents", "chunk_masks"):
        assert torch.equal(captured[key], live[key]), f"{key} should still come from captures without reference"

    assert not torch.equal(captured["text_hidden_states"], live["text_hidden_states"])
    assert not torch.equal(captured["lyric_hidden_states"], live["lyric_hidden_states"])
    assert live["text_hidden_states"].shape[-1] == GATE_CONFIG["text_hidden_dim"]
    assert live["lyric_hidden_states"].shape[-1] == GATE_CONFIG["text_hidden_dim"]


def test_prepare_inputs_reference_sets_is_covers(fixture_wav, hf_model) -> None:
    captured = AceStepPipeline._prepare_inputs(
        prompts=None,
        lyrics=None,
        reference_audio=None,
        seed=SEED,
        hf_model=hf_model,
    )
    with_ref = AceStepPipeline._prepare_inputs(
        prompts=None,
        lyrics=None,
        reference_audio=str(fixture_wav),
        seed=SEED,
        hf_model=hf_model,
    )
    expected = encode_reference_audio(str(fixture_wav), hf_model=hf_model, seed=SEED)

    assert with_ref["is_covers"].tolist() == [1]
    assert not torch.equal(captured["src_latents"], with_ref["src_latents"])
    assert torch.equal(with_ref["refer_audio_order_mask"], expected["refer_audio_order_mask"])
    assert with_ref["src_latents"].shape == (
        GATE_CONFIG["batch"],
        GATE_CONFIG["seq_len_latent"],
        GATE_CONFIG["audio_acoustic_hidden_dim"],
    )


@pytest.mark.skipif(not have_text_encoder_weights(), reason="Qwen3-Embedding-0.6B weights not on disk")
def test_prepare_inputs_live_text_and_reference(fixture_wav, hf_model) -> None:
    inputs = AceStepPipeline._prepare_inputs(
        prompts=[LIVE_PROMPT],
        lyrics=LIVE_LYRICS,
        reference_audio=str(fixture_wav),
        seed=SEED,
        hf_model=hf_model,
    )

    assert inputs["is_covers"].tolist() == [1]
    assert inputs["text_hidden_states"].shape[0] == GATE_CONFIG["batch"]
    assert inputs["src_latents"].shape == (
        GATE_CONFIG["batch"],
        GATE_CONFIG["seq_len_latent"],
        GATE_CONFIG["audio_acoustic_hidden_dim"],
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pipeline_live_inputs_runs_on_device(device: ttnn.Device, fixture_wav, hf_model) -> None:
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    try:
        pipe = AceStepPipeline.create_pipeline(mesh_device=device)
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    latents = pipe(
        prompts=[LIVE_PROMPT],
        lyrics=LIVE_LYRICS,
        reference_audio=str(fixture_wav),
        num_inference_steps=2,
        seed=SEED,
        traced=False,
    )

    assert isinstance(latents, torch.Tensor)
    assert latents.shape == (
        GATE_CONFIG["batch"],
        GATE_CONFIG["seq_len_latent"],
        GATE_CONFIG["audio_acoustic_hidden_dim"],
    )
