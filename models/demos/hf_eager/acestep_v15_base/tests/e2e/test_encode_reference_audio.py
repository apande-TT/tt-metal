# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 2B CPU golden: live reference WAV encode vs HF/diffusers contract.

No device lock required — host torch only.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from models.demos.hf_eager.acestep_v15_base.tt.common import load_hf_model, pcc
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import (
    DEFAULT_SAMPLE_RATE,
    _build_cover_chunk_masks,
    _crop_reference_segments,
    _prepare_cover_src_latents,
    _prepare_reference_latents,
    encode_reference_audio,
    load_oobleck_vae,
    load_wav,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE_WAV = FIXTURE_DIR / "ref_cover_2s.wav"
SEED = 42
PCC_TARGET = 0.99
DURATION_SEC = 2.0


def _write_deterministic_wav(path: Path, duration_sec: float = DURATION_SEC, seed: int = SEED) -> None:
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


@pytest.fixture(scope="session")
def vae():
    return load_oobleck_vae()


def _hf_golden_encode(
    wav_path: Path,
    hf_model,
    vae,
    *,
    batch_size: int = 1,
    seed: int = SEED,
) -> dict[str, torch.Tensor]:
    """Replicate diffusers AceStepPipeline reference + cover src_latents logic."""
    dtype = torch.float32
    generator = torch.Generator().manual_seed(seed)
    audio = load_wav(str(wav_path))

    refer_audio, refer_order_mask = _prepare_reference_latents(
        audio,
        vae,
        batch_size=batch_size,
        dtype=dtype,
        generator=generator,
    )
    src_latents, latent_length = _prepare_cover_src_latents(
        audio,
        vae,
        hf_model,
        batch_size=batch_size,
        dtype=dtype,
        generator=generator,
    )
    chunk_masks = _build_cover_chunk_masks(batch_size, latent_length, dtype=dtype)
    attention_mask = torch.ones(batch_size, latent_length, dtype=dtype)
    silence_latent = torch.zeros(batch_size, latent_length, 64, dtype=dtype)
    is_covers = torch.ones(batch_size, dtype=torch.int64)

    return {
        "refer_audio_acoustic_hidden_states_packed": refer_audio,
        "refer_audio_order_mask": refer_order_mask,
        "src_latents": src_latents,
        "chunk_masks": chunk_masks,
        "attention_mask": attention_mask,
        "silence_latent": silence_latent,
        "is_covers": is_covers,
    }


def test_encode_reference_audio_shapes(fixture_wav, hf_model, vae):
    torch.manual_seed(SEED)
    out = encode_reference_audio(
        str(fixture_wav),
        hf_model=hf_model,
        vae=vae,
        seed=SEED,
        use_same_for_src=True,
    )

    assert out["refer_audio_acoustic_hidden_states_packed"].shape == (1, 750, 64)
    assert out["refer_audio_order_mask"].shape == (1,)
    assert out["src_latents"].shape == (1, 50, 64)
    assert out["chunk_masks"].shape == (1, 50, 64)
    assert out["attention_mask"].shape == (1, 50)
    assert out["silence_latent"].shape == (1, 50, 64)
    assert out["is_covers"].tolist() == [1]
    assert out["latent_length"] == 50
    assert abs(out["audio_duration_sec"] - DURATION_SEC) < 0.05


def test_encode_reference_audio_pcc_vs_hf_golden(fixture_wav, hf_model, vae):
    torch.manual_seed(SEED)

    actual = encode_reference_audio(
        str(fixture_wav),
        hf_model=hf_model,
        vae=vae,
        seed=SEED,
        use_same_for_src=True,
    )
    golden = _hf_golden_encode(fixture_wav, hf_model, vae, seed=SEED)

    for key in (
        "refer_audio_acoustic_hidden_states_packed",
        "src_latents",
        "chunk_masks",
        "attention_mask",
    ):
        ok, value = pcc(golden[key], actual[key])
        print(f"PCC {key}={value}", flush=True)
        assert ok, f"{key} PCC {value} < {PCC_TARGET}"
        del golden[key]

    assert torch.equal(golden["refer_audio_order_mask"], actual["refer_audio_order_mask"])
    assert torch.equal(golden["is_covers"], actual["is_covers"])

    cropped = _crop_reference_segments(load_wav(str(fixture_wav)), DEFAULT_SAMPLE_RATE)
    assert cropped.shape[-1] == 30 * DEFAULT_SAMPLE_RATE
    assert actual["refer_audio_acoustic_hidden_states_packed"].shape[1] == 750
