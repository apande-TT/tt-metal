# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 5 gate: full TT A→Z demo at production settings."""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import save_wav
from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline
from models.tt_dit.pipelines.acestep.text_encode import have_text_encoder_weights

PHASE5_OUT = os.environ.get("ACESTEP_PHASE5_OUT", "/tmp/az_phase5_signoff_pytest.wav")
PHASE5_REF = os.environ.get("ACESTEP_PHASE5_REF", "/tmp/ref_kaazoom_25s.wav")
PHASE5_PROMPT = os.environ.get(
    "ACESTEP_PHASE5_PROMPT",
    "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm",
)
PHASE5_LYRICS = os.environ.get(
    "ACESTEP_PHASE5_LYRICS",
    "[verse]\nCity lights are fading slow\n[chorus]\nStay with me tonight\n",
)
INFER_STEPS = int(os.environ.get("ACESTEP_PHASE5_INFER_STEPS", "30"))
SEED = 42


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_phase5_full_tt_az_production_demo(device: ttnn.Device) -> None:
    """TT text + TT DiT + TT VAE @ 30 steps + CFG → listenable WAV file."""
    if os.environ.get("ACESTEP_RUN_PHASE5", "0") not in ("1", "true", "yes"):
        pytest.skip("Set ACESTEP_RUN_PHASE5=1 to run Phase 5 signoff (slow)")

    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    if not os.path.isfile(PHASE5_REF):
        pytest.skip(f"Reference WAV not found: {PHASE5_REF}")

    try:
        pipe = AceStepPipeline.create_pipeline(
            mesh_device=device,
            num_inference_steps=INFER_STEPS,
            guidance_scale=7.0,
            audio_duration=30.0,
            shift=3.0,
            use_tt_text_encode=True,
        )
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    t0 = time.perf_counter()
    result = pipe(
        prompts=[PHASE5_PROMPT],
        lyrics=PHASE5_LYRICS,
        reference_audio=PHASE5_REF,
        num_inference_steps=INFER_STEPS,
        seed=SEED,
        traced=False,
        return_waveform=True,
        use_tt_vae=True,
    )
    e2e_s = time.perf_counter() - t0

    assert isinstance(result, dict)
    waveform = result["waveform"]
    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 3
    assert waveform.shape[1] == 2
    assert waveform.shape[2] > 0
    assert torch.isfinite(waveform).all()
    assert result["vae_decode_s"] > 0.0

    save_wav(PHASE5_OUT, waveform)
    print(
        f"PHASE5_SIGNOFF tt_text=1 tt_vae=1 cfg=7.0 steps={INFER_STEPS} " f"e2e_s={e2e_s:.2f} wav={PHASE5_OUT}",
        flush=True,
    )
