# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 8 device gate: traced DiT + TT VAE with TT LM planner (full stack)."""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.traced_decoder import _device_supports_2cq
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import save_wav
from models.tt_dit.pipelines.acestep.lm_planner import have_lm_planner_weights
from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline
from models.tt_dit.pipelines.acestep.text_encode import have_text_encoder_weights

PHASE8_REF = os.environ.get("ACESTEP_PHASE8_REF", "/tmp/ref_kaazoom_25s.wav")
PHASE8_PROMPT = os.environ.get(
    "ACESTEP_PHASE8_PROMPT",
    "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm",
)
PHASE8_LYRICS = os.environ.get(
    "ACESTEP_PHASE8_LYRICS",
    "[verse]\nCity lights are fading slow\n[chorus]\nStay with me tonight\n",
)
PHASE8_OUT = os.environ.get("ACESTEP_PHASE8_OUT", "/tmp/az_phase8_traced_lm.wav")
INFER_STEPS = int(os.environ.get("ACESTEP_PHASE8_INFER_STEPS", "4"))
AUDIO_DURATION = float(os.environ.get("ACESTEP_PHASE8_AUDIO_DURATION", "8"))
SEED = 42

TRACE_DEVICE_PARAMS = {
    "l1_small_size": 32768,
    "num_command_queues": 2,
    "trace_region_size": 50_000_000,
}


def _run_phase8_enabled() -> bool:
    return os.environ.get("ACESTEP_RUN_PHASE8", "0") in ("1", "true", "yes")


@pytest.mark.parametrize("device_params", [TRACE_DEVICE_PARAMS], indirect=True)
def test_phase8_traced_full_stack_with_tt_lm(device: ttnn.Device) -> None:
    """Traced DiT + TT LM planner + TT text + TT VAE (CFG off when traced)."""
    if not _run_phase8_enabled():
        pytest.skip("Set ACESTEP_RUN_PHASE8=1 to run Phase 8 traced full-stack gate (slow)")

    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    if not have_lm_planner_weights(model="1.7B"):
        pytest.skip("acestep-5Hz-lm-1.7B weights not on disk")

    if not os.path.isfile(PHASE8_REF):
        pytest.skip(f"Reference WAV not found: {PHASE8_REF}")

    use_2cq = _device_supports_2cq(device)
    print(f"PHASE8 traced=1 use_2cq={use_2cq} steps={INFER_STEPS} duration={AUDIO_DURATION}s", flush=True)

    try:
        pipe = AceStepPipeline.create_pipeline(
            mesh_device=device,
            num_inference_steps=INFER_STEPS,
            guidance_scale=1.0,
            cfg_enabled=False,
            audio_duration=AUDIO_DURATION,
            shift=3.0,
            use_tt_text_encode=True,
            use_lm_planner=True,
            use_tt_lm_planner=True,
            lm_model="1.7B",
        )
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    t0 = time.perf_counter()
    result = pipe(
        prompts=[PHASE8_PROMPT],
        lyrics=PHASE8_LYRICS,
        reference_audio=PHASE8_REF,
        num_inference_steps=INFER_STEPS,
        seed=SEED,
        traced=True,
        return_waveform=True,
        use_tt_vae=True,
    )
    e2e_s = time.perf_counter() - t0

    assert isinstance(result, dict)
    waveform = result["waveform"]
    assert torch.isfinite(waveform).all()
    assert waveform.shape[2] > 0
    save_wav(PHASE8_OUT, waveform)

    vae_decode_s = result.get("vae_decode_s", float("nan"))
    latent_gen_s = e2e_s - vae_decode_s if vae_decode_s == vae_decode_s else float("nan")
    print(
        f"PHASE8_SIGNOFF traced=1 tt_lm=1 latent_gen_s={latent_gen_s:.2f} "
        f"vae_decode_s={vae_decode_s:.2f} e2e_s={e2e_s:.2f} wav={PHASE8_OUT}",
        flush=True,
    )
