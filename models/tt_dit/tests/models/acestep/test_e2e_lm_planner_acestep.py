# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 7C device gate: TT LM planner full e2e + quality metrics vs tokenizer path."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import save_wav
from models.tt_dit.pipelines.acestep.lm_planner import LM_VARIANTS, have_lm_planner_weights, resolve_lm_planner_path
from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline
from models.tt_dit.pipelines.acestep.text_encode import have_text_encoder_weights

PHASE7C_REF = os.environ.get("ACESTEP_PHASE7_REF", "/tmp/ref_kaazoom_25s.wav")
PHASE7C_PROMPT = os.environ.get(
    "ACESTEP_PHASE7_PROMPT",
    "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm",
)
PHASE7C_LYRICS = os.environ.get(
    "ACESTEP_PHASE7_LYRICS",
    "[verse]\nCity lights are fading slow\nWarm piano starts to glow\n"
    "Soft drums keep the time so low\nIn this lounge where feelings flow\n"
    "[chorus]\nStay with me tonight\nUnder neon light\n"
    "Smooth jazz in the air\nLike we haven't got a care\n",
)
PHASE7C_OUT_DIR = Path(os.environ.get("ACESTEP_PHASE7_OUT_DIR", "/tmp"))
FAST_AUDIO_DURATION = float(os.environ.get("ACESTEP_PHASE7_FAST_DURATION", "8"))
FAST_INFER_STEPS = int(os.environ.get("ACESTEP_PHASE7_FAST_INFER_STEPS", "4"))
PROD_AUDIO_DURATION = float(os.environ.get("ACESTEP_PHASE7_PROD_DURATION", "30"))
PROD_INFER_STEPS = int(os.environ.get("ACESTEP_PHASE7_PROD_INFER_STEPS", "30"))
SEED = 42
DEFAULT_LM = "1.7B"


def _run_phase7c_enabled() -> bool:
    return os.environ.get("ACESTEP_RUN_PHASE7C", "0") in ("1", "true", "yes")


def _run_phase7c_prod_enabled() -> bool:
    return os.environ.get("ACESTEP_RUN_PHASE7C_PROD", "0") in ("1", "true", "yes")


def _run_phase7c_variants_enabled() -> bool:
    return os.environ.get("ACESTEP_RUN_PHASE7C_VARIANTS", "0") in ("1", "true", "yes")


def _require_reference_wav() -> str:
    if os.path.isfile(PHASE7C_REF):
        return PHASE7C_REF
    pytest.skip(f"Reference WAV not found: {PHASE7C_REF}")


def _create_pipeline(
    device: ttnn.Device,
    *,
    use_lm_planner: bool,
    use_tt_lm_planner: bool,
    lm_model: str,
    audio_duration: float,
    infer_steps: int,
) -> AceStepPipeline:
    return AceStepPipeline.create_pipeline(
        mesh_device=device,
        num_inference_steps=infer_steps,
        guidance_scale=7.0,
        audio_duration=audio_duration,
        shift=3.0,
        use_tt_text_encode=True,
        use_lm_planner=use_lm_planner,
        use_tt_lm_planner=use_tt_lm_planner,
        lm_model=lm_model,
    )


def _run_e2e(
    pipe: AceStepPipeline,
    *,
    reference: str,
    infer_steps: int,
    return_waveform: bool,
) -> dict | torch.Tensor:
    return pipe(
        prompts=[PHASE7C_PROMPT],
        lyrics=PHASE7C_LYRICS,
        reference_audio=reference,
        num_inference_steps=infer_steps,
        seed=SEED,
        traced=False,
        return_waveform=return_waveform,
        use_tt_vae=True,
    )


@pytest.mark.parametrize("lm_model", LM_VARIANTS.keys())
def test_phase7c_lm_variant_weights_resolve(lm_model: str) -> None:
    if not have_lm_planner_weights(model=lm_model):
        pytest.skip(f"acestep-5Hz-lm-{lm_model} weights not on disk")
    path = resolve_lm_planner_path(model=lm_model)
    assert path.endswith(LM_VARIANTS[lm_model])


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "lm_model",
    [DEFAULT_LM] if not _run_phase7c_variants_enabled() else list(LM_VARIANTS.keys()),
)
def test_phase7c_tt_lm_e2e_waveform(device: ttnn.Device, lm_model: str) -> None:
    """TT LM + TT text + TT DiT + TT VAE → finite waveform (fast gate)."""
    if not _run_phase7c_enabled():
        pytest.skip("Set ACESTEP_RUN_PHASE7C=1 to run Phase 7C device e2e (slow)")

    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    if not have_lm_planner_weights(model=lm_model):
        pytest.skip(f"acestep-5Hz-lm-{lm_model} weights not on disk")

    reference = _require_reference_wav()

    try:
        pipe = _create_pipeline(
            device,
            use_lm_planner=True,
            use_tt_lm_planner=True,
            lm_model=lm_model,
            audio_duration=FAST_AUDIO_DURATION,
            infer_steps=FAST_INFER_STEPS,
        )
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    t0 = time.perf_counter()
    result = _run_e2e(pipe, reference=reference, infer_steps=FAST_INFER_STEPS, return_waveform=True)
    e2e_s = time.perf_counter() - t0

    assert isinstance(result, dict)
    waveform = result["waveform"]
    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 3
    assert waveform.shape[1] == 2
    assert waveform.shape[2] > 0
    assert torch.isfinite(waveform).all()
    assert waveform.abs().mean() > 1e-5

    out_path = PHASE7C_OUT_DIR / f"az_phase7c_tt_lm_{lm_model.lower()}_{int(FAST_AUDIO_DURATION)}s.wav"
    save_wav(str(out_path), waveform)
    print(
        f"PHASE7C_E2E lm={lm_model} tt_lm=1 duration={FAST_AUDIO_DURATION}s "
        f"steps={FAST_INFER_STEPS} e2e_s={e2e_s:.2f} wav={out_path}",
        flush=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_phase7c_tokenizer_vs_tt_lm_latent_pcc(device: ttnn.Device) -> None:
    """Run tokenizer-only vs TT LM on same prompt/ref; report latent PCC (paths differ by design)."""
    if not _run_phase7c_enabled():
        pytest.skip("Set ACESTEP_RUN_PHASE7C=1 to run Phase 7C quality gate (slow)")

    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    if not have_lm_planner_weights(model=DEFAULT_LM):
        pytest.skip("acestep-5Hz-lm-1.7B weights not on disk")

    reference = _require_reference_wav()

    try:
        pipe_tokenizer = _create_pipeline(
            device,
            use_lm_planner=False,
            use_tt_lm_planner=False,
            lm_model=DEFAULT_LM,
            audio_duration=FAST_AUDIO_DURATION,
            infer_steps=FAST_INFER_STEPS,
        )
        pipe_tt_lm = _create_pipeline(
            device,
            use_lm_planner=True,
            use_tt_lm_planner=True,
            lm_model=DEFAULT_LM,
            audio_duration=FAST_AUDIO_DURATION,
            infer_steps=FAST_INFER_STEPS,
        )
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    latents_tokenizer = _run_e2e(
        pipe_tokenizer, reference=reference, infer_steps=FAST_INFER_STEPS, return_waveform=False
    )
    latents_lm = _run_e2e(pipe_tt_lm, reference=reference, infer_steps=FAST_INFER_STEPS, return_waveform=False)

    assert isinstance(latents_tokenizer, torch.Tensor)
    assert isinstance(latents_lm, torch.Tensor)
    assert latents_tokenizer.shape == latents_lm.shape
    assert torch.isfinite(latents_tokenizer).all()
    assert torch.isfinite(latents_lm).all()

    _, pcc = comp_pcc(latents_tokenizer.float().flatten(), latents_lm.float().flatten(), pcc=0.0)
    print(
        f"PHASE7C_QUALITY tokenizer_vs_tt_lm latent_pcc={pcc:.6f} "
        f"duration={FAST_AUDIO_DURATION}s steps={FAST_INFER_STEPS}",
        flush=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_phase7c_tt_lm_production_signoff(device: ttnn.Device) -> None:
    """30 s / 30-step production stack with TT LM 1.7B (manual listen gate)."""
    if not _run_phase7c_prod_enabled():
        pytest.skip("Set ACESTEP_RUN_PHASE7C_PROD=1 to run Phase 7C production signoff (very slow)")

    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    if not have_lm_planner_weights(model=DEFAULT_LM):
        pytest.skip("acestep-5Hz-lm-1.7B weights not on disk")

    reference = _require_reference_wav()
    out_path = os.environ.get("ACESTEP_PHASE7_PROD_OUT", "/tmp/az_phase7c_signoff.wav")

    try:
        pipe = _create_pipeline(
            device,
            use_lm_planner=True,
            use_tt_lm_planner=True,
            lm_model=DEFAULT_LM,
            audio_duration=PROD_AUDIO_DURATION,
            infer_steps=PROD_INFER_STEPS,
        )
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    t0 = time.perf_counter()
    result = _run_e2e(pipe, reference=reference, infer_steps=PROD_INFER_STEPS, return_waveform=True)
    e2e_s = time.perf_counter() - t0

    assert isinstance(result, dict)
    waveform = result["waveform"]
    assert torch.isfinite(waveform).all()
    save_wav(out_path, waveform)
    print(
        f"PHASE7C_PROD_SIGNOFF lm=1.7B duration={PROD_AUDIO_DURATION}s "
        f"steps={PROD_INFER_STEPS} e2e_s={e2e_s:.2f} wav={out_path}",
        flush=True,
    )
