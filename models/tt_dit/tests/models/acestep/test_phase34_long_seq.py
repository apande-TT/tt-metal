# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.4: long-sequence (T=750) TT vs HF parity gates.

Diagnostic order (per phase 3.4 plan):
  1. Single-forward DiT ``vt`` PCC @ patchified seq=375
  2. Sliding-window masks wired in ``ace_step_di_t_model`` (prerequisite for 1)
  3. Per-component PCC @ T=750 for tokenizer (Call B) and detokenizer (Call D)
  4. Per-ODE-step ``vt`` PCC + full pipeline latent PCC
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.acestep_v15_base.tt.common import (
    GATE_CONFIG,
    LONG_SEQ_DURATION_SEC,
    LONG_SEQ_LATENT_FRAMES,
    LONG_SEQ_PCC_TARGET,
    load_hf_model,
    ode_timesteps,
    pcc,
    prepare_noise,
    resolve,
    tokenize_preprocess,
)
from models.demos.hf_eager.acestep_v15_base.tt.subsystem_audio_tokenizer import AudioTokenizerTT
from models.demos.hf_eager.acestep_v15_base.tt.subsystem_condition_encoder import ConditionEncoderTT
from models.demos.hf_eager.acestep_v15_base.tt.subsystem_decoder import DecoderTT
from models.demos.hf_eager.acestep_v15_base.tt.subsystem_detokenizer import DetokenizerTT
from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline
from models.tt_dit.pipelines.acestep.text_encode import have_text_encoder_weights

FIXTURE_DIR = Path(__file__).resolve().parents[4] / "demos" / "hf_eager" / "acestep_v15_base" / "tests" / "fixtures"
FIXTURE_WAV = FIXTURE_DIR / "ref_cover_2s.wav"
SEED = GATE_CONFIG["seed"]
LONG_SEQ_INFER_STEPS = 8
LONG_SEQ_SHIFT = 3.0
LIVE_PROMPT = "unique live prompt for pipeline integration gate"
LIVE_LYRICS = "[verse]\nLive lyric for e2e gate\n[chorus]\nTest chorus line\n"
ENCODER_SEQ = 109


def _pcc_at(target: float, golden: torch.Tensor, actual: torch.Tensor) -> tuple[bool, float]:
    ok, value = comp_pcc(golden.to(torch.float32), actual.to(torch.float32), target)
    return ok, value


def _long_seq_decoder_inputs(seed: int = SEED):
    """Deterministic decoder inputs at T=750 for isolated Call C checks."""
    g = torch.Generator().manual_seed(int(seed))
    b = GATE_CONFIG["batch"]
    t = LONG_SEQ_LATENT_FRAMES
    d = GATE_CONFIG["audio_acoustic_hidden_dim"]
    hidden = torch.randn(b, t, d, generator=g, dtype=torch.float32)
    context = torch.randn(b, t, d * 2, generator=g, dtype=torch.float32)
    encoder = torch.randn(b, ENCODER_SEQ, 2048, generator=g, dtype=torch.float32)
    timestep = 0.42 * torch.ones(b, dtype=torch.float32)
    return hidden, context, encoder, timestep


@pytest.fixture(scope="module")
def hf_model():
    return load_hf_model()


@pytest.fixture(scope="module")
def fixture_wav(tmp_path_factory):
    path = tmp_path_factory.mktemp("phase34") / "ref_cover_2s.wav"
    src = FIXTURE_WAV
    if src.is_file():
        import shutil

        shutil.copy(src, path)
    else:
        pytest.skip(f"fixture wav missing: {src}")
    return path


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_phase34_dit_single_forward_pcc_long_seq(device: ttnn.Device, hf_model) -> None:
    """Step 1: one DiT forward @ T=750 — isolates Call C from ODE accumulation."""
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    hidden, context, encoder, timestep = _long_seq_decoder_inputs()
    attn = torch.ones(GATE_CONFIG["batch"], LONG_SEQ_LATENT_FRAMES, dtype=torch.float32)

    hf_dec = resolve(hf_model, "decoder")
    with torch.no_grad():
        hf_out = hf_dec(
            hidden_states=hidden.clone(),
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=attn,
            encoder_hidden_states=encoder,
            encoder_attention_mask=attn,
            context_latents=context,
            use_cache=False,
        )
    vt_hf = (hf_out[0] if isinstance(hf_out, (tuple, list)) else hf_out).float()

    tt_dec = DecoderTT(device, hf_model)
    vt_tt = tt_dec(
        hidden_states=hidden,
        timestep=timestep,
        timestep_r=timestep,
        attention_mask=attn,
        encoder_hidden_states=encoder,
        encoder_attention_mask=attn,
        context_latents=context,
    ).float()

    ok, value = _pcc_at(LONG_SEQ_PCC_TARGET, vt_hf, vt_tt)
    patchified = LONG_SEQ_LATENT_FRAMES // 2
    print(
        f"[phase3.4] dit_single_forward T={LONG_SEQ_LATENT_FRAMES} patchified={patchified} "
        f"PCC={value:.6f} target={LONG_SEQ_PCC_TARGET}",
        flush=True,
    )
    assert ok, f"DiT single-forward long-seq PCC {value:.6f} < {LONG_SEQ_PCC_TARGET}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_phase34_condition_encoder_pcc_long_seq(device: ttnn.Device, hf_model, fixture_wav) -> None:
    """Step 3c: Call A condition encoder @ 750-frame timbre reference."""
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")
    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    inputs = AceStepPipeline._prepare_inputs(
        prompts=[LIVE_PROMPT],
        lyrics=LIVE_LYRICS,
        reference_audio=str(fixture_wav),
        seed=SEED,
        hf_model=hf_model,
        audio_duration=LONG_SEQ_DURATION_SEC,
    )

    with torch.no_grad():
        enc_hf, mask_hf = hf_model.prepare_condition(
            text_hidden_states=inputs["text_hidden_states"],
            text_attention_mask=inputs["text_attention_mask"],
            lyric_hidden_states=inputs["lyric_hidden_states"],
            lyric_attention_mask=inputs["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=inputs["refer_audio_order_mask"],
            hidden_states=inputs["src_latents"],
            attention_mask=inputs["attention_mask"],
            silence_latent=inputs["silence_latent"],
            src_latents=inputs["src_latents"],
            chunk_masks=inputs["chunk_masks"],
            is_covers=inputs["is_covers"],
        )[:2]

    enc_tt, mask_tt = ConditionEncoderTT(device, hf_model)(
        text_hidden_states=inputs["text_hidden_states"],
        text_attention_mask=inputs["text_attention_mask"],
        lyric_hidden_states=inputs["lyric_hidden_states"],
        lyric_attention_mask=inputs["lyric_attention_mask"],
        refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
        refer_audio_order_mask=inputs["refer_audio_order_mask"],
    )

    ok, value = _pcc_at(LONG_SEQ_PCC_TARGET, enc_hf.float(), enc_tt.float())
    print(
        f"[phase3.4] condition_encoder refer_T={inputs['refer_audio_acoustic_hidden_states_packed'].shape[1]} "
        f"enc_seq={enc_hf.shape[1]} PCC={value:.6f}",
        flush=True,
    )
    assert ok, f"condition_encoder long-seq PCC {value:.6f} < {LONG_SEQ_PCC_TARGET}"
    assert torch.equal(mask_hf, mask_tt)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_phase34_audio_tokenizer_pcc_long_seq(device: ttnn.Device, hf_model) -> None:
    """Step 3a: Call B tokenizer @ T=750 (150 pooled tokens)."""
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    g = torch.Generator().manual_seed(SEED)
    b = GATE_CONFIG["batch"]
    t = LONG_SEQ_LATENT_FRAMES
    d = GATE_CONFIG["audio_acoustic_hidden_dim"]
    src = torch.randn(b, t, d, generator=g, dtype=torch.float32)
    silence = torch.randn(b, t, d, generator=g, dtype=torch.float32)
    attn = torch.ones(b, t, dtype=torch.float32)
    pool = GATE_CONFIG["pool_window_size"]

    x_patched, _ = tokenize_preprocess(src, silence, attn, pool)
    assert x_patched.shape[1] == t // pool

    hf_tok = resolve(hf_model, "tokenizer")
    with torch.no_grad():
        hf_out = hf_tok(x_patched)
    hf_q = (hf_out[0] if isinstance(hf_out, (tuple, list)) else hf_out).float()

    tt_tok = AudioTokenizerTT(device, hf_model)
    tt_q, _ = tt_tok(x_patched)
    tt_q = tt_q.float()

    ok, value = _pcc_at(LONG_SEQ_PCC_TARGET, hf_q, tt_q)
    print(
        f"[phase3.4] audio_tokenizer T={LONG_SEQ_LATENT_FRAMES} patches={x_patched.shape[1]} " f"PCC={value:.6f}",
        flush=True,
    )
    assert ok, f"tokenizer long-seq PCC {value:.6f} < {LONG_SEQ_PCC_TARGET}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_phase34_detokenizer_pcc_long_seq(device: ttnn.Device, hf_model) -> None:
    """Step 3b: Call D detokenizer @ T=750 (HF quantized golden)."""
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    g = torch.Generator().manual_seed(SEED)
    b = GATE_CONFIG["batch"]
    t = LONG_SEQ_LATENT_FRAMES
    d = GATE_CONFIG["audio_acoustic_hidden_dim"]
    src = torch.randn(b, t, d, generator=g, dtype=torch.float32)
    silence = torch.randn(b, t, d, generator=g, dtype=torch.float32)
    attn = torch.ones(b, t, dtype=torch.float32)
    pool = GATE_CONFIG["pool_window_size"]

    x_patched, _ = tokenize_preprocess(src, silence, attn, pool)
    hf_tok = resolve(hf_model, "tokenizer")
    with torch.no_grad():
        hf_q = hf_tok(x_patched)[0].float()

    hf_detok = resolve(hf_model, "detokenizer")
    with torch.no_grad():
        hf_hints = hf_detok(hf_q).float()

    tt_detok = DetokenizerTT(device, hf_model)
    tt_hints = tt_detok(hf_q).float()

    ok, value = _pcc_at(LONG_SEQ_PCC_TARGET, hf_hints, tt_hints)
    print(
        f"[phase3.4] detokenizer T={LONG_SEQ_LATENT_FRAMES} out={tuple(hf_hints.shape)} " f"PCC={value:.6f}",
        flush=True,
    )
    assert ok, f"detokenizer long-seq PCC {value:.6f} < {LONG_SEQ_PCC_TARGET}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_phase34_per_step_vt_pcc_long_seq(device: ttnn.Device, hf_model, fixture_wav) -> None:
    """Step 4a: per-ODE-step ``vt`` PCC @ T=750 with shared HF conditioning (decoder-only)."""
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")
    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    inputs = AceStepPipeline._prepare_inputs(
        prompts=[LIVE_PROMPT],
        lyrics=LIVE_LYRICS,
        reference_audio=str(fixture_wav),
        seed=SEED,
        hf_model=hf_model,
        audio_duration=LONG_SEQ_DURATION_SEC,
    )
    assert inputs["src_latents"].shape[1] == LONG_SEQ_LATENT_FRAMES

    with torch.no_grad():
        enc_h, enc_m, ctx = hf_model.prepare_condition(
            text_hidden_states=inputs["text_hidden_states"],
            text_attention_mask=inputs["text_attention_mask"],
            lyric_hidden_states=inputs["lyric_hidden_states"],
            lyric_attention_mask=inputs["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=inputs["refer_audio_order_mask"],
            hidden_states=inputs["src_latents"],
            attention_mask=inputs["attention_mask"],
            silence_latent=inputs["silence_latent"],
            src_latents=inputs["src_latents"],
            chunk_masks=inputs["chunk_masks"],
            is_covers=inputs["is_covers"],
        )

    attn = inputs["attention_mask"]
    noise = prepare_noise(ctx, SEED)
    t = ode_timesteps(LONG_SEQ_INFER_STEPS, shift=LONG_SEQ_SHIFT)
    xt = noise
    hf_dec = resolve(hf_model, "decoder")
    tt_dec = DecoderTT(device, hf_model)

    min_pcc = 1.0
    with torch.no_grad():
        for i in range(LONG_SEQ_INFER_STEPS):
            t_curr = t[i] * torch.ones((GATE_CONFIG["batch"],), dtype=torch.float32)
            vt_hf = hf_dec(
                hidden_states=xt,
                timestep=t_curr,
                timestep_r=t_curr,
                attention_mask=attn,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=enc_m,
                context_latents=ctx,
                use_cache=False,
            )[0].float()
            vt_tt = tt_dec(
                hidden_states=xt,
                timestep=t_curr,
                timestep_r=t_curr,
                attention_mask=attn,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=enc_m,
                context_latents=ctx,
            ).float()
            ok, value = _pcc_at(LONG_SEQ_PCC_TARGET, vt_hf, vt_tt)
            min_pcc = min(min_pcc, value)
            print(f"[phase3.4] step {i} vt PCC={value:.6f}", flush=True)
            assert ok, f"step {i} vt PCC {value:.6f} < {LONG_SEQ_PCC_TARGET}"
            xt = xt - vt_hf * (t[i] - t[i + 1])
    print(f"[phase3.4] per_step_vt min_pcc={min_pcc:.6f}", flush=True)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_phase34_pipeline_long_seq_hf_parity_latents(device: ttnn.Device, fixture_wav, hf_model) -> None:
    """Step 4b: full pipeline target_latents PCC @ T=750 (Phase 3.4 exit gate)."""
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")
    if not have_text_encoder_weights():
        pytest.skip("Qwen3-Embedding-0.6B weights not on disk")

    try:
        pipe = AceStepPipeline.create_pipeline(
            mesh_device=device,
            num_inference_steps=LONG_SEQ_INFER_STEPS,
            guidance_scale=1.0,
            cfg_enabled=False,
            audio_duration=LONG_SEQ_DURATION_SEC,
            shift=LONG_SEQ_SHIFT,
        )
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    inputs = AceStepPipeline._prepare_inputs(
        prompts=[LIVE_PROMPT],
        lyrics=LIVE_LYRICS,
        reference_audio=str(fixture_wav),
        seed=SEED,
        hf_model=hf_model,
        audio_duration=LONG_SEQ_DURATION_SEC,
    )
    assert inputs["src_latents"].shape[1] == LONG_SEQ_LATENT_FRAMES
    assert inputs["is_covers"].tolist() == [0]

    hf_out = hf_model.generate_audio(
        text_hidden_states=inputs["text_hidden_states"],
        text_attention_mask=inputs["text_attention_mask"],
        lyric_hidden_states=inputs["lyric_hidden_states"],
        lyric_attention_mask=inputs["lyric_attention_mask"],
        refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
        refer_audio_order_mask=inputs["refer_audio_order_mask"],
        src_latents=inputs["src_latents"],
        chunk_masks=inputs["chunk_masks"],
        is_covers=inputs["is_covers"],
        silence_latent=inputs["silence_latent"],
        attention_mask=inputs["attention_mask"],
        seed=SEED,
        infer_steps=LONG_SEQ_INFER_STEPS,
        diffusion_guidance_sale=1.0,
        shift=LONG_SEQ_SHIFT,
        use_progress_bar=False,
    )
    lat_hf = hf_out["target_latents"].float()

    lat_tt = pipe(
        prompts=[LIVE_PROMPT],
        lyrics=LIVE_LYRICS,
        reference_audio=str(fixture_wav),
        num_inference_steps=LONG_SEQ_INFER_STEPS,
        seed=SEED,
        traced=False,
    ).float()

    assert (
        lat_tt.shape
        == lat_hf.shape
        == (
            GATE_CONFIG["batch"],
            LONG_SEQ_LATENT_FRAMES,
            GATE_CONFIG["audio_acoustic_hidden_dim"],
        )
    )

    ok, value = pcc(lat_hf, lat_tt, LONG_SEQ_PCC_TARGET)
    print(
        f"[phase3.4] pipeline long-seq TT vs HF PCC={value:.6f} "
        f"T={LONG_SEQ_LATENT_FRAMES} steps={LONG_SEQ_INFER_STEPS} shift={LONG_SEQ_SHIFT}",
        flush=True,
    )
    assert ok, f"long-seq latent PCC {value:.6f} < {LONG_SEQ_PCC_TARGET}"
