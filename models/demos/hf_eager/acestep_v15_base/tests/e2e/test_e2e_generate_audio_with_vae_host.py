# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase A gate: TT (or HF golden) latents -> host Oobleck VAE -> listenable WAV.

Subtests:
  test_vae_host_decode_golden_latents — no device; HF golden latents only.
  test_e2e_tt_latents_host_vae — device; TT latents + host VAE + timing prints.

The on-device test MUST run under device lock:
  flock /tmp/tt_ace_device.lock ./python_env/bin/python -m pytest ... -k tt_latents_host_vae -s -v
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model, pcc
from models.demos.hf_eager.acestep_v15_base.tt.hf_reference import hf_generate_reference
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import latents_to_waveform, save_wav

INFER_STEPS = int(os.environ.get("ACESTEP_E2E_INFER_STEPS", "4"))
SEED = 1234
PCC_TARGET = 0.99
GOLDEN_WAV = "/tmp/acestep_phase_a_golden.wav"
TT_WAV = "/tmp/acestep_phase_a_tt.wav"


def test_vae_host_decode_golden_latents():
    torch.manual_seed(SEED)

    hf_model = load_hf_model()
    inputs = build_inputs(seed=SEED)

    with torch.no_grad():
        golden = hf_generate_reference(hf_model, inputs, infer_steps=INFER_STEPS, seed=SEED)

    latents = golden["target_latents"]
    waveform = latents_to_waveform(latents)
    save_wav(GOLDEN_WAV, waveform)

    assert waveform.ndim == 3
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == 2
    assert waveform.shape[2] > 0
    assert torch.isfinite(waveform).all()
    assert os.path.isfile(GOLDEN_WAV)
    print(f"PHASE_A output_wav={GOLDEN_WAV}", flush=True)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_e2e_tt_latents_host_vae(device_params, device):
    torch.manual_seed(SEED)

    hf_model = load_hf_model()
    inputs = build_inputs(seed=SEED)

    with torch.no_grad():
        golden = hf_generate_reference(hf_model, inputs, infer_steps=INFER_STEPS, seed=SEED)

    pipe = AceStepPipelineTT(device, hf_model)

    t0 = time.perf_counter()
    tt = pipe.generate(inputs, infer_steps=INFER_STEPS, seed=SEED)
    latent_gen_s = time.perf_counter() - t0

    _, achieved_pcc = pcc(golden["target_latents"], tt["target_latents"])
    print(f"e2e PCC={achieved_pcc}", flush=True)
    assert achieved_pcc >= PCC_TARGET, f"target_latents PCC {achieved_pcc} < {PCC_TARGET}"

    t1 = time.perf_counter()
    waveform = latents_to_waveform(tt["target_latents"])
    vae_decode_s = time.perf_counter() - t1
    e2e_music_s = latent_gen_s + vae_decode_s

    save_wav(TT_WAV, waveform)

    peak = float(waveform.abs().max().item())
    assert torch.isfinite(waveform).all()
    assert waveform.shape[0] == 1 and waveform.shape[1] == 2 and waveform.shape[2] > 0
    assert peak < 10.0, f"waveform peak sanity check failed: {peak}"
    assert os.path.isfile(TT_WAV)

    print(f"PHASE_A latent_gen_s={latent_gen_s:.4f}", flush=True)
    print(f"PHASE_A vae_decode_s={vae_decode_s:.4f}", flush=True)
    print(f"PHASE_A e2e_music_s={e2e_music_s:.4f}", flush=True)
    print(f"PHASE_A output_wav={TT_WAV}", flush=True)
