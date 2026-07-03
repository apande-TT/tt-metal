# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal CLI: TT latent generation + host Oobleck VAE -> WAV.

Run:
    flock /tmp/tt_ace_device.lock ./python_env/bin/python -m \
        models.demos.hf_eager.acestep_v15_base.demo.demo_generate_audio_wav \
        --output /tmp/acestep_demo.wav
"""
from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import latents_to_waveform, save_wav


def main():
    ap = argparse.ArgumentParser(description="ACE-Step v1.5 TT latents + host VAE WAV demo")
    ap.add_argument("--infer-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=str, default="/tmp/acestep_demo.wav")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hf_model = load_hf_model()
    inputs = build_inputs(seed=args.seed)

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        pipe = AceStepPipelineTT(device, hf_model)
        t0 = time.perf_counter()
        out = pipe.generate(inputs, infer_steps=args.infer_steps, seed=args.seed)
        latent_gen_s = time.perf_counter() - t0
    finally:
        ttnn.close_device(device)

    t1 = time.perf_counter()
    waveform = latents_to_waveform(out["target_latents"])
    vae_decode_s = time.perf_counter() - t1
    save_wav(args.output, waveform)

    print(f"PHASE_A latent_gen_s={latent_gen_s:.4f}", flush=True)
    print(f"PHASE_A vae_decode_s={vae_decode_s:.4f}", flush=True)
    print(f"PHASE_A e2e_music_s={latent_gen_s + vae_decode_s:.4f}", flush=True)
    print(f"PHASE_A output_wav={args.output}", flush=True)


if __name__ == "__main__":
    main()
