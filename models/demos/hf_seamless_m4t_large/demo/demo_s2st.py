# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speech-to-speech demo for facebook/hf-seamless-m4t-large."""
from __future__ import annotations

import argparse
import math

import torch

import ttnn
from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline


def _synth_audio(seconds: float = 1.0, sr: int = 16000) -> torch.Tensor:
    t = torch.linspace(0.0, seconds, int(sr * seconds))
    return 0.1 * torch.sin(2 * math.pi * 220.0 * t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tgt-lang", default="eng")
    p.add_argument("--spkr-id", type=int, default=0)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    device = ttnn.open_device(device_id=0)
    try:
        pipe = build_pipeline(device)
        proc = pipe.hf_processor
        audio = _synth_audio().numpy()
        inputs = proc(audio=audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"]
        pcc, tt_wave, hf_wave = pipe.run_s2st(input_features, tgt_lang=args.tgt_lang, spkr_id=args.spkr_id, N=args.n)
        print(f"[demo_s2st] tt waveform shape={tuple(tt_wave.shape)}")
        print(f"[demo_s2st] e2e PCC={pcc}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
