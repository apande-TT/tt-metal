# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E test for SeamlessM4TForSpeechToSpeech via the shared TT pipeline."""
from __future__ import annotations

import math

import torch

from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline


def _synth_audio():
    t = torch.linspace(0.0, 1.0, 16000)
    return (0.1 * torch.sin(2 * math.pi * 220.0 * t)).numpy()


def test_e2e_s2st(device):
    pipe = build_pipeline(device)
    proc = pipe.hf_processor
    inputs = proc(audio=_synth_audio(), sampling_rate=16000, return_tensors="pt")
    pcc, tt_wave, hf_wave = pipe.run_s2st(inputs["input_features"], tgt_lang="eng", spkr_id=0, N=8)
    print(f"e2e PCC={pcc}")
    pipe.assert_gates("s2st", pcc, min_pcc=0.95)
