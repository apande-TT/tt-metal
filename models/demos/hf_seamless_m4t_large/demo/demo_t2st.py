# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text-to-speech demo for facebook/hf-seamless-m4t-large."""
from __future__ import annotations

import argparse

import ttnn
from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello, my dog is cute.")
    p.add_argument("--src-lang", default="eng")
    p.add_argument("--tgt-lang", default="eng")
    p.add_argument("--spkr-id", type=int, default=0)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    device = ttnn.open_device(device_id=0)
    try:
        pipe = build_pipeline(device)
        proc = pipe.hf_processor
        inputs = proc(text=args.text, src_lang=args.src_lang, return_tensors="pt")
        input_ids = inputs["input_ids"]
        pcc, tt_wave, hf_wave = pipe.run_t2st(input_ids, tgt_lang=args.tgt_lang, spkr_id=args.spkr_id, N=args.n)
        print(f"[demo_t2st] tt waveform shape={tuple(tt_wave.shape)}")
        print(f"[demo_t2st] e2e PCC={pcc}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
