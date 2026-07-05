# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Base SeamlessM4TModel demo — dispatches to text- or speech-output path."""
from __future__ import annotations

import argparse

import ttnn
from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello, my dog is cute.")
    p.add_argument("--src-lang", default="eng")
    p.add_argument("--tgt-lang", default="fra")
    p.add_argument("--generate-speech", action="store_true")
    p.add_argument("--spkr-id", type=int, default=0)
    p.add_argument("--n", type=int, default=16)
    args = p.parse_args()

    device = ttnn.open_device(device_id=0)
    try:
        pipe = build_pipeline(device)
        proc = pipe.hf_processor
        inputs = proc(text=args.text, src_lang=args.src_lang, return_tensors="pt")
        input_ids = inputs["input_ids"]
        pcc, out_a, out_b = pipe.run_base(
            generate_speech=args.generate_speech,
            input_ids=input_ids,
            tgt_lang=args.tgt_lang,
            spkr_id=args.spkr_id,
            N=args.n,
        )
        if args.generate_speech:
            print(f"[demo_base] tt waveform shape={tuple(out_a.shape)}")
        else:
            try:
                tt_text = proc.batch_decode([out_a], skip_special_tokens=True)[0]
            except Exception:
                tt_text = "<decode-failed>"
            print(f"[demo_base] tgt text: {tt_text!r}")
        print(f"[demo_base] e2e PCC={pcc}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
