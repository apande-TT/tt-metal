# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text-to-text demo for facebook/hf-seamless-m4t-large."""
from __future__ import annotations

import argparse

import ttnn
from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello, my dog is cute.")
    p.add_argument("--src-lang", default="eng")
    p.add_argument("--tgt-lang", default="fra")
    p.add_argument("--n", type=int, default=16, help="max_new_tokens cap")
    args = p.parse_args()

    device = ttnn.open_device(device_id=0)
    try:
        pipe = build_pipeline(device)
        proc = pipe.hf_processor
        inputs = proc(text=args.text, src_lang=args.src_lang, return_tensors="pt")
        input_ids = inputs["input_ids"]
        pcc, tt_tokens, hf_tokens = pipe.run_t2tt(input_ids, tgt_lang=args.tgt_lang, N=args.n)
        try:
            tt_text = proc.batch_decode([tt_tokens], skip_special_tokens=True)[0]
        except Exception:
            tt_text = "<decode-failed>"
        print(f"[demo_t2tt] tgt text: {tt_text!r}")
        print(f"[demo_t2tt] e2e PCC={pcc}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
