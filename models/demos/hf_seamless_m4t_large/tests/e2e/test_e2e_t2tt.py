# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E test for SeamlessM4TForTextToText via the shared TT pipeline."""
from __future__ import annotations

from models.demos.hf_seamless_m4t_large.tt.pipeline import build_pipeline


def test_e2e_t2tt(device):
    pipe = build_pipeline(device)
    proc = pipe.hf_processor
    inputs = proc(text="Hello, my dog is cute.", src_lang="eng", return_tensors="pt")
    pcc, tt_tokens, hf_tokens = pipe.run_t2tt(inputs["input_ids"], tgt_lang="fra", N=16)
    print(f"e2e PCC={pcc}")
    pipe.assert_gates("t2tt", pcc, min_pcc=0.95)
