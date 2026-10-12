# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E, Call 1 -- text_generation (text -> text) of mistralai/Devstral-Small-2-24B-Instruct-2512 on the 1x4 mesh.

Input provenance (TOOL-AUTHORED, no published example was discovered): demo/inputs.py TEXT_PROMPT through the
checkpoint's chat template, identical in all B = $TT_PERF_BATCH rows; rows differ only by sampling seed (= row).

Real input -> tt.pipeline.run_text_generation (token_embed -> ministral3_model[...] -> decoder_head -> on-device
sampling, decoding to EOS) -> generated text, compared to the HF golden (Source A) for every row.
"""
from __future__ import annotations

import json

from models.demos.devstral_small_2_24b_instruct_2512.demo.inputs import TEXT_PROMPT
from models.demos.devstral_small_2_24b_instruct_2512.tests.e2e.common import check_against_golden, check_free_run
from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P


def test_e2e_text_generation(pipe, processor, hf_model):
    """TT prefill+decode teacher-forced onto the independent HF model.generate() golden: tokens, logits PCC,
    prefill hidden level, Gate 1/2/3, for all B rows over the whole output (both sides stop on EOS)."""
    print(f"PERF_BATCH_STREAMS={pipe.batch}", flush=True)
    req = P.encode_request(processor, TEXT_PROMPT, batch=pipe.batch, kv_capacity=pipe.kv_capacity)
    golden = check_against_golden("text_generation", P.run_text_generation, pipe, hf_model, req)
    rec = P.PACKAGE_DIR / "tests" / "e2e" / "decode_horizon.json"
    rec.write_text(json.dumps({"text_generation": {"decode_steps": int(golden["tokens"].shape[1])}}, indent=2) + "\n")


def test_e2e_text_generation_free_run(pipe, processor, hf_model):
    """TT samples its own tokens to EOS on device; HF is teacher-forced onto that trajectory and must draw the
    same token at every step; the B seeds must produce distinct generations."""
    print(f"PERF_BATCH_STREAMS={pipe.batch}", flush=True)
    req = P.encode_request(processor, TEXT_PROMPT, batch=pipe.batch, kv_capacity=pipe.kv_capacity)
    check_free_run("text_generation", P.run_text_generation, pipe, processor, hf_model, req)
