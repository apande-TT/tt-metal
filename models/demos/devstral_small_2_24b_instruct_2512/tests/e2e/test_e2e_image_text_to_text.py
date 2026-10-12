# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E, Call 2 -- image_text_to_text (image + text -> text) of mistralai/Devstral-Small-2-24B-Instruct-2512.

Input provenance (TOOL-AUTHORED, no published example was discovered): demo/inputs.py load_image() (a 224x224
render of a two-line Python function) + IMAGE_PROMPT through the PixtralProcessor and the checkpoint's chat
template, identical in all B = $TT_PERF_BATCH rows; rows differ only by sampling seed (= row).

Real input -> tt.pipeline.run_image_text_to_text (pixtral_vision_model -> mistral3_multi_modal_projector -> [IMG]
merge -> ministral3_model -> decoder_head -> on-device sampling to EOS) -> generated text vs the HF golden.

This file holds THE correctness gate: test_e2e_image_text_to_text runs every graduated stub (all 22) through all
three stages (encode, prefill, decode) against an HF golden computed independently of the TT run.
"""
from __future__ import annotations

from models.demos.devstral_small_2_24b_instruct_2512.demo.inputs import IMAGE_PROMPT, load_image
from models.demos.devstral_small_2_24b_instruct_2512.tests.e2e.common import check_against_golden, check_free_run
from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P

E2E_CORRECTNESS_GATE = "test_e2e_image_text_to_text"


def test_e2e_image_text_to_text(pipe, processor, hf_model):
    """GATE: TT encode+prefill+decode teacher-forced onto the independent HF model.generate() golden: tokens,
    logits PCC, vision-feature and prefill-hidden level, Gate 1/2/3, all B rows, whole output to EOS."""
    print(f"PERF_BATCH_STREAMS={pipe.batch}", flush=True)
    req = P.encode_request(processor, IMAGE_PROMPT, image=load_image(), batch=pipe.batch, kv_capacity=pipe.kv_capacity)
    check_against_golden("image_text_to_text", P.run_image_text_to_text, pipe, hf_model, req)


def test_e2e_image_text_to_text_free_run(pipe, processor, hf_model):
    """TT samples its own tokens to EOS on device; HF is teacher-forced onto that trajectory and must draw the
    same token at every step; the B seeds must produce distinct generations."""
    print(f"PERF_BATCH_STREAMS={pipe.batch}", flush=True)
    req = P.encode_request(processor, IMAGE_PROMPT, image=load_image(), batch=pipe.batch, kv_capacity=pipe.kv_capacity)
    check_free_run("image_text_to_text", P.run_image_text_to_text, pipe, processor, hf_model, req)
