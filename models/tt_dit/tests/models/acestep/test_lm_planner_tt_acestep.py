# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 7B device gate: TT 5Hz LM planner prefill + generation smoke."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.pipelines.acestep.lm_planner import (
    build_formatted_prompt_cot,
    generate_audio_codes,
    have_lm_planner_weights,
    parse_audio_code_string,
    resolve_lm_planner_path,
)
from models.tt_dit.pipelines.acestep.lm_planner_tt import prefill_last_token_logits_tt

PREFILL_PCC_TARGET = 0.90
FIXED_CAPTION = "calm ambient piano, soft pads, meditative"
FIXED_LYRICS = "[verse]\nQuiet night\n[chorus]\nPeaceful stars\n"
FIXED_DURATION = 2.0
FIXED_SEED = 42


@pytest.mark.skipif(not have_lm_planner_weights(model="1.7B"), reason="LM planner weights missing")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_prefill_last_token_logits_tt_matches_host(device) -> None:
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    weight_path = resolve_lm_planner_path(model="1.7B")
    tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True)
    prompt = build_formatted_prompt_cot(tokenizer, FIXED_CAPTION, FIXED_LYRICS)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    hf_model = AutoModelForCausalLM.from_pretrained(
        weight_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    with torch.inference_mode():
        host_logits = hf_model(input_ids).logits[0, -1, :].float().cpu()

    tt_logits = prefill_last_token_logits_tt(device, prompt, model="1.7B")
    min_vocab = min(host_logits.numel(), tt_logits.numel())
    ok, value = comp_pcc(host_logits[:min_vocab], tt_logits[:min_vocab], PREFILL_PCC_TARGET)
    print(f"LM_PLANNER_TT_PREFILL_PCC: {value:.6f} vocab={min_vocab}", flush=True)
    assert ok, f"TT LM prefill logits PCC {value:.6f} < {PREFILL_PCC_TARGET}"


@pytest.mark.skipif(not have_lm_planner_weights(model="1.7B"), reason="LM planner weights missing")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_generate_audio_codes_tt_smoke(device) -> None:
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    codes, metadata = generate_audio_codes(
        caption=FIXED_CAPTION,
        lyrics=FIXED_LYRICS,
        audio_duration=FIXED_DURATION,
        model="1.7B",
        seed=FIXED_SEED,
        temperature=0.0,
        mesh_device=device,
        use_tt=True,
    )
    parsed = parse_audio_code_string(codes)
    assert len(parsed) == int(FIXED_DURATION * 5)
    assert metadata or codes
