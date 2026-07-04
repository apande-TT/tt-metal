# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ACE-Step constrained LM decoding helpers."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from models.tt_dit.pipelines.acestep.lm_planner import have_lm_planner_weights, resolve_lm_planner_path
from models.tt_dit.pipelines.acestep.lm_planner_constrained import (
    configure_codes_phase,
    configure_cot_phase,
    create_constrained_processor,
    default_use_constrained_decoding,
)


def test_default_use_constrained_decoding_enabled() -> None:
    assert default_use_constrained_decoding() is True


@pytest.mark.skipif(not have_lm_planner_weights(model="1.7B"), reason="LM planner weights missing")
def test_constrained_processor_builds_audio_code_masks() -> None:
    weight_path = resolve_lm_planner_path(model="1.7B")
    tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True)
    processor = create_constrained_processor(tokenizer)
    configure_cot_phase(processor, enabled=True, stop_at_reasoning=True)
    assert processor.audio_code_mask is not None
    assert processor.non_audio_code_mask is not None

    configure_codes_phase(processor, enabled=True, target_duration=8.0)
    assert processor.target_codes == 40
    assert processor.generation_phase == "codes"
