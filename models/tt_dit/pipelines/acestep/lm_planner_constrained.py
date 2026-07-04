# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers for ACE-Step FSM constrained decoding (host + TT LM planner)."""
from __future__ import annotations

import os
from typing import Any

from models.tt_dit.pipelines.acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor


def default_use_constrained_decoding() -> bool:
    value = os.environ.get("ACESTEP_USE_CONSTRAINED_LM_DECODING")
    if value is not None:
        return value.strip().lower() in ("1", "true", "yes")
    return True


def create_constrained_processor(tokenizer: Any, *, debug: bool = False) -> MetadataConstrainedLogitsProcessor:
    """Create reusable processor (ACE-Step default: skip genres field)."""
    return MetadataConstrainedLogitsProcessor(
        tokenizer,
        enabled=True,
        debug=debug,
        skip_genres=True,
    )


def configure_cot_phase(
    processor: MetadataConstrainedLogitsProcessor,
    *,
    enabled: bool,
    stop_at_reasoning: bool = True,
) -> None:
    processor.reset()
    processor.enabled = enabled
    processor.set_generation_phase("cot")
    processor.set_stop_at_reasoning(stop_at_reasoning)
    processor.set_skip_genres(True)
    processor.set_target_duration(None)


def configure_codes_phase(
    processor: MetadataConstrainedLogitsProcessor,
    *,
    enabled: bool,
    target_duration: float | None,
) -> None:
    processor.reset()
    processor.enabled = enabled
    processor.set_generation_phase("codes")
    processor.set_stop_at_reasoning(False)
    processor.set_skip_genres(True)
    processor.set_target_duration(target_duration)
