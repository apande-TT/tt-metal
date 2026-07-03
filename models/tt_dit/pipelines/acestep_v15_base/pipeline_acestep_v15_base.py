# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step v1.5 flow-matching audio pipeline (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence

    from models.tt_dit.pipelines.events import PipelineEventCallback


@dataclass(frozen=True, kw_only=True)
class AceStepPipelineConfig:
    """Minimal config for ACE-Step host orchestration scaffold."""

    checkpoint_name: str = "ACE-Step/acestep_v15_base-v15-base"
    cfg_enabled: bool = True
    num_inference_steps: int = 30
    sample_rate: int = 48000


class AceStepPipeline(PipelineAPIMixin):
    """Host-side flow-matching loop scaffold for ACE-Step.

    Wire points (not yet implemented on device):
      - ``ConditionEncoder``: text + lyric + timbre conditioning
      - ``AceStepDiT``: 24-layer diffusion transformer
      - ``FSQCodec``: audio tokenizer / detokenizer (host)
      - ``AudioDecoder``: VAE + vocoder tail
    """

    def __init__(self, config: AceStepPipelineConfig) -> None:
        self._config = config

    @classmethod
    def create_pipeline(cls, config: AceStepPipelineConfig) -> AceStepPipeline:
        return cls(config)

    def __call__(
        self,
        *,
        prompts: AbcSequence[str],
        negative_prompts: AbcSequence[str] | None = None,
        num_inference_steps: int = 30,
        seed: int = 0,
        traced: bool = True,
        on_event: PipelineEventCallback | None = None,
    ) -> Any:
        raise NotImplementedError(
            "AceStepPipeline scaffold only. Port order: condition encoders → "
            "DiT denoising loop (reuse flux1/SD3.5 CFG+EulerSolver) → FSQ → VAE/vocoder."
        )
