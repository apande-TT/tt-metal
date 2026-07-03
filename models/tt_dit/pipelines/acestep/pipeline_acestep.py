# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step v1.5 flow-matching audio pipeline (tt_dit production entry)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT
from models.demos.hf_eager.acestep_v15_base.tt.traced_decoder import _device_supports_2cq
from models.tt_dit.pipelines.events import PipelineEventCallback, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence

_DEFAULT_CHECKPOINT = "ACE-Step/acestep-v15-base"


@dataclass(frozen=True, kw_only=True)
class AceStepPipelineConfig:
    """Minimal config for ACE-Step host orchestration on a 1x1 mesh (p150)."""

    checkpoint_name: str = _DEFAULT_CHECKPOINT
    cfg_enabled: bool = True
    num_inference_steps: int = 4
    sample_rate: int = 48000
    pool_window_size: int = 5


class AceStepPipeline(PipelineAPIMixin):
    """Host-side ACE-Step v1.5 pipeline.

    v0 scope: text+lyric+timbre conditioning (captured inputs) → flow-matching
    DiT denoising (trace + 2-CQ on the decoder hot path when ``traced=True``) →
    ``target_latents`` only (no VAE waveform yet).

    Delegates subsystem math to the graduated hf_eager TT stubs via
    ``AceStepPipelineTT``; this module adds tt_dit config/device wiring and
    profiler section events.
    """

    def __init__(
        self,
        *,
        device: ttnn.Device | ttnn.MeshDevice,
        config: AceStepPipelineConfig,
    ) -> None:
        self._device = device
        self._config = config
        self._hf_model = load_hf_model()
        self._inner = AceStepPipelineTT(device, self._hf_model)

    @classmethod
    def create_pipeline(
        cls,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        *,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
        cfg_enabled: bool = True,
        num_inference_steps: int = 4,
        sample_rate: int = 48000,
        pool_window_size: int = 5,
    ) -> AceStepPipeline:
        config = AceStepPipelineConfig(
            checkpoint_name=checkpoint_name,
            cfg_enabled=cfg_enabled,
            num_inference_steps=num_inference_steps,
            sample_rate=sample_rate,
            pool_window_size=pool_window_size,
        )
        return cls(device=mesh_device, config=config)

    @staticmethod
    def _use_2cq(device: ttnn.Device | ttnn.MeshDevice, traced: bool) -> bool:
        return traced and _device_supports_2cq(device)

    @staticmethod
    def _event_callback(on_event: PipelineEventCallback):
        def _forward(section_event):
            on_event(section_event)

        return _forward

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: AbcSequence[str],
        negative_prompts: AbcSequence[str] | None = None,
        num_inference_steps: int | None = None,
        seed: int = 0,
        traced: bool = True,
        on_event: PipelineEventCallback | None = None,
    ):
        on_event = on_event if on_event is not None else null_callback

        if negative_prompts is not None and self._config.cfg_enabled:
            logger.warning("ACE-Step v0: negative_prompts ignored (CFG not wired yet)")

        if prompts:
            logger.debug("ACE-Step v0: prompts ignored; using captured inputs from hf_eager build_inputs()")

        infer_steps = num_inference_steps if num_inference_steps is not None else self._config.num_inference_steps
        inputs = build_inputs(seed=seed if seed else None)
        use_2cq = self._use_2cq(self._device, traced)
        if traced:
            logger.debug(f"ACE-Step denoising: traced=True use_2cq={use_2cq}")

        result = self._inner.generate(
            inputs,
            infer_steps=infer_steps,
            seed=seed if seed else None,
            pool_window_size=self._config.pool_window_size,
            traced=traced,
            use_2cq=use_2cq,
            on_event=self._event_callback(on_event),
        )
        return result["target_latents"]
