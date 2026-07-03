# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step v1.5 flow-matching audio pipeline (tt_dit production entry)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT
from models.demos.hf_eager.acestep_v15_base.tt.traced_decoder import _device_supports_2cq
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import encode_reference_audio
from models.tt_dit.pipelines.acestep.audio_decode import decode_latents_to_waveform
from models.tt_dit.pipelines.events import PipelineEventCallback, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence

_DEVICE_LOG = "/tmp/acestep_agent_device.log"

_REFERENCE_INPUT_KEYS = (
    "refer_audio_acoustic_hidden_states_packed",
    "refer_audio_order_mask",
    "src_latents",
    "chunk_masks",
    "attention_mask",
    "silence_latent",
    "is_covers",
)


def _log_device_progress(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {message}\n"
    try:
        with open(_DEVICE_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


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

    Functional path (``traced=False`` default): live prompt/lyrics via Qwen3 text
    encoder and optional reference WAV via host Oobleck VAE; falls back to
    captured hf_eager inputs when prompts are omitted. DiT denoising supports
    trace + 2-CQ on the decoder hot path when ``traced=True``.

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

    @staticmethod
    def _resolve_use_tt_vae(use_tt_vae: bool | None) -> bool:
        if use_tt_vae is not None:
            return use_tt_vae
        value = os.environ.get("ACESTEP_USE_TT_VAE", "0")
        return value.strip().lower() in ("1", "true", "yes")

    @staticmethod
    def _prepare_inputs(
        *,
        prompts: AbcSequence[str] | None,
        lyrics: str | AbcSequence[str] | None,
        reference_audio: str | None,
        seed: int,
        hf_model,
    ) -> dict:
        """Assemble DiT inputs from captured, live text, and/or reference audio."""
        use_live_text = bool(prompts)
        if use_live_text:
            _log_device_progress(
                f"prepare_inputs: live text (prompts={len(prompts)}, lyrics={'set' if lyrics else 'default'})"
            )
            inputs = build_inputs(
                seed=seed if seed else None,
                use_captured=False,
                prompts=prompts,
                lyrics=lyrics,
            )
        else:
            _log_device_progress("prepare_inputs: captured hf_eager conditioning")
            inputs = build_inputs(seed=seed if seed else None)

        if reference_audio:
            _log_device_progress(f"prepare_inputs: reference_audio={reference_audio}")
            ref_tensors = encode_reference_audio(
                reference_audio,
                hf_model=hf_model,
                seed=seed,
            )
            for key in _REFERENCE_INPUT_KEYS:
                inputs[key] = ref_tensors[key]
            _log_device_progress(
                f"prepare_inputs: is_covers={inputs['is_covers'].tolist()} "
                f"src_latents={tuple(inputs['src_latents'].shape)}"
            )

        return inputs

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompts: AbcSequence[str] | None = None,
        lyrics: str | AbcSequence[str] | None = None,
        reference_audio: str | None = None,
        negative_prompts: AbcSequence[str] | None = None,
        num_inference_steps: int | None = None,
        seed: int = 0,
        traced: bool = False,
        on_event: PipelineEventCallback | None = None,
        return_waveform: bool = False,
        use_tt_vae: bool | None = None,
    ):
        on_event = on_event if on_event is not None else null_callback

        if negative_prompts is not None and self._config.cfg_enabled:
            logger.warning("ACE-Step v0: negative_prompts ignored (CFG not wired yet)")

        infer_steps = num_inference_steps if num_inference_steps is not None else self._config.num_inference_steps
        inputs = self._prepare_inputs(
            prompts=prompts,
            lyrics=lyrics,
            reference_audio=reference_audio,
            seed=seed,
            hf_model=self._hf_model,
        )
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
        target_latents = result["target_latents"]
        if not return_waveform:
            return target_latents

        decode_tt_vae = self._resolve_use_tt_vae(use_tt_vae)
        waveform, vae_decode_s = decode_latents_to_waveform(
            self._device,
            target_latents,
            use_tt_vae=decode_tt_vae,
        )
        return {
            "target_latents": target_latents,
            "waveform": waveform,
            "vae_decode_s": vae_decode_s,
        }
