"""Smoke entry for tt_hw_planner routing (ACE-Step v1.5 tt_dit pipeline)."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import GATE_CONFIG
from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pipeline_acestep_runs_on_device(device: ttnn.Device) -> None:
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    try:
        pipe = AceStepPipeline.create_pipeline(mesh_device=device)
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    latents = pipe(
        prompts=["test"],
        num_inference_steps=2,
        seed=GATE_CONFIG["seed"],
        traced=False,
    )

    assert isinstance(latents, torch.Tensor)
    assert latents.shape == (
        GATE_CONFIG["batch"],
        GATE_CONFIG["seq_len_latent"],
        GATE_CONFIG["audio_acoustic_hidden_dim"],
    )
