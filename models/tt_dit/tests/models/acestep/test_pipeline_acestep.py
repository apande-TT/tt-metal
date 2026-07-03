"""Smoke entry for tt_hw_planner routing (scaffold — expects NotImplementedError)."""

from __future__ import annotations

from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline, AceStepPipelineConfig


def test_pipeline_acestep_scaffold_raises_not_implemented(expect_error) -> None:
    pipe = AceStepPipeline.create_pipeline(AceStepPipelineConfig())
    with expect_error(NotImplementedError, "."):
        pipe(prompts=["test"], num_inference_steps=1, seed=0, traced=False)
