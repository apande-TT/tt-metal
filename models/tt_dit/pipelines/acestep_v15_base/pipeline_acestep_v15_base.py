# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step v1.5 pipeline alias — re-exports the canonical tt_dit acestep module."""

from __future__ import annotations

from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline, AceStepPipelineConfig

__all__ = ["AceStepPipeline", "AceStepPipelineConfig"]
