# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pipeline's trace / host-op / depth contract (COMMAND 3), on the same 2x4 mesh as the e2e gate.

These run at a capped depth (every repeated stack built `layers` times): the op set is the full model's.
"""

from __future__ import annotations

import pytest

from models.demos.qwen_image_edit.demo.mesh import DEVICE_PARAMS, MESH_SHAPE
from models.demos.qwen_image_edit.tt.pipeline import (
    PIPELINE_STAGES,
    build_pipeline,
    host_op_selftest,
    trace_capture_selftest,
)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_trace_capture_every_stage(mesh_device):
    assert trace_capture_selftest(mesh_device, layers=2)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_forward_is_on_device(mesh_device):
    v = host_op_selftest(mesh_device, layers=2)
    print(v)
    assert v["on_device"], v["reason"]


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_depth_knobs(mesh_device):
    pipe = build_pipeline(mesh_device, layers=2, text_encode_layers=4)
    assert len(pipe.text_encoder.visual.blocks) == 2
    assert pipe.text_encoder.num_text_layers == 4
    assert len(pipe.transformer.transformer_blocks) == 2
    for stage in PIPELINE_STAGES:
        for suffix in ("trace_setup", "trace_step", "trace_inputs", "trace_items"):
            assert callable(getattr(pipe, f"{stage}_{suffix}"))
