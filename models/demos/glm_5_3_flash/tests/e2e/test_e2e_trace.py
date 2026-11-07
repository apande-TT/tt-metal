# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace contract + fully-on-device check for the GLM-5.3-Flash pipeline (tt/pipeline.py).

Inputs (provenance): the published example (HF docstring of Glm5NextForConditionalGeneration) stored under
_captured/e2e_example/, encoded by the HF processor; batch read from the pipeline ($TT_PERF_BATCH).
"""

from __future__ import annotations

import os

import pytest

import ttnn
from models.demos.glm_5_3_flash.tt import pipeline as P


def _depth():
    n = os.environ.get("TT_PERF_LAYERS")
    return int(n) if n else None


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": P.TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 32)], indirect=True)
def test_trace_and_host_ops(mesh_device):
    """Every stage captures host-free in its own trace and replays to its eager output; the whole model
    forward fires no host aten op."""
    assert mesh_device.get_num_devices() == 32, "needs the 32-chip Galaxy mesh (TP=32)"
    pipe = P.build_pipeline(mesh_device, layers=_depth())
    print(f"PERF_BATCH_STREAMS={pipe.batch}", flush=True)
    v = P.host_op_selftest(pipe=pipe)
    print(f"host_op_selftest: {v['reason']}", flush=True)
    ok, report = P.trace_capture_selftest(mesh_device, pipe=pipe)
    print(f"trace_capture_selftest: {report}", flush=True)
    assert v["on_device"], v["reason"]
    assert ok, report
