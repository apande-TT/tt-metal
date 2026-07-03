# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""E2E PCC gate with trace + 2-CQ decoder execution enabled."""
from __future__ import annotations

import os

import pytest
import torch

from models.demos.hf_eager.acestep_v15_base.tests.e2e.test_e2e_generate_audio import (
    _DEMO_ROOT,
    INFER_STEPS,
    PCC_TARGET,
    SEED,
)
from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model, pcc
from models.demos.hf_eager.acestep_v15_base.tt.hf_reference import hf_generate_reference
from models.demos.hf_eager.acestep_v15_base.tt.invocation_tracker import track_invocations
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT

TRACE_DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "trace_region_size": 50_000_000,
}


@pytest.mark.parametrize("device_params", [TRACE_DEVICE_PARAMS], indirect=True)
def test_e2e_generate_audio_traced(device_params, device):
    torch.manual_seed(SEED)

    hf_model = load_hf_model()
    inputs = build_inputs(seed=SEED)

    with torch.no_grad():
        golden = hf_generate_reference(hf_model, inputs, infer_steps=INFER_STEPS, seed=SEED)

    pipe = AceStepPipelineTT(device, hf_model)

    with track_invocations():
        tt = pipe.generate(inputs, infer_steps=INFER_STEPS, seed=SEED, traced=True, use_2cq=True)

    for i, (xg, xt_) in enumerate(zip(golden["per_step_xt"], tt["per_step_xt"])):
        _, step_pcc = pcc(xg, xt_)
        print(f"[e2e-traced] decoder state step {i} PCC={step_pcc}", flush=True)

    _, achieved_pcc = pcc(golden["target_latents"], tt["target_latents"])
    print(f"[e2e-traced] e2e PCC={achieved_pcc}", flush=True)

    fallbacks_path = os.path.join(_DEMO_ROOT, "_runtime_fallbacks.json")
    fallbacks = {}
    if os.path.isfile(fallbacks_path):
        import json

        try:
            fallbacks = json.loads(open(fallbacks_path).read() or "{}")
        except Exception:
            fallbacks = {}

    assert not fallbacks, f"Gate 1 FAILED: runtime torch fallbacks present: {fallbacks}"
    assert achieved_pcc >= PCC_TARGET, f"Gate 3 FAILED: traced e2e PCC {achieved_pcc} < {PCC_TARGET}"

    if pipe._traced_condition_encoder is not None:
        pipe._traced_condition_encoder.release()
    if pipe._traced_audio_path is not None:
        pipe._traced_audio_path.release()
    if pipe._traced_decoder is not None:
        pipe._traced_decoder.release()
