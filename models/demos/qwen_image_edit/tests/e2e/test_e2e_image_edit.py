# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Qwen-Image-Edit on TT (Call 1: image_edit), against the HF QwenImageEditPipeline golden.

Real input: B condition images (B = $TT_PERF_BATCH, 32 when unset; crops of the photos in
models/sample_data), B distinct edit instructions and B seeds from 1000 (the bundled set). They are
encoded with the HF Qwen2VLProcessor, VaeImageProcessor and FlowMatch scheduler (tt/inputs.py). One chained TT forward (tt/pipeline.py:
run_image_edit, the same function the demo calls) takes them to B edited images in one program per
step. The golden is the HF pipeline in float32 on CPU with the same inputs and the same initial noise
(reference/golden.py). It takes hours on CPU, so it is precomputed and cached; this test fails fast if
it is missing and never builds it.

Mesh: Galaxy 8x4. The denoise batch is split over the 4 mesh columns (DP=4, 8 samples per column)
with the transformer TP=8 down each column, which is what makes B=32 x 50 steps fit the 45 min the
harness allows the whole run. Every sample is scored against its own golden.

Gates:
  1  every routed graduated stub is ttnn: no torch compute in its forward code (static scan), and the
     forward fires zero host aten ops (host_op_observer, the authoritative runtime check)
  2  all 25 graduated modules were invoked by that forward
  3  every sample's final image PCC vs its own golden >= PCC_TARGET (0.95)
Also asserted: the full scheduler schedule ran (no step cap); the outputs are distinct; each output
matches its own golden better than any other sample's golden.
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen_image_edit.reference.golden import golden_path, load_or_build_golden
from models.demos.qwen_image_edit.tt import gates
from models.demos.qwen_image_edit.tt import pipeline as P
from models.demos.qwen_image_edit.tt.inputs import EditConfig
from models.experimental.perf_automation.agent.perf_adapter import BATCH_ENV, batch_report_line

# the gate's bar for this run (emit-e2e --pcc-target 0.95), applied to EVERY sample
PCC_TARGET = 0.95
# the batch the harness asks for (perf_adapter.BATCH_ENV); the gate's batch, 32, when it asks for none
E2E_BATCH = int(os.environ.get(BATCH_ENV) or 32)
# the correctness gate: per-sample image PCC vs the independently computed HF golden (not teacher-forced)
E2E_CORRECTNESS_GATE = "test_e2e_image_edit"
STUB_PKGS = {
    "text_encoder": "models.demos.qwen_image_edit_text_encoder._stubs.",
    "vae": "models.tt_dit.pipelines.qwen_image_edit_vae._stubs.",
    "transformer": "models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.",
}
# the non-graduated ports the chain routes through (their forwards are scanned too)
GLUE = {
    "text_encoder": ["attention", "encoder_stack", "layer", "token_embed", "v_l_rotary_embedding"],
    "vae": ["mlp", "_resident"],
    "transformer": ["attention", "encoder_stack", "patch_embed", "decoder_head", "_ccl", "_precise"],
}
CHAIN = ["pipeline", "text_encoder", "transformer", "vae"]

MESH_PARAMS = pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "trace_region_size": P.DEVICE_PARAMS["trace_region_size"],
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
MESH = pytest.mark.parametrize("mesh_device", [P.MESH_SHAPE], indirect=True)


def _config():
    steps = int(os.environ.get("QIE_STEPS", "50"))  # QwenImageEditPipeline default
    return EditConfig(batch=E2E_BATCH, num_inference_steps=steps)


@pytest.fixture(scope="module")
def golden():
    """The cached HF golden. Requested first so a missing golden fails before the device or the HF
    model is touched; it is never built here (hours on CPU: python -m models.demos.qwen_image_edit.
    reference.golden --batch 32 --steps 50)."""
    cfg = _config()
    g = load_or_build_golden(cfg, build_if_missing=False)
    if g is None:
        pytest.fail(f"cached golden missing: {golden_path(cfg)} (build it with reference/golden.py first)")
    return g


@pytest.fixture(scope="module")
def hf_pipe():
    return P.load_hf_reference(torch.float32)


# the harness's hang budget for the whole run (build + 32-sample x 50-step forward + checks)
@pytest.mark.timeout(2700)
@MESH_PARAMS
@MESH
def test_e2e_image_edit(golden, mesh_device, hf_pipe):
    cfg = _config()

    pipe = P.build_pipeline(mesh_device, model=hf_pipe, cfg=cfg)
    enc = pipe.encode(cfg)
    p = pipe.prepare(enc)
    B = p.B  # the batch this test actually drives, read from the pipeline
    assert B == cfg.batch == golden["image"].shape[0]
    print(batch_report_line(B), flush=True)
    print(f"[e2e] batch={B} steps={p.num_steps} size={enc.width}x{enc.height} cfg_scale={p.cfg_scale}", flush=True)

    # ---- the real forward, under the host-op observer (inputs already encoded + uploaded) ----------
    verdict, out = pipe.host_op_selftest(p)
    image = P.to_host(out).to(torch.float32)
    latents = P.to_host(pipe.last_latents).to(torch.float32)
    ref = golden["image"].to(torch.float32)

    # ---- horizon: the whole schedule ran (no cap) ------------------------------------------------------
    assert pipe.steps_run == p.num_steps == cfg.num_inference_steps, (pipe.steps_run, p.num_steps)

    # ---- Gate 1: native ttnn ---------------------------------------------------------------------------
    stub_mods = [importlib.import_module(STUB_PKGS[g] + n) for g, ns in P.GRADUATED.items() for n in ns]
    glue_mods = [importlib.import_module(STUB_PKGS[g] + n) for g, ns in GLUE.items() for n in ns]
    chain_mods = [importlib.import_module("models.demos.qwen_image_edit.tt." + n) for n in CHAIN]
    violations = gates.gate1_native(stub_mods + glue_mods, chain_mods)
    print(f"[gate1] static torch-compute scan: {violations or 'clean'}", flush=True)
    print(f"[gate1] host_op_observer: on_device={verdict['on_device']} host_ops={verdict['host_ops'][:12]}", flush=True)

    # ---- Gate 2: every graduated module invoked --------------------------------------------------------
    counts, missing = gates.gate2_invoked(pipe.tracker, P.GRADUATED_ALL)
    print(f"[gate2] invocation counts: {counts}", flush=True)

    # ---- Gate 3: per-sample image PCC vs its own golden ------------------------------------------------
    pccs = [float(comp_pcc(ref[b], image[b], PCC_TARGET)[1]) for b in range(B)]
    lat_pccs = [float(comp_pcc(golden["step_latents"][-1][b], latents[b], PCC_TARGET)[1]) for b in range(B)]
    for b in range(B):
        print(
            f"[gate3] sample {b:2d} image PCC {pccs[b]:.6f} latent PCC {lat_pccs[b]:.6f}  '{enc.prompts[b]}'",
            flush=True,
        )
    # independence: distinct outputs, and each output is closest to its own golden
    cross = torch.tensor([[float(comp_pcc(ref[j], image[i], 0.0)[1]) for j in range(B)] for i in range(B)])
    own_best = bool((cross.argmax(dim=1) == torch.arange(B)).all())
    distinct = all((image[i] - image[j]).abs().max() > 1e-3 for i in range(B) for j in range(i + 1, B))
    achieved_pcc = min(pccs)
    print(f"[e2e] independence: distinct={distinct} own-golden-best={own_best}", flush=True)
    print(f"e2e PCC={achieved_pcc}", flush=True)

    assert not violations, f"Gate 1: torch compute in the forward: {violations}"
    assert verdict["on_device"], f"Gate 1: host aten ops in the forward: {verdict['host_ops']}"
    assert not missing, f"Gate 2: graduated modules never invoked: {missing}"
    assert distinct and own_best, f"outputs are not {B} independent samples"
    assert achieved_pcc >= PCC_TARGET, f"Gate 3: min image PCC {achieved_pcc:.6f} < {PCC_TARGET} ({pccs})"
