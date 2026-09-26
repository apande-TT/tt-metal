# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Qwen-Image-Edit on TT (Call 1: image_edit), against the HF QwenImageEditPipeline golden.

Real input: B condition images (crops of the photos in models/sample_data), B distinct edit instructions
and B seeds, samples 0..B-1 of the bundled set. They are encoded with the HF Qwen2VLProcessor,
VaeImageProcessor and FlowMatch scheduler (tt/inputs.py). One chained TT forward
(tt/pipeline.py:run_image_edit, the same function the demo calls) turns them into B edited images. The
golden is the HF pipeline in float32 on CPU, with the same inputs and the same initial noise
(reference/golden.py, cached per batch and prompt set).

Batch: B is read from $TT_PERF_BATCH, falling back to the pipeline's own batch (EditConfig.batch = 32).
The test prints `PERF_BATCH_STREAMS=<B>` for the batch it actually drives. At B=32 the full 50-step
schedule takes ~100 min on this T3K (120 s per scheduler step with the precise transformer).

Gates:
  1  every routed graduated stub is ttnn: no torch compute in its forward code (static scan), and the
     forward fires zero host aten ops (host_op_observer, the authoritative runtime check)
  2  all 25 graduated modules were invoked by that forward
  3  every sample's final image PCC vs its own golden >= 0.99
Also asserted: the full scheduler schedule ran (no step cap), the outputs are distinct, and each output
matches its own golden better than any other sample's golden.

This is the correctness gate (E2E_CORRECTNESS_GATE). Its reference is the independent HF run, never the
TT trajectory, and it scores the final image, so every stage that produces the output is covered.
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

E2E_CORRECTNESS_GATE = "test_e2e_image_edit"
PCC_TARGET = 0.99
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
    batch = int(os.environ.get(BATCH_ENV, "0") or 0) or EditConfig().batch  # the harness's batch, else the pipeline's
    assert batch % 2 == 0, f"batch {batch}: the VAE runs batch-parallel over the 2 mesh rows"
    return EditConfig(batch=batch, num_inference_steps=steps)


@pytest.fixture(scope="module")
def hf_pipe():
    return P.load_hf_reference(torch.float32)


# the forward is ~100 min at B=32 x 50 steps; building a missing golden on CPU adds ~10 h at B=32
@pytest.mark.timeout(16 * 3600)
@MESH_PARAMS
@MESH
def test_e2e_image_edit(mesh_device, hf_pipe):
    cfg = _config()
    golden = load_or_build_golden(cfg, build_if_missing=True)  # built on CPU the first time, then cached
    assert golden is not None, f"golden missing: {golden_path(cfg)}"

    pipe = P.build_pipeline(mesh_device, model=hf_pipe, cfg=cfg)
    enc = pipe.encode(cfg)
    p = pipe.prepare(enc)
    B = p.B  # the batch this test actually drives, read from the pipeline
    assert B == cfg.batch == golden["image"].shape[0]
    print(batch_report_line(B), flush=True)
    print(f"[e2e] batch={B} steps={p.num_steps} size={enc.width}x{enc.height} cfg_scale={p.cfg_scale}", flush=True)

    # ---- the real forward, under the host-op observer (inputs already encoded + uploaded) ----------
    # The trace replays are enqueued non-blocking, so without this the host waits silently for the
    # whole ~100 min schedule and the harness's stall watch (log growth) reads it as a hang. Waiting
    # on each step and reporting it keeps the run visibly progressing; the ops and their order are
    # unchanged.
    def _report_step(i, _latents):
        ttnn.synchronize_device(mesh_device)
        print(f"[e2e] denoise step {i + 1}/{p.num_steps} done", flush=True)

    verdict, out = pipe.host_op_selftest(p, on_step=_report_step)
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
