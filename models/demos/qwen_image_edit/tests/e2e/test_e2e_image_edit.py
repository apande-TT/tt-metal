# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E gate for Call 1, image_edit: image + instruction -> edited image, on the 2x4 mesh.

Inputs (provenance): 32 x the QwenImageEditPipeline docstring example 1 (yarn-art-pikachu.png + its
prompt, 50 steps), seeds 0..B-1 (seed 0 from the Qwen-Image-Edit README example 1), at the 256 x 256
condition area (tt/inputs.py). B is $TT_PERF_BATCH (default 32).

Golden: the HF QwenImageEditPipeline itself, float32 on CPU, same inputs and generators
(reference/golden.py; built on first use and cached).

  Gate 1  every routed graduated stub (and the chain around it) is ttnn: no torch compute in its forward
          code (static scan, tt/gates.py), and the forward (everything after the input upload) fires
          zero host aten ops (host_op_observer, the authoritative runtime check)
  Gate 2  every one of the 25 graduated modules is invoked on the forward path
  Gate 3  min over samples of image PCC vs the golden >= 0.99 (printed as `e2e PCC=` on every run)
"""

from __future__ import annotations

import importlib

import pytest
import torch

from models.demos.qwen_image_edit.demo.mesh import DEVICE_PARAMS, MESH_SHAPE
from models.demos.qwen_image_edit.reference.golden import load_golden
from models.demos.qwen_image_edit.tt import gates
from models.demos.qwen_image_edit.tt import inputs as I
from models.demos.qwen_image_edit.tt.pipeline import build_pipeline, run_image_edit, to_host
from models.demos.qwen_image_edit.tt.tracker import ALL_GRADUATED, GRADUATED, Tracker

E2E_CORRECTNESS_GATE = "test_image_edit_e2e"
PCC_TARGET = 0.99
STUB_PKGS = {
    "text_encoder": "models.demos.qwen_image_edit_text_encoder._stubs.",
    "vae": "models.tt_dit.pipelines.qwen_image_edit_vae._stubs.",
    "transformer": "models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.",
}
# the non-graduated ports the graduated ones route through (their forwards are scanned too)
GLUE = {
    "text_encoder": ["attention", "encoder_stack", "layer", "token_embed", "v_l_rotary_embedding"],
    "vae": ["mlp", "_resident"],
    "transformer": ["attention", "encoder_stack", "patch_embed", "decoder_head", "_ccl", "_precise"],
}
CHAIN = ["pipeline", "text_encoder", "transformer", "vae"]


def _pcc_rows(a, b):
    """Per-sample PCC between [B, ...] tensors."""
    a = a.double().flatten(1)
    b = b.double().flatten(1)
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)


def _pcc_matrix(a, b):
    a = a.double().flatten(1)
    b = b.double().flatten(1)
    a = (a - a.mean(1, keepdim=True)) / (a - a.mean(1, keepdim=True)).norm(dim=1, keepdim=True)
    b = (b - b.mean(1, keepdim=True)) / (b - b.mean(1, keepdim=True)).norm(dim=1, keepdim=True)
    return a @ b.t()


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_image_edit_e2e(mesh_device):
    B = I.batch_size_from_env()
    print(f"PERF_BATCH_STREAMS={B}", flush=True)
    print(f"inputs: {I.INPUTS_PROVENANCE}", flush=True)
    enc = I.encode_inputs(B)
    golden = load_golden(enc)  # HF float32 reference for exactly these seeds

    tracker = Tracker()
    pipe = build_pipeline(mesh_device, tracker=tracker)
    tracker.reset()  # count the forward only
    steps_run = []
    out = run_image_edit(pipe, enc, use_trace=True, on_step=lambda i, st: steps_run.append(i), observe_host_ops=True)
    image = to_host(out["image"], mesh_device).float()
    latents = to_host(out["latents"], mesh_device).float()
    prompt = to_host(out["prompt_embeds"], mesh_device).float()
    image_latents = to_host(out["image_latents"], mesh_device).float()

    # horizon: the scheduler's full schedule ran (both sides use the example's num_inference_steps)
    n_sched = len(enc.timesteps)
    assert n_sched == I.EXAMPLE_NUM_INFERENCE_STEPS and len(steps_run) == n_sched
    assert golden["step_latents"].shape[0] == n_sched

    # per-stage localisation (reported, the final image is gated)
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline

    gl = golden["image_latents"]
    gl = QwenImageEditPipeline._pack_latents(gl, gl.shape[0], gl.shape[1], gl.shape[3], gl.shape[4])
    for name, tt, ref in (
        ("prompt_embeds", prompt, golden["prompt_embeds"]),
        ("image_latents", image_latents, gl),
        ("final_latents", latents, golden["step_latents"][-1]),
    ):
        p = _pcc_rows(tt, ref.float())
        print(f"stage {name}: min pcc {p.min().item():.6f} mean {p.mean().item():.6f}", flush=True)

    # Gate 1: native ttnn (static) and on device (runtime)
    mods = [importlib.import_module(STUB_PKGS[g] + n) for g, ns in GRADUATED.items() for n in ns]
    mods += [importlib.import_module(STUB_PKGS[g] + n) for g, ns in GLUE.items() for n in ns]
    mods += [importlib.import_module("models.demos.qwen_image_edit.tt." + n) for n in CHAIN]
    violations = gates.gate1_native(mods)
    print(f"gate1 torch compute in forward code: {violations or 'none'}", flush=True)
    v = out["host_ops"]
    print(f"gate1 host ops in the forward: {v['n_host_ops']} {v['host_ops'][:8]}", flush=True)
    # Gate 2: every graduated module on the forward path
    report = tracker.report()
    print(f"gate2 invocations: {report}", flush=True)
    missing = tracker.missing()

    # Gate 3: per-sample image PCC vs the HF golden
    ref_img = golden["image"].float()
    assert image.shape == ref_img.shape, (image.shape, ref_img.shape)
    pcc = _pcc_rows(image, ref_img)
    for i, s in enumerate(enc.seeds):
        print(f"sample {i} seed {s}: image pcc {pcc[i].item():.6f}", flush=True)
    # independence: each output matches its OWN golden best, and the outputs are not all one image
    if B > 1:
        m = _pcc_matrix(image, ref_img)
        own_best = bool((m.argmax(1) == torch.arange(B)).all())
        spread = float((image - image[:1]).abs().amax())
        print(f"independence: own-golden best for every sample {own_best}; max |x_i - x_0| {spread:.4f}")
        assert own_best and spread > 0
    achieved_pcc = float(pcc.min())
    print(f"e2e PCC={achieved_pcc:.6f}", flush=True)
    assert not violations, f"torch compute in the forward: {violations}"
    assert v["on_device"], v["reason"]
    assert not missing, f"graduated modules not invoked: {missing}"
    assert len(report) == len(ALL_GRADUATED) == 25
    assert achieved_pcc >= PCC_TARGET, f"min image PCC {achieved_pcc:.6f} < {PCC_TARGET} over {B} samples"
