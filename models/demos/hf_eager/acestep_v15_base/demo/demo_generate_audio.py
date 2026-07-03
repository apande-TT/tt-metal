# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable demo for the ACE-Step v1.5 text+lyric+timbre -> audio-latents task.

Loads real inputs (config-derived, exactly the shapes ACE-Step generate_audio
consumes), runs the chained TTNN pipeline over all 13 graduated stubs, and emits
the real task output: the generated acoustic latents (what generate_audio
returns as 'target_latents'). Runs the SAME tt/pipeline.py the e2e test asserts.

Run:
    flock /tmp/tt_ace_device.lock ./python_env/bin/python -m \
        models.demos.hf_eager.acestep_v15_base.demo.demo_generate_audio --infer-steps 2
"""
from __future__ import annotations

import argparse

import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model
from models.demos.hf_eager.acestep_v15_base.tt.invocation_tracker import track_invocations
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT


def main():
    ap = argparse.ArgumentParser(description="ACE-Step v1.5 TTNN audio-latent generation demo")
    ap.add_argument("--infer-steps", type=int, default=4, help="flow-matching ODE steps")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=str, default="acestep_target_latents.pt")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"[demo] loading HF reference model for weights", flush=True)
    hf_model = load_hf_model()
    inputs = build_inputs(seed=args.seed)

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        print(f"[demo] building TTNN pipeline (13 graduated stubs) on {device}", flush=True)
        pipe = AceStepPipelineTT(device, hf_model)
        print(f"[demo] generating (infer_steps={args.infer_steps}, seed={args.seed})", flush=True)
        with track_invocations() as tracker:
            out = pipe.generate(inputs, infer_steps=args.infer_steps, seed=args.seed)
        print(tracker.report(), flush=True)
    finally:
        ttnn.close_device(device)

    tl = out["target_latents"]
    print(f"[demo] Generated latents shape: {tuple(tl.shape)}", flush=True)
    print(
        f"[demo] Stats - min: {tl.min().item():.4f}, max: {tl.max().item():.4f}, "
        f"mean: {tl.mean().item():.4f}, std: {tl.std().item():.4f}",
        flush=True,
    )
    torch.save(tl, args.output)
    print(f"[demo] wrote target_latents -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
