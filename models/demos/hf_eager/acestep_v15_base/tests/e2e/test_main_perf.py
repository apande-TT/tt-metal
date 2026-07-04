# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Performance profile for the ACE-Step v1.5 'main' (generate_audio) TTNN pipeline.

Runs the SAME on-device TTNN chain the PCC gate exercises -- build real inputs,
build AceStepPipelineTT from the HF model, then pipe.generate over a bounded
flow-matching denoising horizon -- but DROPS the HF/torch golden reference and
every PCC / comp_pcc comparison. This is a perf-only profile: it asserts the
pipeline produced output, not that it matches a reference.

The device forward runs IN-PROCESS (never shelled out) so tracy can profile the
TTNN ops, and the profiler is drained model-agnostically (every dispatched op,
by type) so the 12000-marker buffer never overflows.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT

# Bounded, profiler-safe horizon. The "decode loop" here is the flow-matching
# denoising ODE; cap its step count via TT_PERF_MAX_NEW_TOKENS (small default) so
# a representative dispatch-dense pass runs without overflowing tracy's marker
# buffer or stalling the host in ttnn.synchronize_device.
PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
SEED = 1234


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_main_perf(device_params, device):
    torch.manual_seed(SEED)

    # 1) build the pipeline EXACTLY as the demo/PCC test does -- load the HF model
    #    (used only to construct the TT modules' weights), build real inputs, and
    #    instantiate the on-device TT pipeline. No reference generate chain.
    print("[perf] loading HF model (weight source for TT modules)", flush=True)
    hf_model = load_hf_model()
    inputs = build_inputs(seed=SEED)

    print("[perf] building TT pipeline (13 graduated stubs)", flush=True)
    pipe = AceStepPipelineTT(device, hf_model)

    # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap
    #    EVERY ttnn operation (type 'FastOperation') across ttnn + its op submodules,
    #    so the flush counter tracks TOTAL device dispatch for ANY op mix. A curated
    #    op list under-counts (sdpa/eltwise/transpose/reduction slip through) and the
    #    12000-marker buffer overflows on some device, dropping ops -> non-reproducible
    #    device_ms. Wrapping by TYPE never misses an op.
    counter = [0]
    _orig = []

    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            return r

        return inner

    _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
    for _mod in [_m for _m in _mods if _m is not None]:
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":
                _orig.append((_mod, _n, _op))
                setattr(_mod, _n, _draining(_op))

    _fw0 = time.monotonic()
    try:
        print(f"[perf] running TT pipeline BOUNDED infer_steps={PERF_MAX_NEW_TOKENS}", flush=True)
        out = pipe.generate(inputs, infer_steps=PERF_MAX_NEW_TOKENS, seed=SEED)
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))

    assert out is not None  # perf only -- NO PCC
