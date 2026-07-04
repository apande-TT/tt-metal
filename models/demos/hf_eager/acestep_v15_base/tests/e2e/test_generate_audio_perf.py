# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Performance test for the ACE-Step v1.5 'generate_audio' TTNN pipeline.

Builds and runs the chained TTNN pipeline EXACTLY as demo/demo_generate_audio.py
does (load_hf_model -> build_inputs -> AceStepPipelineTT -> generate), but BOUNDED
and profiler-safe for tracy:

  * The flow-matching ODE loop (generate's infer_steps) is the decode loop here,
    so it is capped via TT_PERF_MAX_NEW_TOKENS (default 4) to keep the dispatch
    count small.
  * The device profiler is drained every TT_PERF_FLUSH_EVERY ops (default 32) plus
    a final ReadDeviceProfiler, so tracy's 12000-marker buffer never overflows.

The forward runs IN-PROCESS (no subprocess / shell-out) so every TTNN op is visible
to the profiler. Perf only: NO PCC / correctness assertions.
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import build_inputs, load_hf_model
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_SEED = int(os.environ.get("TT_PERF_SEED", "1234"))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_generate_audio_perf(device_params, device):
    # Deterministic build, exactly like the demo.
    torch.manual_seed(PERF_SEED)

    # 1) build the pipeline EXACTLY as demo/demo_generate_audio.py does
    hf_model = load_hf_model()
    inputs = build_inputs(seed=PERF_SEED)
    pipe = AceStepPipelineTT(device, hf_model)

    # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
    #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
    #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
    #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
    counter = [0]
    _orig = []

    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k)
            counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try:
                    ttnn.ReadDeviceProfiler(device)  # 'device' = mesh_device on multi-chip
                except Exception:
                    pass
            return r

        return inner

    _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
    for _mod in [_m for _m in _mods if _m is not None]:
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                _orig.append((_mod, _n, _op))
                setattr(_mod, _n, _draining(_op))
    _fw0 = time.monotonic()
    try:
        # Bounded forward: infer_steps IS the flow-matching ODE loop -> cap it small.
        out = pipe.generate(inputs, infer_steps=PERF_MAX_NEW_TOKENS, seed=PERF_SEED)
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass
    finally:
        for _mod, _n, _f in _orig:
            setattr(_mod, _n, _f)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None  # perf only — NO PCC
    assert out["target_latents"] is not None
