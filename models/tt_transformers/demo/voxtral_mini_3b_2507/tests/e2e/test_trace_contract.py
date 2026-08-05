# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""COMMAND 3 — trace contract + fully-on-device checks for the Voxtral pipeline.

Asserts, on device:

  * `build_pipeline(device, ...)` RETURNS the resident pipeline object (it does
    not run a one-shot generate), and that object carries `PIPELINE_STAGES` plus
    the generic per-stage seam the perf engine binds:
        <stage>_trace_setup(inputs) / <stage>_trace_step() / <stage>_trace_inputs()
    and the AR decode contract decode_prefill(...) / decode_step().
  * `<stage>_trace_inputs()` is ZERO-ARG and returns exactly what
    `<stage>_trace_setup` takes, for every stage.
  * `trace_capture_selftest(device)` captures ONE step of EVERY stage inside
    begin_trace_capture/end_trace_capture, replays it with execute_trace, PCCs the
    replay against the eager output and RELEASES each trace before the next stage.
  * `host_op_selftest()` — the authoritative fully-on-device check — fires ZERO
    host aten ops for EVERY task head, with input encoding and the one-time weight
    build outside the observed region.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.pipeline import (
    DECODE_BATCH,
    ENCODE_C,
    PIPELINE_STAGES,
    build_pipeline,
    host_op_selftest,
    trace_capture_selftest,
)

DEMO_DIR = Path(__file__).resolve().parents[2]
L1_SMALL_SIZE = 24576
# sized from the LARGEST stage (prefill: pinned C x 30 layers).  If a capture
# overflows this, trace_capture_selftest PRINTS the fallback rather than
# silently dropping the stage.
TRACE_REGION_SIZE = int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", 90 * 1024 * 1024))
TRACE_PCC = float(os.environ.get("VOXTRAL_TRACE_PCC", "0.99"))


def _banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_trace_contract(device_params, device):
    _banner(f"BUILD — build_pipeline(device) must RETURN the resident object; stages={PIPELINE_STAGES}")
    # demo kwargs must be accepted and ignored (call-signature compatibility)
    pipe = build_pipeline(device, text="ignored", prompt="ignored", language="en")

    assert pipe is not None, "build_pipeline returned None"
    assert not isinstance(pipe, (torch.Tensor, list, tuple, str)), (
        f"build_pipeline must return the resident pipeline OBJECT, got {type(pipe)} — "
        "a one-shot result exposes none of the trace hooks and the trace engine would skip it"
    )
    assert PIPELINE_STAGES == ["encode", "prefill", "decode"], PIPELINE_STAGES
    assert pipe.B == DECODE_BATCH, f"decode batch must be {DECODE_BATCH}, got {pipe.B}"

    _banner("CONTRACT — per-stage hooks exist and *_trace_inputs() is zero-arg")
    for stage in PIPELINE_STAGES:
        for suffix in ("_trace_setup", "_trace_step", "_trace_inputs"):
            name = f"{stage}{suffix}"
            assert hasattr(pipe, name), f"missing trace hook {name} — the perf test cannot drive '{stage}'"
        fn = getattr(pipe, f"{stage}_trace_inputs")
        required = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert not required, f"{stage}_trace_inputs must be ZERO-ARG, has {[p.name for p in required]}"
        print(f"[contract] {stage}: trace_setup/_step/_inputs present, _trace_inputs is zero-arg")

    for name in ("decode_prefill", "decode_step"):
        assert hasattr(pipe, name), f"AR decode contract missing {name}"
    print("[contract] AR decode contract present: decode_prefill (seeds resident KV) + decode_step")

    _banner("CONTRACT — <stage>_trace_inputs() feeds <stage>_trace_setup() verbatim")
    for stage in PIPELINE_STAGES:
        got = getattr(pipe, f"{stage}_trace_inputs")()
        assert got is not None, f"{stage}_trace_inputs returned None"
        getattr(pipe, f"{stage}_trace_setup")(got)  # must not raise
        print(f"[contract] {stage}_trace_setup({stage}_trace_inputs()) OK -> {type(got).__name__}")

    _banner("PINNED CAPACITIES — variable dim pinned per stage, bound by the config")
    max_pos = pipe.config.text_config.max_position_embeddings
    print(f"[capacity] encode  C={ENCODE_C} (no variable dim: max_source_positions*strides)")
    print(f"[capacity] prefill C={pipe.C}  (sequence axis; config bound max_position_embeddings={max_pos})")
    print(f"[capacity] decode  C={pipe.KV_C} (KV length), batch={pipe.B}")
    assert pipe.C <= max_pos and pipe.KV_C <= max_pos

    _banner("TRACE — capture / execute / PCC / release, one stage at a time")
    ok = trace_capture_selftest(device, pipeline=pipe, pcc_threshold=TRACE_PCC)
    print(f"trace_capture_selftest={ok}")
    assert ok, "trace capture selftest failed — see the per-stage lines above"

    _banner("HOST OPS — authoritative fully-on-device check, EVERY task head")
    v = host_op_selftest(device, pipeline=pipe, max_new_tokens=2)
    for head, hv in v.get("per_head", {}).items():
        print(f"[host_ops] head={head} on_device={hv['on_device']} n_host_ops={hv['n_host_ops']} {hv['reason'][:200]}")
    print(f"host_op_selftest on_device={v['on_device']} n_host_ops={v['n_host_ops']}")
    assert v["on_device"], f"host compute inside the forward: {v['reason']}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-s"]))
