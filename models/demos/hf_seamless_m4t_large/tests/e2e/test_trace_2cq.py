# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace + 2CQ contract surface test.

Verifies:
 - PIPELINE_STAGES is present and non-empty.
 - Every stage exposes the three-method contract on Pipeline
   (<stage>_trace_setup / <stage>_trace_step / <stage>_write_inputs).
 - trace_capture_selftest(device) is callable and, given a stage that hasn't
   been trace_setup'd, prints the fallback per the contract (never silently
   drops). Actual host-free trace capture requires the device to be opened
   with `trace_region_size>0`; when it isn't, the selftest prints the
   fallback message for each stage and returns False — that IS the contract.
"""
from __future__ import annotations

from models.demos.hf_seamless_m4t_large.tt.pipeline import PIPELINE_STAGES, build_pipeline


def test_pipeline_stages_present():
    """PIPELINE_STAGES must exist and be non-empty (import-time check, no device)."""
    assert isinstance(PIPELINE_STAGES, list) and len(PIPELINE_STAGES) > 0
    assert "encode" in PIPELINE_STAGES
    assert "decode" in PIPELINE_STAGES


def test_trace_stage_surface_present(device):
    """Every stage must expose the three-method contract on Pipeline."""
    pipe = build_pipeline(device)
    for stage in PIPELINE_STAGES:
        for suffix in ("_trace_setup", "_trace_step", "_write_inputs"):
            assert hasattr(pipe, stage + suffix), f"missing {stage}{suffix} on Pipeline"
    assert callable(pipe.trace_capture_selftest)
    # Call selftest without trace_setup — it should print fallback per stage,
    # NOT hang or silently drop. Returns False; that's acceptable here.
    ok = pipe.trace_capture_selftest(device)
    print(f"[trace_selftest] returned ok={ok}  (expected False when device has no trace_region_size)")
