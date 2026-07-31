# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TRACE CONTRACT test for coqui/XTTS-v2 (COMMAND 3): host-free per-stage capture.

Two checks over the SHARED pipeline object (``tt/pipeline.py`` ``build_pipeline``),
both at the full DECODE_BATCH so the traced shapes are the shapes the pipeline runs:

  1. ``trace_capture_selftest`` — for EVERY stage in ``PIPELINE_STAGES``
     (prefill / decode / vocode, derived from the config: an AR core emitting
     speech), pin the stage's variable (sequence) axis to a fixed capacity C,
     pre-upload the padded input and every shape-dependent constant OUTSIDE the
     trace, then capture ONE step inside begin/end_trace_capture, replay it and
     check the replayed output against the eager reference. Each stage's trace is
     released before the next one is captured (stage traces must not co-reside).

  2. ``host_op_selftest`` — the authoritative fully-on-device check: the model math
     (encoded inputs -> waveform, incl. the prefix embedding and one AR step) runs
     under ``host_op_observer.observe_host_ops`` with input encoding and the
     one-time weight build OUTSIDE the observed region. ttnn ops do not dispatch
     through torch, so a truly on-device forward fires ZERO host aten ops.

Run: ./python_env/bin/python -m pytest models/demos/xtts_v2/tests/e2e/test_trace_contract.py -s
"""
from __future__ import annotations

from models.demos.xtts_v2._selftest_device import close_selftest_device, open_selftest_device
from models.demos.xtts_v2.tt import pipeline as P


def test_trace_contract():
    dev, is_mesh = open_selftest_device(trace=True)
    try:
        pipe = P.build_pipeline(dev, batch=P.DECODE_BATCH)
        print(f"[trace-contract] stages={P.PIPELINE_STAGES} batch={pipe.batch}")

        # the ZERO-ARG per-stage seam the perf engine binds must exist and agree
        for stage in P.PIPELINE_STAGES:
            for hook in (f"{stage}_trace_setup", f"{stage}_trace_step", f"{stage}_trace_inputs"):
                assert callable(getattr(pipe, hook, None)), f"missing hook {hook}"
            inputs = getattr(pipe, f"{stage}_trace_inputs")()
            assert len(inputs) == 4 and int(inputs[0].shape[0]) == pipe.batch, (
                f"{stage}_trace_inputs must return the batched input tuple")
        # AR decode contract
        for hook in ("decode_prefill", "decode_step", "decode_write_inputs"):
            assert callable(getattr(pipe, hook, None)), f"missing AR hook {hook}"

        ok = pipe.trace_capture_selftest(dev)
        print(f"TRACE_CAPTURE_OK={ok}")

        # one host-free AR step through the 2CQ decode contract (device-resident ids)
        pipe.decode_prefill(pipe.decode_trace_inputs())
        nid = pipe.decode_step()
        pipe.decode_write_inputs(nid)
        step_ids = pipe._to_torch(nid).reshape(-1)
        print(f"[trace-contract] decode_step ids (one per stream) = {step_ids.long().tolist()}")
        assert int(step_ids.shape[0]) == pipe.batch

        v = pipe.host_op_selftest()
        print(f"HOST_OP_ON_DEVICE={v['on_device']}")
        assert ok, "a pipeline stage did not capture host-free / matched its reference"
        assert v["on_device"], f"host compute in the forward: {v['host_ops'][:12]}"
        print("[trace-contract] PASSED")
    finally:
        close_selftest_device(dev, is_mesh)


if __name__ == "__main__":
    test_trace_contract()
