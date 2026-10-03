# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERF test for Call 1, image_edit, on the 2x4 mesh: every stage of PIPELINE_STAGES is captured as ONE
device trace (its `<stage>_trace_step`, fixed shape, reading only buffers `<stage>_trace_setup`
uploaded) and replayed on a single command queue. No golden and no PCC here; correctness is
test_e2e_image_edit.py.

Knobs (the perf tool's, read from the environment):
  TT_PERF_BATCH             samples per call (0 / unset = the pipeline's own batch, $TT_PERF_BATCH -> 32)
  TT_PERF_LAYERS            depth cap on every repeated stack (unset = all layers)
  TT_PERF_<STAGE>_LAYERS    per-stage depth (vision_encode, text_encode, denoise; unset = all layers)
  TT_PERF_ISL_TOKENS / TT_PERF_OSL_TOKENS   echoed for the harness; the per-stage inputs are the
                            published example's (tt/inputs.py), so neither changes the work
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

from models.demos.qwen_image_edit.demo.mesh import close_mesh, open_mesh
from models.demos.qwen_image_edit.tt import inputs as I
from models.demos.qwen_image_edit.tt import pipeline as P
from models.experimental.perf_automation.agent.perf_adapter import BATCH_ENV, batch_report_line

PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
PERF_BATCH = int(os.environ.get(BATCH_ENV, "0") or "0")


def _depth(var):
    v = (os.environ.get(var) or "").strip()
    return int(v) if v.isdigit() and int(v) > 0 else None


def _build_kwargs():
    return dict(
        layers=_depth("TT_PERF_LAYERS"),
        vision_encode_layers=_depth("TT_PERF_VISION_ENCODE_LAYERS"),
        text_encode_layers=_depth("TT_PERF_TEXT_ENCODE_LAYERS"),
        denoise_layers=_depth("TT_PERF_DENOISE_LAYERS"),
    )


class _Tee(io.TextIOBase):
    """Writes through to the real stream AND keeps a copy. capfd held back every byte until the end of
    the run, so a multi-hour perf run printed nothing at all and the harness killed it as a wedge."""

    def __init__(self, stream):
        self.stream, self.buf = stream, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()


@contextlib.contextmanager
def _tee_std():
    out, err = _Tee(sys.stdout), _Tee(sys.stderr)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


def test_image_edit_perf():
    from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
    from models.experimental.perf_automation.agent.trace_replay import measure_adapter

    if PERF_BATCH > 0:
        os.environ[BATCH_ENV] = str(PERF_BATCH)  # the stage inputs are built at this batch
    B = I.batch_size_from_env()
    hf = P.load_hf_pipeline()
    tok = hf.tokenizer
    prompt_ids = tok(I.INPUTS_PROVENANCE, return_tensors="pt").input_ids.reshape(-1)
    print(batch_report_line(B), flush=True)
    print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print("PERF_DEPTH=%s STAGES=%s" % (_build_kwargs(), P.PIPELINE_STAGES), flush=True)

    device = open_mesh()
    try:
        with _tee_std() as (tee_out, tee_err):
            adapter = PipelineStageAdapter(
                lambda dev: P.build_pipeline(dev, model=hf, **_build_kwargs()), prompt_ids.tolist(), batch=B
            )
            measure_adapter(adapter, device)
    finally:
        close_mesh(device)

    out, err = tee_out.buf.getvalue(), tee_err.buf.getvalue()
    # a stage that could not prepare or capture is only logged by the adapter: every stage must trace
    bound = [s.name for s in adapter.stages]
    assert bound == P.PIPELINE_STAGES, f"stages bound for trace {bound} != {P.PIPELINE_STAGES}\n{err[-4000:]}"
    missing = [s for s in P.PIPELINE_STAGES if f"TRACE_STAGE_MS[{s}]=" not in out]
    assert not missing, f"stages not captured + replayed as a trace: {missing}\n{err[-4000:]}"
