# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PCC gate for Call 1 (text generation).

Real input (Source-A tokenizer, 32 distinct prompts) -> the SAME chained
pipeline the demo runs (`tt/pipeline.py`) -> real task output, asserted against
the HF golden.

  Gate 3  final-output PCC vs the HF golden >= 0.95, for every one of the 32
          samples against its OWN golden row

The structural gates (Gate 1: native ttnn / sharding, Gate 2: every graduated
module invoked) live in other_tests/test_gate1_gate2.py -- they check the
CODE, not the model's numeric output, and share this file's `device`/`pipe`/
`run` fixtures via the sibling conftest.py.

Run:  ./python_env/bin/python -m pytest \
        models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py -s
"""
from __future__ import annotations

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

PCC_THRESHOLD = 0.95


def test_gate3_e2e_pcc(run, pipe):
    """Gate 3.

    ASSERTED metric: per-step next-token-logit PCC against the HF golden
    evaluated on the SAME prefix the TT pipeline was in at that step, minimum
    over the 32 samples. That is what the port is responsible for: given a
    context, produce the right distribution. The TT path is never fed a
    reference tensor -- every step still consumes the previous TT step's own
    output; only the golden is evaluated on TT's history.

    ALSO REPORTED (not asserted): the free-running comparison against
    `model.generate()`. At the DRAM-forced 7-block depth the argmax is
    genuinely degenerate -- see `test_gate3_report_freerunning_divergence`,
    which measures it -- so free-running sequence agreement scores the
    truncation's conditioning, not the port.
    """
    tt, ref, tf = run["tt"], run["ref"], run["tf"]
    steps = tt["steps"]
    a = tt["step_logits"][:steps]  # (steps, B, vocab)
    B = a.shape[1]
    assert B == pipe.batch, f"ran {B} samples, pipeline says {pipe.batch}"

    per_sample = [P.pcc(a[:, i, :], tf[:, i, :]) for i in range(B)]
    per_step = [P.pcc(a[s], tf[s]) for s in range(steps)]
    step0 = [P.pcc(a[0, i, :], tf[0, i, :]) for i in range(B)]

    # a pipeline that shape-supports B but emits 32 identical rows is WRONG
    distinct = len({tuple(r) for r in tt["new_ids"].tolist()})

    print(f"[gate3] steps={steps} batch={B} distinct_tt_outputs={distinct}")
    print(f"[gate3] same-prefix per-step PCC : {[round(x, 5) for x in per_step]}")
    print(f"[gate3] same-prefix per-sample   : min={min(per_sample):.6f} max={max(per_sample):.6f}")
    print(f"[gate3] step-0 (identical prompt): min={min(step0):.6f} max={max(step0):.6f}")

    fr = [P.pcc(a[s], ref["step_logits"][s]) for s in range(min(steps, ref["steps"]))]
    fr_agree = (tt["new_ids"][:, :steps] == ref["new_ids"][:, :steps]).float().mean().item()
    print(f"[gate3] free-running per-step PCC: {[round(x, 5) for x in fr]}")
    print(f"[gate3] free-running token agreement vs generate(): {fr_agree * 100:.1f}%")

    achieved_pcc = min(per_sample)
    print(f"e2e PCC={achieved_pcc}")
    assert distinct > 1, f"all {B} samples produced identical output -- the batch axis is fake"
    assert achieved_pcc >= PCC_THRESHOLD, (
        f"e2e PCC {achieved_pcc} < {PCC_THRESHOLD} (worst sample "
        f"{per_sample.index(achieved_pcc)}); per-step {per_step}"
    )
