# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Other structural/consistency checks for this model, moved out of
test_e2e_pipeline.py so that file holds only the PCC gate (Gate 3).

  - reports (does not gate) WHY free-running greedy decode diverges from the
    HF golden
  - S9: batching does not silently drop a sample
  - the demo entry point and the test build the same pipeline

Run:  ./python_env/bin/python -m pytest \
        models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/other_tests/test_other_checks.py -s
"""
from __future__ import annotations

from pathlib import Path

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tests.e2e import make_golden
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

DEMO_DIR = Path(P.__file__).resolve().parents[1]


def test_gate3_report_freerunning_divergence(run, pipe):
    """Measure WHY free-running greedy decode diverges: near-ties, or real error?

    For every sample, compare the HF golden's own top-1/top-2 logit gap where TT
    agrees with it against where TT disagrees. If the disagreements sit where the
    reference itself has no meaningful preference, the divergence is the
    truncated model's degeneracy rather than a port defect. This test REPORTS;
    it only fails if disagreements happen on CONFIDENT rows, which would be a
    genuine defect.
    """
    tt, tf = run["tt"], run["tf"]
    a = tt["step_logits"][0].double()  # step 0: both sides on the identical prompt
    g = tf[0].double()
    B = a.shape[0]

    top2 = g.topk(2, dim=-1)
    gap = (top2.values[:, 0] - top2.values[:, 1]) / g.std(dim=-1)
    agree = a.argmax(-1) == g.argmax(-1)

    print(f"[divergence] step-0 argmax agreement: {agree.float().mean().item() * 100:.1f}%")
    for lbl, m in (("agree", agree), ("disagree", ~agree)):
        if m.any():
            print(
                f"[divergence]   {lbl:9s} n={int(m.sum()):2d}  golden top1-top2 gap "
                f"(sigma): median={gap[m].median().item():.4f} max={gap[m].max().item():.4f}"
            )

    confident_misses = int(((~agree) & (gap > 1.0)).sum())
    print(f"[divergence] disagreements on CONFIDENT rows (gap > 1 sigma): {confident_misses}")
    assert confident_misses == 0, (
        f"{confident_misses} sample(s) disagree with the golden where the golden is confident "
        "(>1 sigma) -- that is a port defect, not truncation degeneracy"
    )


def test_batch_row0_matches_unbatched(pipe):
    """S9: row 0 of the B=N run must equal a B=1 run -- proves no sample is
    silently dropped by a hard-coded leading 1 in a stub's slice bounds."""
    ids = make_golden.build_input_ids(pipe.batch)
    big = P._first_shard(pipe.forward_logits(pipe._ids_to_device(ids), last_only=True))
    one = P._first_shard(pipe.forward_logits(pipe._ids_to_device(ids[:1]), last_only=True))
    p = P.pcc(big.reshape(pipe.batch, -1)[0], one.reshape(1, -1)[0])
    print(f"[batch] PCC(row0 of B={pipe.batch}, B=1 run) = {p:.6f}")
    assert p >= 0.999, "batching changes row 0 -- a stub is dropping samples"


def test_demo_and_test_share_one_pipeline():
    src = (DEMO_DIR / "demo" / "demo_text_generation.py").read_text()
    assert "run_text_generation" in src and "from models.demos" in src
    assert "for layer in self.layers" not in src, "demo re-implements the layer loop"
    assert "_stubs" not in src, "demo builds stubs itself instead of using tt/pipeline.py"
