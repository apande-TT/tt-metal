# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2, end to end: real text -> the causal LM's teacher-forced continuation, against Source A's golden.

Routes the three graduated modules Call 1 does not: `encoder_stack`, `mistral_model` (byte-
identical aliases, split by batch row so both do a disjoint share of the real work) and
`decoder_head`, which is the LM head. The TTS chain reads its next token from the acoustic
transformer's semantic head and never touches `lm_head`, so this is that head's only real home.

THE TASK IS ONE FORWARD, NOT A FREE-RUNNING DECODE. At every position `s` of the real text the
pipeline returns the model's next-token logits given tokens `0..s` and its greedy pick
(`ttnn.argmax` on device). A free-running greedy decode is not a well-posed task on this
checkpoint: its untrained tied text head emitted eos (id 2, the model's only stop rule) on 0/32
rows in 448 CPU steps and loops on 1-5 tokens from step ~9, so every horizon such a test could
run would be one this package invented, and the test would end on its own cap. Teacher forcing
has no horizon at all: the output is the whole `[B, S, vocab]` tensor, `S` is the longest
unpadded length the 32 prompts support (`common.full_prompt_len`), and every position of every
row is compared.

COHERENCE CAVEAT: this is a TTS checkpoint. The backbone emits AUDIO codebook tokens and its tied
TEXT head is effectively untrained, so the predicted text is near-uniform garbage even when the
load is bit-correct. That does not weaken the gate -- the gate compares TT against the HF reference
on the SAME input, and garbage-that-matches is a valid parity result.

IT ALSO MAKES SOME ARGMAXES UNDECIDABLE. An untrained tied head puts the top two logits on top of
each other: the reference's own top1-top2 margin bottoms out near 4e-3, while one matmul on this
device carries ~1.2e-3 relative rounding and the LM head is a K=3072 x 131072 matmul at the end of
26 layers. So token equality is asserted wherever the reference's own scores separate the TT pick
from its top-1 by more than the measured logit deviation; the ties inside that band are counted and
printed one by one.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from models.demos.voxtral_4b_tts_2603.reference import golden
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

pytestmark = pytest.mark.timeout(3600)

PCC_TARGET = 0.99

# The tie band is 2 x SIGMA x the measured RMS logit deviation: a TT pick that disagrees with the
# reference is a TIE only when the reference scores it within that band of its own top-1. Unlike
# the acoustic codes -- where the reference's rule can be replayed exactly on this pipeline's own
# x_final, so no bound is needed at all -- an argmax over two independently-computed logit vectors
# has no exact form. The band is applied at 2x (6 sigma), so at 32 x S comparisons it is not
# swamped by its own tail.
SIGMA = 3.0

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cpu_threads():
    """All cores for the golden, unless VOXTRAL_TORCH_THREADS caps it (a shared box)."""
    common.use_all_cpu_threads()
    cap = os.environ.get("VOXTRAL_TORCH_THREADS")
    if cap:
        torch.set_num_threads(int(cap))
    return torch.get_num_threads()


def routed_modules(call_name: str) -> set:
    """The graduated modules this call routes, read from the PLAN rather than typed here."""
    with open(os.path.join(PKG_ROOT, "e2e_plan.json")) as f:
        plan = json.load(f)
    for head in plan["task_heads"]:
        if head["name"] == call_name:
            return set(head["graduated_modules_routed"])
    raise KeyError(call_name)


@pytest.fixture(scope="module")
def evidence(device, hf_model):
    _cpu_threads()
    pipe = pipeline.build_pipeline(device, model=hf_model, heads=("text_continuation",))
    batch = pipe.batch
    seq_len = common.full_prompt_len(batch)
    input_ids, texts = common.build_batch_inputs(batch=batch, seq_len=seq_len)

    print(f"\nbatch driven (read from the pipeline): {batch}")
    print(f"tokens per row, every one scored: {seq_len} (the shortest of the {batch} prompts; no padding)")

    tt = pipe.run_text_continuation(input_ids=input_ids)
    hf = common.cached_golden(
        common.golden_key(task="cont_teacher_forced", ids=input_ids, chain=golden.CHAIN_VERSION),
        lambda: golden.hf_reference_text_continuation(hf_model, input_ids),
    )
    return {"pipe": pipe, "tt": tt, "hf": hf, "batch": batch, "input_ids": input_ids, "texts": texts}


def test_gate2_every_call_2_stub_was_invoked(evidence):
    invoked = evidence["pipe"].invoked()
    expected = routed_modules("text_continuation")
    missing = sorted(expected - set(invoked))
    print(f"\ninvoked {len(invoked)} stubs; counts: {dict(sorted(invoked.items()))}")
    assert not missing, f"graduated modules routed to Call 2 but never invoked: {missing}"
    assert all(count >= 1 for count in invoked.values())


def test_batch_is_32_independent_samples(evidence):
    tt, batch = evidence["tt"], evidence["batch"]
    assert batch == common.DEFAULT_BATCH == 32
    rows = {tuple(r.tolist()) for r in evidence["input_ids"]}
    assert len(rows) == batch, "the 32 inputs are not pairwise distinct"
    last = tt["logits"][:, -1]
    distinct = {last[i].numpy().tobytes() for i in range(batch)}
    print(f"\ndistinct inputs {len(rows)}/{batch}; distinct last-position logit rows {len(distinct)}/{batch}")
    assert len(distinct) == batch, f"only {len(distinct)} of {batch} logit rows are distinct"


def test_output_covers_every_position(evidence):
    """WHOLE OUTPUT: one logit row and one pick per (sample, position), on both sides."""
    tt, hf, ids = evidence["tt"], evidence["hf"], evidence["input_ids"]
    batch, seq = int(ids.shape[0]), int(ids.shape[1])
    vocab = int(tt["logits"].shape[-1])
    assert tuple(tt["logits"].shape) == (batch, seq, vocab)
    assert tuple(hf["logits"].shape) == (batch, seq, vocab)
    assert tuple(tt["next_tokens"].shape) == (batch, seq) == tuple(hf["next_tokens"].shape)
    # The on-device argmax is the argmax of the logits the pipeline returned.
    own = tt["logits"].argmax(dim=-1)
    gap = tt["logits"].max(dim=-1).values - tt["logits"].gather(-1, tt["next_tokens"].unsqueeze(-1)).squeeze(-1)
    print(f"\nTT on-device argmax vs its own logits: {int((own == tt['next_tokens']).sum())}/{own.numel()} equal, max gap {float(gap.max()):.3e}")
    assert float(gap.max()) == 0.0, "ttnn.argmax picked a token whose logit is not the row maximum"


def test_generated_tokens_match_the_reference(evidence):
    """The discrete output: every greedy pick, EQUAL to the reference's on the identical context.

    Teacher forcing puts both sides on the same real text, so position `s` is decided from the
    same tokens `0..s` on both -- over the WHOLE output, no window. A disagreement is a real error
    unless the reference ITSELF scores the TT pick within the measured logit-deviation band of its
    own top-1; those ties are counted and printed one by one, never waved through silently.
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    tt_tokens, hf_tokens = tt["next_tokens"], hf["next_tokens"]
    ref = hf["logits"]
    bound = SIGMA * float((tt["logits"] - ref).pow(2).mean().sqrt())
    top2 = ref.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    agree = tt_tokens == hf_tokens
    # A TT pick that the reference ranks nowhere near its top-1 (a corrupted logit on the TT side)
    # must NOT pass just because the reference's own top two happen to be close.
    tt_pick_ref = ref.gather(-1, tt_tokens.unsqueeze(-1)).squeeze(-1)
    pick_gap = top2[..., 0] - tt_pick_ref
    tie_mismatch = ~agree & (pick_gap <= 2 * bound)
    rate = float(agree.float().mean())
    print(f"\nlogit deviation RMS={bound / SIGMA:.3e}   {SIGMA:.0f}-sigma bound={bound:.3e}   tie band={2 * bound:.3e}")
    print(f"reference top-2 margin: min={float(margin.min()):.4e} median={float(margin.median()):.4e}")
    print(
        f"token agreement over {tt_tokens.shape[1]} positions x {batch} rows: {rate:.6f} "
        f"({int(agree.sum())}/{agree.numel()}); near-ties (reference top-2 margin inside band): "
        f"{int((margin <= 2 * bound).sum())}; disagreements accepted as ties: {int(tie_mismatch.sum())}"
    )
    for i, s in tie_mismatch.nonzero().tolist()[:32]:
        print(f"  tie row {i} pos {s}: TT {int(tt_tokens[i, s])} HF {int(hf_tokens[i, s])} gap {float(pick_gap[i, s]):.3e}")
    tok = common.load_tokenizer()
    print(f"TT  sample 0 picks: {tt_tokens[0].tolist()}")
    print(f"HF  sample 0 picks: {hf_tokens[0].tolist()}")
    print(f"TT  sample 0 text: {tok.decode(tt_tokens[0].tolist())!r}")

    bad = ~agree & ~tie_mismatch
    for i, s in bad.nonzero().tolist()[:32]:
        print(
            f"  MISMATCH row {i} pos {s}: TT {int(tt_tokens[i, s])} (ref logit {float(tt_pick_ref[i, s]):.4f}) "
            f"HF {int(hf_tokens[i, s])} (ref logit {float(top2[i, s, 0]):.4f})"
        )
    assert int(bad.sum()) == 0, (
        f"{int(bad.sum())} token(s) disagree where the reference scores the TT pick more than {2 * bound:.3e} "
        "below its own top-1 -- that is a real error, not a tie; chase it with fidelity, never by relaxing this"
    )


def test_gate3_e2e_pcc(evidence):
    """GATE 3: the FINAL output -- the logits at every position -- against the HF golden, all 32 samples.

    Per sample, the WORST position's PCC over the 131072-wide logit row.
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    seq = int(tt["logits"].shape[1])
    table = torch.tensor(
        [[common.pcc(tt["logits"][i, s], hf["logits"][i, s]) for s in range(seq)] for i in range(batch)]
    )
    per_sample = table.min(dim=1).values
    achieved_pcc = float(per_sample.min())
    worst = int(per_sample.argmin())
    for s in range(seq):
        print(f"pos {s}: min per-sample logit PCC = {float(table[:, s].min()):.6f}")
    d = (tt["logits"] - hf["logits"]).abs()
    print(f"logit elements off by > 1.0 (DIAGNOSTIC; PCC over 131072 columns cannot see one): {int((d > 1.0).sum())}")
    print(
        f"per-sample worst-position PCC: min={achieved_pcc:.6f} "
        f"mean={float(per_sample.mean()):.6f} max={float(per_sample.max()):.6f} (worst sample {worst})"
    )
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_TARGET, f"Gate 3 FAILED: logit PCC {achieved_pcc:.6f} < {PCC_TARGET} on sample {worst}"
