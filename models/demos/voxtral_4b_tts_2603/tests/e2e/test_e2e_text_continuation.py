# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2, end to end: real text -> greedy causal-LM continuation, against Source A's golden.

Routes the three graduated modules Call 1 does not: `encoder_stack`, `mistral_model` (byte-
identical aliases, split by batch row so both do a disjoint share of the real work) and
`decoder_head`, which is the LM head. The TTS chain reads its next token from the acoustic
transformer's semantic head and never touches `lm_head`, so this is that head's only real home.

COHERENCE CAVEAT: this is a TTS checkpoint. The backbone emits AUDIO codebook tokens and its tied
TEXT head is effectively untrained, so the continuation is near-uniform garbage even when the load
is bit-correct. That does not weaken the gate -- the gate compares TT against the HF reference on
the SAME input, and garbage-that-matches is a valid parity result. The load itself is verified
STRUCTURALLY by the reference loader (all 386 tensors consumed, nothing left on meta), never by
reading generated text.

IT ALSO MAKES THE GREEDY ARGMAX UNDECIDABLE AT TIMES. An untrained tied head puts the top two
logits on top of each other: the reference's OWN step-1 top1-top2 margin bottoms out near 4e-3,
while one matmul on this device carries ~1.2e-3 relative rounding (measured against float64:
fp32 act x fp32 weight 1.169e-3, x bf16 weight 1.738e-3) and the LM head is a K=3072 x 131072
matmul at the end of 26 layers. So a handful of rows land on a coin-flip. The gate therefore puts
the REFERENCE on the TT chain's own token trajectory (`fed_tokens=tt["tokens"]`) and compares the
per-step logits on identical contexts -- the well-posed question -- and asserts token equality
wherever the reference's own margin clears the measured logit deviation. Nothing is spliced into
the TT side: `run_text_continuation` picks every one of its own tokens with `ttnn.argmax` on
device, exactly as the demo runs it.
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

# A greedy pick is called DECIDABLE when the reference's own top-2 margin clears this many times
# the measured RMS logit deviation. Unlike the acoustic codes -- where the reference's rule can be
# replayed exactly on this pipeline's own x_final, so no bound is needed at all -- an argmax over
# two independently-computed logit vectors has no exact form, and 32 x 4 = 128 comparisons is small
# enough that a 3-sigma band is not swamped by its own tail.
SIGMA = 3.0

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Decode horizon for this call. Neither `generation_config.max_new_tokens` nor a usable
# `max_length` exists on this checkpoint and the tied text head never emits eos, so there is no
# model signal for a length -- 4 steps is chosen for lack of one, and the SAME 4 are applied to
# the golden. The eos id is still read from the config and still breaks the loop if it ever fires.
HORIZON = 4


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
    common.use_all_cpu_threads()
    pipe = pipeline.build_pipeline(device, model=hf_model, heads=("text_continuation",))
    batch = pipe.batch
    input_ids, texts = common.build_batch_inputs(batch=batch)
    eos = common.eos_token_id(hf_model)

    print(f"\nbatch driven (read from the pipeline): {batch}")
    print(f"prompt tokens: {input_ids.shape[1]}   horizon: {HORIZON}   eos id: {eos}")

    tt = pipe.run_text_continuation(input_ids=input_ids, horizon=HORIZON, eos_id=eos)
    base = dict(task="cont", ids=input_ids, horizon=HORIZON, eos=eos, chain=golden.CHAIN_VERSION)
    free = common.cached_golden(
        common.golden_key(arm="free", **base),
        lambda: golden.hf_reference_text_continuation(hf_model, input_ids, HORIZON, eos_id=eos),
    )
    aligned = common.cached_golden(
        common.golden_key(arm="aligned", toks=tt["tokens"], **base),
        lambda: golden.hf_reference_text_continuation(
            hf_model, input_ids, HORIZON, eos_id=eos, fed_tokens=tt["tokens"]
        ),
    )
    return {
        "pipe": pipe,
        "tt": tt,
        "hf": aligned,
        "free": free,
        "batch": batch,
        "input_ids": input_ids,
        "texts": texts,
    }


def test_golden_matches_a_plain_forward(hf_model):
    """The continuation golden is a plain causal-LM chain, not HF orchestration."""
    common.use_all_cpu_threads()
    ids, _ = common.build_batch_inputs(batch=4)
    g = golden.hf_reference_text_continuation(hf_model, ids, horizon=1)
    with torch.no_grad():
        plain = hf_model(input_ids=ids).logits[:, -1].float()
    score = common.pcc(g["step_logits"][0], plain)
    print(f"golden step-0 logits vs a plain forward: PCC={score:.8f}")
    assert score > 0.999999


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
    logits0 = tt["step_logits"][0]
    distinct = {logits0[i].numpy().tobytes() for i in range(batch)}
    print(f"\ndistinct inputs {len(rows)}/{batch}; distinct step-0 logit rows {len(distinct)}/{batch}")
    assert len(distinct) == batch, f"only {len(distinct)} of {batch} logit rows are distinct"


def test_generated_tokens_match_the_reference(evidence):
    """The discrete output: the greedy tokens, compared where the reference's own pick is decidable.

    Step 0 has no trajectory in it at all -- both sides read the same prompt -- so it is asserted
    whole. Later steps are asserted on every row whose reference top-2 margin clears twice the
    measured logit deviation; rows below that are ties the arithmetic cannot decide on either
    side, and they are counted and printed.
    """
    tt, hf, free, batch = evidence["tt"], evidence["hf"], evidence["free"], evidence["batch"]
    steps = min(tt["tokens"].shape[1], hf["tokens"].shape[1])
    tt_tokens, hf_tokens = tt["tokens"][:, :steps], hf["tokens"][:, :steps]

    dev = torch.stack([tt["step_logits"][s] - hf["step_logits"][s] for s in range(steps)], dim=1)
    bound = SIGMA * float(dev.pow(2).mean().sqrt())
    ref = torch.stack([hf["step_logits"][s] for s in range(steps)], dim=1)
    top2 = ref.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    decidable = margin > 2 * bound
    agree = tt_tokens == hf_tokens

    rate = float(agree.float().mean())
    print(f"\nlogit deviation RMS={bound / SIGMA:.3e}   {SIGMA:.0f}-sigma bound={bound:.3e}")
    print(f"reference top-2 margin: min={float(margin.min()):.4e} median={float(margin.median()):.4e}")
    print(
        f"decidable {int(decidable.sum())}/{decidable.numel()} row-steps; "
        f"token agreement over {steps} steps: {rate:.6f} "
        f"({int(agree.sum())}/{agree.numel()})"
    )
    print(f"TT  sample 0: {tt_tokens[0].tolist()}")
    print(f"HF  sample 0: {hf_tokens[0].tolist()}")
    tok = common.load_tokenizer()
    print(f"TT  sample 0 text: {tok.decode(tt_tokens[0].tolist())!r}")

    free_rate = float((tt_tokens == free["tokens"][:, :steps]).float().mean())
    print(f"free-running reference (DIAGNOSTIC, not a gate metric): token agreement={free_rate:.6f}")

    assert bool(agree[:, 0].all()), (
        "step 0 disagrees with the reference on the SAME prompt -- no trajectory exists yet, so "
        f"this is a real error ({int(agree[:, 0].sum())}/{batch} matched)"
    )
    bad = int((~agree & decidable).sum())
    assert bad == 0, (
        f"{bad} token(s) disagree where the reference's own top-2 margin cleared {2 * bound:.3e} "
        f"-- that is a real error, not a tie; chase it with fidelity, never by relaxing this"
    )


def test_gate3_e2e_pcc(evidence):
    """GATE 3: the FINAL output -- the per-step logits -- against the HF golden, all 32 samples.

    The reference is on this pipeline's own token trajectory, so every step's logits are computed
    from the identical context the TT side had: this measures the 26-layer stack and the LM head,
    not how fast two greedy chains separate.
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    steps = min(len(tt["step_logits"]), len(hf["step_logits"]))
    assert torch.equal(
        hf["fed_tokens"], tt["tokens"][:, : hf["fed_tokens"].shape[1]]
    ), "the reference was not put on the TT trajectory"

    per_sample = []
    for i in range(batch):
        per_sample.append(min(common.pcc(tt["step_logits"][s][i], hf["step_logits"][s][i]) for s in range(steps)))
    achieved_pcc = min(per_sample)
    worst = int(torch.tensor(per_sample).argmin())

    for s in range(steps):
        step_min = min(common.pcc(tt["step_logits"][s][i], hf["step_logits"][s][i]) for i in range(batch))
        print(f"step {s}: min per-sample logit PCC = {step_min:.6f}")
    print(
        f"per-sample worst-step PCC: min={achieved_pcc:.6f} "
        f"mean={sum(per_sample) / batch:.6f} max={max(per_sample):.6f} (worst sample {worst})"
    )
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_TARGET, f"Gate 3 FAILED: logit PCC {achieved_pcc:.6f} < {PCC_TARGET} on sample {worst}"
