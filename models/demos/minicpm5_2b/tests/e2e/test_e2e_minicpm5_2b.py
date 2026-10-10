# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for openbmb/MiniCPM5-2B text generation on TT (tt/pipeline.py, shared with the demo).

Inputs (provenance): the model card's README.md example 1 chat prompt, verbatim, for every row; rows
differ only in the sampling seed BASE_SEED + i (tt/inputs.py), the sampling axis generation_config
exposes (do_sample=True). Batch B comes from $TT_PERF_BATCH (default 32).

Reference: HF LlamaForCausalLM (fp32 compute of the bf16 checkpoint) model.generate() with
generation_config's TopK/TopP warpers, eos ids and the example's max_new_tokens; the random draw is
Gumbel-max with the same seeded noise as TT (tt/reference.py).
"""
from __future__ import annotations

import os

import pytest
import torch

from models.demos.minicpm5_2b.tt import inputs as tt_inputs
from models.demos.minicpm5_2b.tt import reference as tt_reference
from models.demos.minicpm5_2b.tt.pipeline import (
    GRADUATED_STUBS,
    PIPELINE_STAGES,
    _pcc,
    build_pipeline,
    host_op_selftest,
    load_hf_model,
    trace_capture_selftest,
)

# The ONE test optimize re-runs after every change: TT free-running generation vs an independently
# computed model.generate() golden (tokens exact over the whole output + logits PCC + prefill state).
E2E_CORRECTNESS_GATE = "test_generate_matches_hf_generate"

PCC_TARGET = 0.99
BACKBONE_PCC_FLOOR = 0.995
BACKBONE_NORM_TOL = 0.005
# Sampling at temperature 1.0 meets near-ties: measured on this pipeline (B=32 x 128 tokens) the top-50
# logit error vs the fp32 HF reference was median 0.39 / max 1.22, and every TT/HF token flip sat
# within 0.87 of the reference's own choice (1.0 % of tokens). A flip is accepted only inside TIE_GAP
# and only that rarely; a wiring or numeric defect moves logits by far more than this.
TIE_GAP = 1.0
MAX_TIE_RATE = 0.02
OSL_ENV = "TT_PERF_OSL_TOKENS"  # set only by the perf harness, which caps the horizon by design

DEVICE_PARAMS = {"l1_small_size": 24576, "trace_region_size": 200 * 1024 * 1024}

_HF = {}


def _hf_model():
    if "model" not in _HF:
        _HF["model"] = load_hf_model()
    return _HF["model"]


def _horizon():
    """generate()'s stop rule: eos ids or the published example's max_new_tokens (TT and HF alike)."""
    cap = os.environ.get(OSL_ENV)
    return (int(cap), True) if cap else (tt_inputs.EXAMPLE_MAX_NEW_TOKENS, False)


def _generate(device):
    hf = _hf_model()
    pipe = build_pipeline(device, model=hf, batch=tt_inputs.batch_size())
    print(f"PERF_BATCH_STREAMS={pipe.batch}")
    ids, seeds = tt_inputs.batch_inputs(pipe.batch)
    max_new_tokens, capped = _horizon()
    noise = tt_inputs.gumbel_noise(seeds, max_new_tokens, pipe.vocab)
    res = pipe.run_text_generation(ids, noise, max_new_tokens=max_new_tokens, collect_logits=True)
    return pipe, hf, ids, seeds, noise, max_new_tokens, capped, res


def _check_invocations(pipe):
    n = pipe.n_layers
    per_forward = {
        "token_embed": 1,
        "rotary_embedding": 1,
        "encoder_stack": 1,
        "decoder_layer": (n + 1) // 2,
        "layer": n // 2,
        "attention": n,
        "mlp": (n + 1) // 2,
        "m_l_p": n // 2,
        "r_m_s_norm": 2 * n + 1,
        "decoder_head": 1,
    }
    forwards = pipe.invocations["encoder_stack"]
    print(f"Gate2 invocations ({forwards} forwards): {dict(pipe.invocations)}")
    missing = [s for s in GRADUATED_STUBS if pipe.invocations[s] == 0]
    assert not missing, f"Gate 2: graduated stubs never invoked: {missing}"
    for name, k in per_forward.items():
        assert (
            pipe.invocations[name] == k * forwards
        ), f"Gate 2: {name} invoked {pipe.invocations[name]}x, expected {k * forwards}"


def _check_native(pipe):
    """Gate 1: every routed object is the graduated ttnn stub class from _stubs/ (no torch fallback)."""
    routed = [pipe.token_embed, pipe.rotary_embedding, pipe.encoder_stack, pipe.final_norm, pipe.decoder_head]
    for blk in pipe.encoder_stack.stub.layers:
        routed += [blk, blk.stub.input_layernorm, blk.stub.self_attn, blk.stub.post_attention_layernorm, blk.stub.mlp]
    for c in routed:
        mod = type(c.stub).__module__
        assert mod.startswith("models.demos.minicpm5_2b._stubs."), f"Gate 1: {c.name} is {mod}"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_generate_matches_hf_generate(device):
    """THE correctness gate. Free-running TT generation for B seeded samples against an independent
    model.generate() golden, then the whole output teacher-forced:

      1. prefill hidden state: PCC >= 0.995 on every row and |tt|/|ref| within 0.5 %;
      2. independent golden: each row's tokens equal generate()'s until the first step where they
         differ, that step must be a numeric near-tie on the golden's OWN scores (gap < TIE_GAP), and
         TT logits track the golden's over that common prefix (PCC >= 0.99);
      3. whole output (every step of every row, on the TT trajectory): each TT token is generate()'s
         choice from the reference logits unless it is a near-tie, near-ties stay rare
         (<= MAX_TIE_RATE), and the logits PCC >= 0.99 per row -- that minimum is the e2e PCC;
      4. both sides stop on generate()'s rule (eos / max_new_tokens), never on the safety cap.
    """
    pipe, hf, ids, seeds, noise, max_new_tokens, capped, res = _generate(device)
    _check_native(pipe)
    _check_invocations(pipe)
    B = pipe.batch
    golden = tt_reference._hf_reference_text_generation(hf, ids, seeds, noise, max_new_tokens)

    print(f"TT stop_reason={res['stop_reason']} steps={res['steps']}  HF steps={golden['steps']}")
    if capped:
        print(f"{OSL_ENV} is set: the perf harness capped the horizon by design; termination assert skipped")
    else:
        assert res["stop_reason"] in ("eos", "max_new_tokens"), res["stop_reason"]
        assert res["steps"] <= max_new_tokens

    # 1. First-stage state.
    ref_hidden = tt_reference._hf_prefill_hidden(hf, ids)[0]
    hid_pcc = min(_pcc(res["prefill_hidden"][b], ref_hidden) for b in range(B))
    ratios = [float(res["prefill_hidden"][b].norm() / ref_hidden.norm()) for b in range(B)]
    print(f"prefill hidden PCC(min over rows)={hid_pcc:.6f} norm ratio range=[{min(ratios):.5f}, {max(ratios):.5f}]")

    # 2. Independent golden, up to each row's first (near-tie) divergence.
    gold_logits = tt_reference._hf_step_logits(hf, ids, golden["tokens"], golden["lengths"])
    prefix_pcc, divergence = [], []
    for b in range(B):
        n = min(int(golden["lengths"][b]), int(res["lengths"][b]))
        same = res["tokens"][b, :n] == golden["tokens"][b, :n]
        d = n if bool(same.all()) else int((~same).nonzero()[0])
        upto = min(d + 1, n)
        tt_l = torch.stack([res["logits"][t][b] for t in range(upto)])
        prefix_pcc.append(_pcc(tt_l, gold_logits[b][:upto]))
        if d < n:
            gap = tt_reference.tie_gap(
                gold_logits[b][d], noise[d, b], hf, int(golden["tokens"][b, d]), int(res["tokens"][b, d])
            )
            divergence.append((b, d, gap))
        elif int(golden["lengths"][b]) != int(res["lengths"][b]):
            divergence.append((b, n, float("inf")))  # same tokens, different stop: not a tie
    matched = B - len(divergence)
    print(f"rows identical to generate() over the whole output: {matched}/{B}")
    print(
        f"first divergences (row, step, gap on the golden's scores): {[(b, d, round(g, 4)) for b, d, g in divergence]}"
    )

    # 3. Whole output on the TT trajectory.
    tf_logits = tt_reference._hf_step_logits(hf, ids, res["tokens"], res["lengths"])
    row_pcc, ties, total = [], [], 0
    for b in range(B):
        n = int(res["lengths"][b])
        total += n
        tt_l = torch.stack([res["logits"][t][b] for t in range(n)])
        row_pcc.append(_pcc(tt_l, tf_logits[b]))
        for t in range(n):
            choice, _ = tt_reference.hf_choice(tf_logits[b][t], noise[t, b], hf)
            tok = int(res["tokens"][b, t])
            if choice != tok:
                ties.append((b, t, tt_reference.tie_gap(tf_logits[b][t], noise[t, b], hf, choice, tok)))
    worst_tie = max([g for _, _, g in ties], default=0.0)
    print(f"whole-output token disagreements: {len(ties)}/{total} (largest gap {worst_tie:.4f})")
    print(f"independent-golden prefix PCC(min over rows)={min(prefix_pcc):.6f}")
    print(f"per-row e2e PCC: {[round(p, 5) for p in row_pcc]}")
    achieved_pcc = min(row_pcc)
    print(f"e2e PCC={achieved_pcc}")

    assert hid_pcc >= BACKBONE_PCC_FLOOR, f"prefill hidden PCC {hid_pcc}"
    assert all(abs(r - 1.0) <= BACKBONE_NORM_TOL for r in ratios), f"prefill hidden norm ratio {ratios}"
    assert all(g < TIE_GAP for _, _, g in divergence), f"rows left generate()'s output on a non-tie: {divergence}"
    assert min(prefix_pcc) >= PCC_TARGET, f"logits vs the independent golden: {prefix_pcc}"
    assert worst_tie < TIE_GAP, f"a TT token is not generate()'s choice and not a near-tie: {ties}"
    assert len(ties) <= MAX_TIE_RATE * total, f"{len(ties)}/{total} near-tie flips"
    assert achieved_pcc >= PCC_TARGET


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_trace_capture_selftest(device):
    pipe = build_pipeline(device, model=_hf_model(), batch=tt_inputs.batch_size())
    print(f"PERF_BATCH_STREAMS={pipe.batch}")
    assert pipe.trace_region_bytes() <= DEVICE_PARAMS["trace_region_size"]
    assert trace_capture_selftest(device, pipeline=pipe), f"a stage of {PIPELINE_STAGES} failed trace capture/replay"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_host_op_selftest(device):
    pipe = build_pipeline(device, model=_hf_model(), batch=tt_inputs.batch_size())
    print(f"PERF_BATCH_STREAMS={pipe.batch}")
    verdict = host_op_selftest(device, pipeline=pipe)
    print(f"host_op_selftest: {verdict['reason']}")
    assert verdict["on_device"], verdict["host_ops"]
    _check_invocations(pipe)
