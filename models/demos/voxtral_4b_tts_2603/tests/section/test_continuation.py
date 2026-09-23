# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device validation for Call 2, `text_continuation` (`tt/continuation.py`).

Drives the REAL input -- `common.build_batch_inputs()`: 32 distinct prompts, each truncated to
exactly 32 real tokens, so the batch is UNPADDED and the causal mask SDPA's `is_causal` implies is
the plain lower-triangular one.

What is checked, at B=32:

* all three routed stubs (`encoder_stack`, `mistral_model`, `decoder_head`) were really INVOKED,
* PCC(TT last-position logits, `hf_model(input_ids).logits[:, -1]`) per sample,
* `generate(horizon=4)` against the reference's OWN greedy loop -- exact token match plus
  per-step per-sample logit PCC. The reference loop is built here out of `hf_model(...)` forwards;
  `hf_model.generate()` is NOT used (the plan forbids HF orchestration as the reference for this
  chain, and the tied text head makes its defaults meaningless on a TTS checkpoint),
* the 32 outputs are pairwise distinct,
* the `layers=` cap clamps to its floor rather than silently building a 1-layer "model", and a
  capped build is still a MODEL -- embedding, final norm and LM head intact -- checked against a
  torch reference truncated to the same depth.

Every PCC is printed on every run, pass or fail.
"""
from __future__ import annotations

import functools

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.continuation import MIN_LAYERS, STUBS, build_continuation, resolve_layer_cap

pytestmark = pytest.mark.timeout(1800)

PCC_TARGET = 0.99
HORIZON = 4
CAP_DEPTH = 3


# --------------------------------------------------------------------------------------
# the HF reference, loaded once per session
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def reference_model():
    """`load_reference_model()` with the TEXT half widened to float32.

    The checkpoint is bfloat16, so widening changes no VALUE -- it only stops the torch golden
    from accumulating in bfloat16, which is the same reason the TT port runs float32 activations
    against bfloat16 weights. Only the text half is widened; the audio halves are untouched.
    Re-tying is not optional: `.float()` rebinds `embed_tokens.weight` to a new tensor and would
    otherwise leave the tied `lm_head` pointing at the old bfloat16 one.
    """
    threads = common.use_all_cpu_threads()
    print(f"[continuation-test] torch threads={threads}", flush=True)
    hf = common.load_reference_model()
    hf.model.float()
    hf.tie_weights()
    if hf.lm_head.weight.dtype != torch.float32:
        hf.lm_head.float()
    tied = hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()
    print(
        f"[continuation-test] reference: layers={len(hf.model.layers)} "
        f"hidden={hf.config.hidden_size} vocab={hf.config.vocab_size} "
        f"lm_head.dtype={hf.lm_head.weight.dtype} tied={tied}",
        flush=True,
    )
    assert tied, "lm_head must stay tied to embed_tokens after the float32 widening"
    hf.eval()
    return hf


def reference_greedy(hf, input_ids, horizon):
    """The reference's OWN greedy loop: `(tokens [B, horizon], [logits [B, vocab]] * horizon)`.

    Full recompute of the grown sequence at every step, with no KV cache -- the same chain the TT
    side runs, because the graduated whole-stack bodies carry no cache either.
    """
    ctx = input_ids
    tokens, step_logits = [], []
    for step in range(horizon):
        with torch.no_grad():
            out = hf(input_ids=ctx, use_cache=False)
        logits = out.logits[:, -1, :].float().clone()
        step_logits.append(logits)
        nxt = logits.argmax(dim=-1, keepdim=True)
        tokens.append(nxt)
        ctx = torch.cat([ctx, nxt], dim=-1)
        print(f"[continuation-test] reference greedy step {step} seq={ctx.shape[1]}", flush=True)
    return torch.cat(tokens, dim=-1), step_logits


def cached_reference_greedy(hf, input_ids, horizon, n_layers):
    key = common.golden_key(
        head="text_continuation",
        input_ids=input_ids,
        horizon=horizon,
        n_layers=n_layers,
    )
    return common.cached_golden(key, lambda: reference_greedy(hf, input_ids, horizon))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def per_sample_pcc(tt: torch.Tensor, ref: torch.Tensor) -> list:
    assert tt.shape == ref.shape, f"{tuple(tt.shape)} vs {tuple(ref.shape)}"
    return [common.pcc(tt[i], ref[i]) for i in range(tt.shape[0])]


def distinct_row_pairs(rows: torch.Tensor) -> set:
    """Indices `(i, j)` whose rows are EQUAL -- the collapse a hardcoded leading 1 produces."""
    n = rows.shape[0]
    return {(i, j) for i in range(n) for j in range(i + 1, n) if torch.equal(rows[i], rows[j])}


def ids_to_device(input_ids, device):
    return ttnn.from_torch(input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# --------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_continuation_b32(device_params, device):
    hf = reference_model()
    input_ids, texts = common.build_batch_inputs()
    batch, seq = input_ids.shape
    print(f"[continuation-test] batch={batch} seq={seq} (read from build_batch_inputs)", flush=True)
    assert batch == common.DEFAULT_BATCH == 32
    assert len({tuple(r.tolist()) for r in input_ids}) == batch, "prompts are not pairwise distinct"

    ref_tokens, ref_step_logits = cached_reference_greedy(hf, input_ids, HORIZON, len(hf.model.layers))

    counter = common.InvocationCounter()
    cont = build_continuation(device, hf, counter=counter)
    assert cont.n_layers == 26, cont.n_layers
    split = cont.split_for(batch)
    print(
        f"[continuation-test] batch split: rows [0:{split}] -> {STUBS[0]}, " f"rows [{split}:{batch}] -> {STUBS[1]}",
        flush=True,
    )
    assert 0 < split < batch

    ids_tt = ids_to_device(input_ids, device)

    # ---- forward_logits -------------------------------------------------------------
    logits_tt = ttnn.to_torch(cont.forward_logits(ids_tt)).float()
    assert tuple(logits_tt.shape) == (batch, cont.vocab_size), tuple(logits_tt.shape)
    prefill_pccs = per_sample_pcc(logits_tt, ref_step_logits[0])
    for i, p in enumerate(prefill_pccs):
        print(f"[continuation-test] forward_logits sample {i:2d} PCC={p:.6f}", flush=True)
    min_prefill = min(prefill_pccs)
    print(f"[continuation-test] forward_logits min per-sample PCC={min_prefill}", flush=True)

    equal_logit_pairs = distinct_row_pairs(logits_tt)
    print(f"[continuation-test] equal TT logit-row pairs: {len(equal_logit_pairs)} (want 0)", flush=True)

    # ---- generate -------------------------------------------------------------------
    tokens_tt_t, step_logits_tt = cont.generate(ids_tt, HORIZON)
    tokens_tt = ttnn.to_torch(tokens_tt_t).to(torch.long)
    assert tuple(tokens_tt.shape) == (batch, HORIZON), tuple(tokens_tt.shape)
    assert len(step_logits_tt) == HORIZON
    print(f"[continuation-test] generate stop_reason={cont.last_stop_reason}", flush=True)

    step_pccs = []
    for t in range(HORIZON):
        got = ttnn.to_torch(step_logits_tt[t]).float()
        pccs = per_sample_pcc(got, ref_step_logits[t])
        step_pccs.append(pccs)
        for i, p in enumerate(pccs):
            print(f"[continuation-test] step {t} sample {i:2d} PCC={p:.6f}", flush=True)
        print(f"[continuation-test] step {t} min per-sample PCC={min(pccs)}", flush=True)

    all_pccs = prefill_pccs + [p for step in step_pccs for p in step]
    achieved = min(all_pccs)

    mismatched = (tokens_tt != ref_tokens).nonzero().tolist()
    match_rate = float((tokens_tt == ref_tokens).float().mean())
    print(f"[continuation-test] greedy token match rate={match_rate:.4f} mismatches={mismatched[:16]}", flush=True)
    print(f"[continuation-test] tt tokens[:4]={tokens_tt[:4].tolist()}", flush=True)
    print(f"[continuation-test] hf tokens[:4]={ref_tokens[:4].tolist()}", flush=True)

    tt_equal_token_pairs = distinct_row_pairs(tokens_tt)
    ref_equal_token_pairs = distinct_row_pairs(ref_tokens)
    print(
        f"[continuation-test] equal token-row pairs: tt={len(tt_equal_token_pairs)} "
        f"hf={len(ref_equal_token_pairs)}",
        flush=True,
    )

    # ---- the eos stop rule really runs ---------------------------------------------
    unused_eos = int(cont.vocab_size - 1)
    while unused_eos in set(tokens_tt.flatten().tolist()):
        unused_eos -= 1
    eos_tokens_t, _ = cont.generate(ids_tt, HORIZON, eos_id=unused_eos)
    eos_tokens = ttnn.to_torch(eos_tokens_t).to(torch.long)
    print(
        f"[continuation-test] eos_id={unused_eos} (never emitted): "
        f"stop_reason={cont.last_stop_reason} shape={tuple(eos_tokens.shape)}",
        flush=True,
    )

    # ---- a padding mask is rejected, not ignored ------------------------------------
    with pytest.raises(NotImplementedError):
        cont.forward_logits(ids_tt, attention_mask=torch.ones_like(input_ids))
    print("[continuation-test] attention_mask is rejected rather than silently ignored", flush=True)

    # ---- gate 2: the stubs really ran ----------------------------------------------
    print(f"[continuation-test] invocations={counter.counts}", flush=True)

    print(f"[continuation-test] detokenized sample 0 prompt: {texts[0][:60]!r}", flush=True)
    tok = common.load_tokenizer()
    print(f"[continuation-test] tt  continuation 0: {tok.decode(tokens_tt[0].tolist())!r}", flush=True)
    print(f"[continuation-test] hf  continuation 0: {tok.decode(ref_tokens[0].tolist())!r}", flush=True)

    print(f"e2e PCC={achieved}")

    # ---- asserts, after everything has been printed --------------------------------
    assert set(counter.counts) == set(STUBS), f"routed stubs {set(STUBS)} vs invoked {set(counter.counts)}"
    assert all(v >= 1 for v in counter.counts.values()), counter.counts
    assert not equal_logit_pairs, (
        f"{len(equal_logit_pairs)} pairs of the {batch} samples produced IDENTICAL logits -- "
        "a leading axis was dropped somewhere"
    )
    assert tt_equal_token_pairs == ref_equal_token_pairs, (
        "TT and HF disagree on which samples collapse to the same continuation: "
        f"tt={sorted(tt_equal_token_pairs)} hf={sorted(ref_equal_token_pairs)}"
    )
    assert min_prefill >= PCC_TARGET, f"forward_logits min per-sample PCC {min_prefill} < {PCC_TARGET}"
    for t, pccs in enumerate(step_pccs):
        assert min(pccs) >= PCC_TARGET, f"step {t} min per-sample PCC {min(pccs)} < {PCC_TARGET}"
    assert torch.equal(tokens_tt, ref_tokens), f"greedy tokens differ at {mismatched[:16]}"
    assert cont.last_stop_reason == "horizon"
    assert torch.equal(eos_tokens, tokens_tt), "the eos comparison changed the emitted tokens"
    assert achieved >= PCC_TARGET


def test_layer_cap_resolution():
    """`layers=N` clamps UP to the floor, and `None` is never read as 0."""
    for requested, expected in (
        (None, 26),
        (0, MIN_LAYERS),
        (1, MIN_LAYERS),
        (2, MIN_LAYERS),
        (3, 3),
        (26, 26),
        (99, 26),
    ):
        got, note = resolve_layer_cap(requested, 26)
        print(f"[continuation-test] resolve_layer_cap({requested!r}, 26) -> {got} :: {note}", flush=True)
        assert got == expected, (requested, got, expected)
        if expected != (26 if requested is None else requested):
            assert "clamp" in note or "cap" in note, note


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_continuation_layer_cap_is_still_a_model(device_params, device):
    """A capped build keeps the embedding, the final norm and the LM head, and matches torch at
    the same depth. `layers=1` is clamped UP to the floor with a printed message."""
    hf = reference_model()
    input_ids, _ = common.build_batch_inputs()
    batch = input_ids.shape[0]

    counter = common.InvocationCounter()
    cont = build_continuation(device, hf, layers=1, counter=counter)
    print(f"[continuation-test] capped build n_layers={cont.n_layers} full={cont.full_layers}", flush=True)
    assert cont.n_layers == MIN_LAYERS == CAP_DEPTH
    assert cont.full_layers == 26
    assert len(hf.model.layers) == 26, "build must restore the reference's own layer list"

    saved = hf.model.layers
    hf.model.layers = torch.nn.ModuleList(list(saved)[:CAP_DEPTH])
    try:
        with torch.no_grad():
            ref_logits = hf(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()
    finally:
        hf.model.layers = saved

    logits_tt = ttnn.to_torch(cont.forward_logits(ids_to_device(input_ids, device))).float()
    pccs = per_sample_pcc(logits_tt, ref_logits)
    for i, p in enumerate(pccs):
        print(f"[continuation-test] capped({CAP_DEPTH}) sample {i:2d} PCC={p:.6f}", flush=True)
    print(f"[continuation-test] capped({CAP_DEPTH}) min per-sample PCC={min(pccs)}", flush=True)
    print(f"[continuation-test] capped invocations={counter.counts}", flush=True)

    assert set(counter.counts) == set(STUBS), counter.counts
    assert not distinct_row_pairs(logits_tt), "capped build collapsed samples"
    assert min(pccs) >= PCC_TARGET
    assert logits_tt.shape == (batch, cont.vocab_size)
