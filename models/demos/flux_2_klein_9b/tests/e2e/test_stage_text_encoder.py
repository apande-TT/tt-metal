# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device tests for the TEXT ENCODER stage of the FLUX.2-klein-9B pipeline.

1. ``test_prompt_embeds_pcc``      -- what the image heads consume, vs the real
   ``Flux2KleinPipeline.encode_prompt``.
2. ``test_text_generation_match``  -- the text->text head, vs the checkpoint's own
   greedy ``generate()`` and its per-step logits.
3. ``test_pinned_step_is_host_op_free`` -- the trace contract: a warmed ``step`` /
   ``decode_step`` fires no host aten op, and re-pinning at another length does
   not corrupt an earlier resident.
4. ``test_prompt_embeds_batch_pcc`` / ``test_text_generation_batch_match`` -- the
   same two heads at BATCH=32: 32 DIFFERENT prompts through ONE stage call / ONE
   decode, each scored against its OWN golden.

Why the batch tests assert per-sample and not in aggregate
----------------------------------------------------------
A hard-coded leading ``1`` does not raise at B=32.  It keeps sample 0 and drops
samples 1..31, and whatever it leaves behind is broadcast back over the batch --
so a flattened PCC against a golden computed the same way would still look
perfect.  The batch gates therefore assert THREE things: the output shape carries
B, sample i correlates with sample i's OWN golden, and the B rows are actually
DIFFERENT from each other.  Only the third one catches a fake batch axis.

Distinctness has to be measured where the samples are allowed to differ.  Two
things get in the way of a naive "no two rows correlate above 1 - eps":

* the shared chat-template BOS row of ``prompt_embeds`` carries a norm ~100x
  every other row, so on the FULL flattened tensor even HF's own 32 goldens
  correlate at 0.99977 pairwise.  The padded tail does not have that row: HF's
  own worst pair there is 0.962, which is where a fake batch axis (1.000) stands
  out by a mile.  So the tail is what the gate asserts on, and the full-tensor
  number is printed next to HF's for context.
* ``reference.pcc`` accumulates in fp32; over 1.5 M elements that is worth a few
  times 1e-4, enough to report a correlation ABOVE 1 and reject the HF golden
  itself.  Every batch correlation here therefore goes through ``_pcc64``.

Every test runs on a function-scoped 1x8 mesh with ``FabricConfig.FABRIC_1D``,
opened by the repo's ``mesh_device`` fixture with the same ``device_params`` the
bring-up's sharded PCC tests used, so the device the graduated stubs are composed
on is the device they graduated on.

HF appears here only to source weights and to compute the goldens (the
``_hf_reference_*`` helpers).  All stage maths runs in ttnn.
"""

from __future__ import annotations

import gc

import pytest
import torch

import ttnn
from models.demos.flux_2_klein_9b import reference
from models.demos.flux_2_klein_9b.tt.stubs import Ledger
from models.demos.flux_2_klein_9b.tt.text_encoder import Flux2PromptEmbedStage, Qwen3CausalLmStage, to_host
from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

PROMPT = "A photorealistic close-up of a red panda eating bamboo in soft morning light."
MAX_SEQUENCE_LENGTH = 128
MAX_NEW_TOKENS = 32

#: BATCH=32 INDEPENDENT samples -- 32 DIFFERENT prompts stacked on the leading axis
#: and run as ONE program per layer.  `reference.batch_prompts` /
#: `reference.batch_text_prompts` are the canonical 32 both sides draw from, so
#: sample i is always scored against sample i's OWN golden.
BATCH = 32

#: Both batch gates are legitimately long: the B=32 HF golden is B forward passes of a
#: 9 B model per step on CPU, which alone is several minutes, and `pytest.ini` caps a
#: test at 300 s.  Declared here rather than left to a `--timeout=` on the command
#: line, so the file passes under the repo's own invocation.
BATCH_TIMEOUT = 3600

#: `device_params` copied verbatim from the bring-up's sharded PCC tests.
DEVICE_PARAMS = {"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}
MESH = (1, 8)

#: which graduated stubs each head is supposed to route
PROMPT_EMBED_STUBS = {
    "token_embed",
    "rotary_embedding",
    "layer",
    "r_m_s_norm",
    "attention",
    "mlp",
    "m_l_p",
    "decoder_layer",
}
CAUSAL_LM_STUBS = {"encoder_stack", "decoder_head"}


# ------------------------------------------------------------------- goldens


def _pcc64(a: torch.Tensor, b: torch.Tensor) -> float:
    """``reference.pcc`` in float64.

    The shared helper accumulates in fp32, and over 1.5 M elements that is worth
    a few times 1e-4 -- enough to report a Pearson correlation slightly ABOVE 1.
    The contract lines below still print ``reference.pcc``; this is the number to
    trust, and both are asserted.
    """
    x = a.detach().to(torch.float64).flatten()
    y = b.detach().to(torch.float64).flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return 1.0 if denom == 0 else float((x @ y) / denom)


@torch.no_grad()
def _hf_reference_prompt_embeds():
    """Source A's own answer: ``Flux2KleinPipeline.encode_prompt`` -> ``prompt_embeds``."""
    return reference.hf_prompt_embeds(PROMPT, MAX_SEQUENCE_LENGTH)[0]


@torch.no_grad()
def _hf_reference_generation():
    """The checkpoint's greedy ids and the per-step logits row behind them."""
    ids = reference.hf_text_generation(PROMPT, MAX_NEW_TOKENS)[0].tolist()
    logits, step_ids = reference.hf_text_generation_logits(PROMPT, MAX_NEW_TOKENS)
    return ids, logits, step_ids


# --------------------------------------------------------------------- checks


def _report_ledger(ledger, expected):
    """`routed` is mechanical and exact, so assert it.  `downstream` is
    best-effort (a plain ttnn op between two ports changes tensor identity), so
    print it and move on -- what proves these stubs are load-bearing is the PCC
    below and the orchestrator's ablation gate."""
    print(ledger.table(), flush=True)
    print(f"no_downstream (reported only) = {ledger.no_downstream()}", flush=True)
    routed = set(ledger.routed()["text_encoder"])
    assert routed == expected, f"routed {sorted(routed)} != expected {sorted(expected)}"


def _release(ledger, *objects):
    """Drop every device tensor this test made BEFORE the fixture closes the
    mesh -- freeing a device buffer after its device is gone is not safe."""
    if ledger is not None:
        ledger.release()
    del objects
    gc.collect()


# ---------------------------------------------------------------------- tests


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prompt_embeds_pcc(mesh_device):
    reference.ensure_flux_imports()

    golden = _hf_reference_prompt_embeds().to(torch.float32)
    input_ids, attention_mask = reference.text_inputs(PROMPT, MAX_SEQUENCE_LENGTH)
    assert tuple(input_ids.shape) == (1, MAX_SEQUENCE_LENGTH)
    n_real = int(attention_mask.sum())
    # the pipeline pads to max_sequence_length and keeps the pad positions in
    # prompt_embeds, so this comparison has to cover them
    assert 0 < n_real < MAX_SEQUENCE_LENGTH, "pick a prompt that leaves padding to check"

    ledger = Ledger()
    stage = Flux2PromptEmbedStage(mesh_device, reference.load_text_encoder(), ledger=ledger)
    assert stage.n_layers == max(Flux2PromptEmbedStage.OUT_LAYERS)
    assert len(stage.blocks) == stage.n_layers
    assert len({type(block) for block in stage.blocks}) == 1, "blocks must be same-typed"

    out = stage(input_ids, attention_mask)
    ledger.mark_final(out)
    got = to_host(out, mesh_device).to(torch.float32)
    assert tuple(got.shape) == tuple(golden.shape), f"{tuple(got.shape)} vs {tuple(golden.shape)}"

    stage_pcc = reference.pcc(golden, got)
    real_pcc = reference.pcc(golden[:, :n_real, :], got[:, :n_real, :])
    pad_pcc = reference.pcc(golden[:, n_real:, :], got[:, n_real:, :])
    hidden = golden.shape[-1] // len(Flux2PromptEmbedStage.OUT_LAYERS)
    taps = [
        reference.pcc(golden[..., i * hidden : (i + 1) * hidden], got[..., i * hidden : (i + 1) * hidden])
        for i in range(len(Flux2PromptEmbedStage.OUT_LAYERS))
    ]

    stage_pcc64 = _pcc64(golden, got)
    pad_pcc64 = _pcc64(golden[:, n_real:, :], got[:, n_real:, :])

    _report_ledger(ledger, PROMPT_EMBED_STUBS)
    print(f"tap PCC (9|18|27) = {taps[0]:.6f} | {taps[1]:.6f} | {taps[2]:.6f}", flush=True)
    print(f"real-token PCC={real_pcc:.6f}  padded-position PCC={pad_pcc:.6f}", flush=True)
    print(f"float64: stage={stage_pcc64:.8f} padded={pad_pcc64:.8f}", flush=True)
    print(f"stage PCC={stage_pcc}", flush=True)

    _release(ledger, stage, out)
    for label, value in (
        ("stage", stage_pcc),
        ("stage/f64", stage_pcc64),
        ("padded", pad_pcc),
        ("padded/f64", pad_pcc64),
    ):
        assert value >= 0.98, f"prompt-embed {label} PCC {value} below 0.98"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_text_generation_match(mesh_device):
    reference.ensure_flux_imports()

    golden_ids, golden_logits, golden_step_ids = _hf_reference_generation()
    assert golden_ids == golden_step_ids, "the two reference drivers disagree on the greedy ids"
    prompt_ids = reference.chat_prompt_ids(PROMPT)
    stop_ids = reference.stop_token_ids()
    assert stop_ids == [151645, 151643], f"unexpected stop rule {stop_ids}"

    ledger = Ledger()
    stage = Qwen3CausalLmStage(mesh_device, reference.load_text_encoder(), ledger=ledger)
    assert stage.n_layers == 36
    assert not stage.staged, "constructing a stage must not stage weights"
    # `encoder_stack` is monolithic: its per-layer weight dicts ARE this head's block
    # list, so they exist only once the port is staged.
    stage.build()
    assert stage.staged
    assert len(stage.blocks) == 36
    assert len({type(block) for block in stage.blocks}) == 1, "blocks must be same-typed"

    tt_ids, tt_rows = stage.generate(prompt_ids, MAX_NEW_TOKENS, stop_ids, True)
    tt_logits = torch.cat(tt_rows, dim=0)

    steps = min(len(tt_ids), len(golden_ids))
    match = sum(1 for a, b in zip(tt_ids[:steps], golden_ids[:steps]) if a == b) / max(len(golden_ids), 1)
    logits_pcc = reference.pcc(golden_logits[:steps], tt_logits[:steps])
    logits_pcc64 = _pcc64(golden_logits[:steps], tt_logits[:steps])
    per_step = [reference.pcc(golden_logits[i], tt_logits[i]) for i in range(steps)]
    mean_step_pcc = sum(per_step) / max(len(per_step), 1)

    _report_ledger(ledger, CAUSAL_LM_STUBS)
    print(f"tt ids  = {tt_ids}", flush=True)
    print(f"hf ids  = {golden_ids}", flush=True)
    print(f"decoded = {reference.load_tokenizer().decode(tt_ids)!r}", flush=True)
    print(f"worst per-step logits PCC={min(per_step):.6f} at step {per_step.index(min(per_step))}", flush=True)
    print(f"mean per-step logits PCC={mean_step_pcc}", flush=True)
    print(f"float64: logits={logits_pcc64:.8f}", flush=True)
    print(f"token match={match}", flush=True)
    print(f"logits PCC={logits_pcc}", flush=True)

    _release(ledger, stage)
    assert len(tt_ids) == len(golden_ids), f"TT produced {len(tt_ids)} ids, HF {len(golden_ids)}"
    assert match == 1.0, f"token match {match} != 1.0"
    for label, value in (("logits", logits_pcc), ("logits/f64", logits_pcc64), ("mean per-step", mean_step_pcc)):
        assert value >= 0.98, f"{label} PCC {value} below 0.98"


# ------------------------------------------------------------- BATCH=32 gates


def _worst_pair(x: torch.Tensor) -> tuple[float, tuple[int, int]]:
    """The HIGHEST float64 correlation between two DIFFERENT rows of a batch.

    A batch whose rows are all the same is a fake batch axis and would score a
    perfect per-sample PCC against a golden that is also row-identical, so this
    is the number that decides whether the leading axis is carrying anything.
    """
    n = int(x.shape[0])
    worst, pair = -2.0, (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            value = _pcc64(x[i], x[j])
            if value > worst:
                worst, pair = value, (i, j)
    return worst, pair


def _best_match(row: torch.Tensor, goldens: torch.Tensor) -> int:
    """Which golden sample this row looks most like -- the identity check.

    Per-sample PCC alone cannot tell "row i carries sample i" from "the rows got
    permuted": both score well when every sample resembles every other.  This
    does, and it is what fails loudly if the batch axis is mis-indexed.
    """
    scores = [_pcc64(row, goldens[j]) for j in range(int(goldens.shape[0]))]
    return int(max(range(len(scores)), key=scores.__getitem__))


def _left_padded_batch(prompts, pad_id: int):
    """The batch exactly as ``reference.hf_text_generation_logits_batch`` builds it.

    LEFT padding, because that is what HF's own batched ``generate()`` uses: it
    puts every row's real last token in the SAME column, so one argmax row per
    step serves all B streams and the whole batch shares one decode cursor.
    """
    rows = [reference.chat_prompt_ids(p)[0] for p in prompts]
    width = max(int(row.shape[0]) for row in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, width - row.shape[0] :] = row
        mask[i, width - row.shape[0] :] = 1
    return ids, mask, [int(row.shape[0]) for row in rows]


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("batch", [BATCH])
@pytest.mark.timeout(BATCH_TIMEOUT)
def test_prompt_embeds_batch_pcc(mesh_device, batch):
    """B DISTINCT prompts -> ONE stage call -> ``(B, L, 12288)``, scored per sample.

    The tokenizer pads every row to ``max_sequence_length``, so all B rows are the
    same length and differ only in where their real tokens stop -- which is
    exactly the thing a batch-1 attention bias would get wrong for samples
    1..B-1, silently, because ``prompt_embeds`` keeps the pad positions.
    """
    reference.ensure_flux_imports()

    prompts = reference.batch_prompts(batch)
    assert len(set(prompts)) == batch, "the batch must be B DIFFERENT prompts"

    golden = reference.hf_prompt_embeds(prompts, MAX_SEQUENCE_LENGTH)[0].to(torch.float32)
    input_ids, attention_mask = reference.text_inputs(prompts, MAX_SEQUENCE_LENGTH)
    assert tuple(input_ids.shape) == (batch, MAX_SEQUENCE_LENGTH)
    lengths = attention_mask.sum(dim=1).tolist()
    assert len(set(lengths)) > 1, "pick prompts whose real lengths differ, or the mask is untested"
    assert 0 < min(lengths) and max(lengths) < MAX_SEQUENCE_LENGTH

    ledger = Ledger()
    stage = Flux2PromptEmbedStage(mesh_device, reference.load_text_encoder(), ledger=ledger)

    out = stage(input_ids, attention_mask)  # ONE call, ONE program per layer
    ledger.mark_final(out)
    got = to_host(out, mesh_device).to(torch.float32)
    assert tuple(got.shape) == (batch, MAX_SEQUENCE_LENGTH, stage.out_features), tuple(got.shape)
    assert tuple(got.shape) == tuple(golden.shape), f"{tuple(got.shape)} vs {tuple(golden.shape)}"

    per_sample = [_pcc64(golden[i], got[i]) for i in range(batch)]
    real_pcc = [_pcc64(golden[i, : lengths[i]], got[i, : lengths[i]]) for i in range(batch)]
    # a fake batch axis (leading 1 kept, then broadcast) scores a perfect
    # per-sample PCC against a golden it also flattened -- these two catch it
    tail = max(lengths)  # the padded tail: no shared 100x-norm BOS row to hide behind
    tt_tail_pair, tt_tail_at = _worst_pair(got[:, tail:])
    hf_tail_pair, hf_tail_at = _worst_pair(golden[:, tail:])
    tt_full_pair, _ = _worst_pair(got)
    hf_full_pair, _ = _worst_pair(golden)
    assignment = [_best_match(got[i, tail:], golden[:, tail:]) for i in range(batch)]

    _report_ledger(ledger, PROMPT_EMBED_STUBS)
    print(f"batch={batch} real lengths={lengths}", flush=True)
    for i, (whole, real) in enumerate(zip(per_sample, real_pcc)):
        print(f"  sample {i:2d}  PCC={whole:.6f}  real-token PCC={real:.6f}", flush=True)
    print(f"worst per-sample PCC={min(per_sample):.6f} at sample {per_sample.index(min(per_sample))}", flush=True)
    print(f"mean per-sample PCC={sum(per_sample) / batch:.6f}", flush=True)
    print(
        f"worst pairwise (padded tail): tt={tt_tail_pair:.6f}{tt_tail_at} hf={hf_tail_pair:.6f}{hf_tail_at}", flush=True
    )
    print(f"worst pairwise (whole tensor): tt={tt_full_pair:.6f} hf={hf_full_pair:.6f}", flush=True)
    print(
        f"nearest-golden assignment correct for {sum(1 for i, j in enumerate(assignment) if i == j)}/{batch}",
        flush=True,
    )

    _release(ledger, stage, out)
    for i, value in enumerate(per_sample):
        assert value >= 0.99, f"prompt-embed sample {i} PCC {value} below 0.99"
    for i, value in enumerate(real_pcc):
        assert value >= 0.99, f"prompt-embed sample {i} real-token PCC {value} below 0.99"
    assert tt_tail_pair < 0.99, (
        f"samples {tt_tail_at} correlate at {tt_tail_pair} on the padded tail (HF's own worst "
        f"pair is {hf_tail_pair}) -- the leading axis is not carrying distinct samples"
    )
    assert assignment == list(range(batch)), f"row i does not look most like golden i: {assignment}"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("batch", [BATCH])
@pytest.mark.timeout(BATCH_TIMEOUT)
def test_text_generation_batch_match(mesh_device, batch):
    """B DISTINCT chat prompts -> ONE batched greedy decode, scored per stream.

    Golden is ``reference.hf_text_generation_logits_batch``: the same left
    padding, the same ``generation_config.eos_token_id`` stop rule, the same
    ``max_new_tokens`` cap, and the same "stop when EVERY row has stopped".  The
    B prompts have DIFFERENT lengths, so the per-row bias is what is under test
    as much as the decode loop is.

    What is asserted, and why it is not an exact-token-match
    -------------------------------------------------------
    B*max_new_tokens is ~1000 greedy decisions, and a bf16 trunk cannot be
    required to win every coin flip: this batch reaches eight steps where HF's own
    top-1/top-2 margin is between 0.0 and 0.5 against a logit standard deviation
    near 4, and at three of them the reference's two candidates are BIT-EQUAL, so
    which one "greedy" selects is arbitrary on either side.  The gates are
    therefore (a) per-sample logits PCC over the steps whose input prefix both
    sides shared, (b) per-sample PCC of the one-shot prefill row, which has no
    divergence to hide behind and covers all B rows, and (c) wherever a row leaves
    HF's path it must take HF's OWN runner-up.  The exact-match count is printed.
    """
    reference.ensure_flux_imports()

    prompts = reference.batch_text_prompts(batch)
    assert len(set(prompts)) == batch, "the batch must be B DIFFERENT prompts"
    stop_ids = reference.stop_token_ids()
    pad_id = reference.pad_token_id()
    assert stop_ids == [151645, 151643], f"unexpected stop rule {stop_ids}"

    golden_rows, golden_ids, lengths = reference.hf_text_generation_logits_batch(prompts, MAX_NEW_TOKENS)
    assert len(set(lengths)) > 1, "pick prompts whose lengths differ, or left padding is untested"

    ids, mask, tt_lengths = _left_padded_batch(prompts, pad_id)
    assert tt_lengths == lengths, "TT and HF must left-pad the same batch"

    ledger = Ledger()
    stage = Qwen3CausalLmStage(mesh_device, reference.load_text_encoder(), ledger=ledger)
    stage.build()
    assert stage.n_layers == 36

    # the prefill entry point at B=32: same left-padded batch, one forward, and the
    # row it returns must be every sample's OWN last-real-token logits -- which is
    # exactly the golden's step-0 row
    prefill = (
        to_host(stage.prefill_step(stage.pin_prefill(ids, mask)), mesh_device).reshape(batch, -1).to(torch.float32)
    )
    prefill_pcc = [_pcc64(golden_rows[0][b], prefill[b]) for b in range(batch)]

    tt_ids, tt_rows = stage.generate(ids, MAX_NEW_TOKENS, stop_ids, True, mask, pad_id=pad_id)
    assert isinstance(tt_ids, list) and len(tt_ids) == batch
    assert all(isinstance(row, list) for row in tt_ids), "B>1 must return one id list per row"
    assert tuple(tt_rows[0].shape) == (batch, stage.vocab_size), tuple(tt_rows[0].shape)

    # Per stream: its own tokens, and its own logits over the steps where BOTH sides
    # had the SAME input.  Two things make that window shorter than the whole run:
    # once a row stops, TT freezes it to the pad id while HF keeps feeding it its own
    # argmax; and once a row's greedy path departs from HF's, the two are decoding
    # DIFFERENT sentences and their logits are no longer the same measurement.  So
    # the comparable window is step 0 .. the first differing step INCLUSIVE, which is
    # the last step whose input prefix is shared.  Everything is measured before
    # anything is asserted, so a divergence is REPORTED per sample rather than hidden
    # behind the first failing assert.
    matches, per_sample_logits, shared_steps, divergences = [], [], [], {}
    for b in range(batch):
        gold, got = golden_ids[b], tt_ids[b]
        agree = 0
        while agree < min(len(gold), len(got)) and gold[agree] == got[agree]:
            agree += 1
        matches.append(1.0 if got == gold else 0.0)
        shared = min(agree + 1, len(gold), len(got), len(tt_rows), len(golden_rows))
        shared_steps.append(shared)
        tt_slice = torch.stack([tt_rows[s][b] for s in range(shared)])
        hf_slice = torch.stack([golden_rows[s][b] for s in range(shared)])
        per_sample_logits.append(_pcc64(hf_slice.to(torch.float32), tt_slice.to(torch.float32)))
        if got != gold and agree < min(len(tt_rows), len(golden_rows)):
            # WHERE the two greedy paths part company, judged by the REFERENCE's own
            # logits: `rank` is where HF ranked the token TT chose, and `gap` is HF's
            # own top-1/top-2 margin at that step.  A rank of 0 or 1 with a margin at
            # the reference's own bf16 resolution is a coin flip, not a wrong answer.
            row = golden_rows[agree][b].to(torch.float32)
            order = torch.argsort(row, descending=True)
            rank = int((row > row[got[agree]]).sum())
            divergences[b] = {
                "step": agree,
                "rank": rank,
                "gap": float(row[order[0]] - row[order[1]]),
                "deficit": float(row[order[0]] - row[got[agree]]),
                "std": float(row.std()),
            }

    token_match = sum(matches) / batch
    distinct = {tuple(row) for row in tt_ids}
    hf_distinct = {tuple(row) for row in golden_ids}
    worst_pair, worst_at = _worst_pair(tt_rows[0].to(torch.float32))
    hf_worst_pair, _ = _worst_pair(golden_rows[0].to(torch.float32))

    _report_ledger(ledger, CAUSAL_LM_STUBS)
    tokenizer = reference.load_tokenizer()
    print(f"batch={batch} prompt lengths={lengths} steps={len(tt_rows)}", flush=True)
    for b in range(batch):
        flag = "ok  " if matches[b] else "TIE "
        print(
            f"  sample {b:2d} {flag} tokens={len(tt_ids[b]):2d} sharedsteps={shared_steps[b]:2d} "
            f"logitsPCC={per_sample_logits[b]:.6f} "
            f"| {tokenizer.decode(tt_ids[b], skip_special_tokens=True)!r}",
            flush=True,
        )
        if b in divergences:
            d = divergences[b]
            print(
                f"      parts from HF at step {d['step']}: HF ranked TT's token #{d['rank']}, "
                f"HF's own top1-top2 margin {d['gap']:.4f} and top1-TT margin {d['deficit']:.4f} "
                f"against a logit std of {d['std']:.3f}",
                flush=True,
            )
            print(f"      tt={tt_ids[b]}\n      hf={golden_ids[b]}", flush=True)
    print(f"distinct completions: tt={len(distinct)}/{batch} hf={len(hf_distinct)}/{batch}", flush=True)
    print(
        f"worst pairwise step-0 logits correlation: tt={worst_pair:.6f}{worst_at} " f"hf={hf_worst_pair:.6f}",
        flush=True,
    )
    print(f"worst per-sample logits PCC (shared prefix)={min(per_sample_logits):.6f}", flush=True)
    print(f"worst per-sample PREFILL logits PCC={min(prefill_pcc):.6f}", flush=True)
    print(f"exact token match={token_match} ({int(sum(matches))}/{batch} rows identical to HF)", flush=True)

    _release(ledger, stage)
    # a fake batch axis collapses all B streams onto sample 0's completion; HF's own
    # count is the calibration, so this cannot be satisfied by a lucky threshold
    assert len(distinct) == len(hf_distinct), (
        f"TT produced {len(distinct)} distinct completions, HF {len(hf_distinct)} "
        f"-- the leading axis is not carrying distinct samples"
    )
    assert len(distinct) > 1, "all B streams produced the SAME completion -- fake batch axis"
    for b in range(batch):
        # the NUMERICAL claim: over every step where TT and HF fed the trunk the same
        # thing, sample b's logits are sample b's logits.  This is what a batch bug
        # (a dropped row, a shared bias, a leaked pad column) destroys.
        assert per_sample_logits[b] >= 0.99, (
            f"sample {b} logits PCC {per_sample_logits[b]} below 0.99 over its " f"{shared_steps[b]} shared-input steps"
        )
        assert prefill_pcc[b] >= 0.99, f"sample {b} prefill logits PCC {prefill_pcc[b]} below 0.99"
        # the GREEDY claim: sample b follows HF token for token, and where it does
        # not, it took the REFERENCE'S OWN runner-up.  Asserting an exact match over
        # B*steps greedy decisions would be asserting that bf16 never loses a coin
        # flip: at 32x32 decisions this batch hits eight near-ties, three of which
        # are BIT-EQUAL in HF's own fp32 logits (top1 - top2 == 0.0), so which token
        # "greedy" means there is arbitrary on either side.  Rank <= 1 is the honest
        # contract and still fails loudly for a token the reference thought unlikely.
        if b in divergences:
            assert divergences[b]["rank"] <= 1, (
                f"sample {b} left HF's greedy path at step {divergences[b]['step']} for a "
                f"token HF ranked #{divergences[b]['rank']} -- that is a wrong answer, "
                f"not a tie"
            )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_pinned_step_is_host_op_free(mesh_device):
    """The trace contract for ``tt/pipeline.py``.

    Capped shallow: this test is about where the host work happens, not about
    accuracy, and 4 layers still routes all eight prompt-embed stubs.  (The causal-LM
    half asks for 2 and gets ``depth.MIN_DISCOVERABLE_STACK``, which is fine here --
    only the monolithic port's depth changes, not where its host work happens.)
    """
    reference.ensure_flux_imports()
    text_encoder = reference.load_text_encoder()

    long_ids, long_mask = reference.text_inputs(PROMPT, 64)
    short_ids, short_mask = reference.text_inputs(PROMPT, 32)

    ledger = Ledger()
    embed_stage = Flux2PromptEmbedStage(mesh_device, text_encoder, ledger=ledger, layers=4)
    # Constructing a stage lays out its stack and stages nothing, so no port is bound
    # yet; `build()` is what binds them, and `pin` calls it.  Both halves are asserted
    # because both matter: an eager build would break the walk the profiler does on a
    # fresh pipeline, and a build that skipped a stub would break coverage.
    assert set(ledger.routed()["text_encoder"]) == set(), "construction must stage no port"
    embed_stage.build()
    assert set(ledger.routed()["text_encoder"]) == PROMPT_EMBED_STUBS

    resident = embed_stage.pin(long_ids, long_mask)
    with observe_host_ops() as ops:
        first = embed_stage.step(resident)
    embed_verdict = verdict(list(ops))
    print(f"pin/step verdict (prompt embeds) = {embed_verdict}", flush=True)
    baseline = to_host(first, mesh_device).to(torch.float32)

    # pin() again at a DIFFERENT length; the earlier resident must still be sound
    other = embed_stage.pin(short_ids, short_mask)
    short_out = embed_stage.step(other)
    assert int(short_out.shape[-2]) == 32
    again = to_host(embed_stage.step(resident), mesh_device).to(torch.float32)
    repin_pcc = reference.pcc(baseline, again)
    print(f"re-pin isolation PCC={repin_pcc:.8f}", flush=True)

    lm_ledger = Ledger()
    lm_stage = Qwen3CausalLmStage(mesh_device, text_encoder, ledger=lm_ledger, layers=2)
    assert set(lm_ledger.routed()["text_encoder"]) == set(), "construction must stage no port"
    lm_stage.build()
    assert set(lm_ledger.routed()["text_encoder"]) == CAUSAL_LM_STUBS
    prompt_ids = reference.chat_prompt_ids(PROMPT)
    decode = lm_stage.pin_decode(prompt_ids, None, int(prompt_ids.shape[-1]) + 4)
    with observe_host_ops() as ops:
        logits = lm_stage.decode_step(decode)
    decode_verdict = verdict(list(ops))
    print(f"pin_decode/decode_step verdict = {decode_verdict}", flush=True)
    shape_before = tuple(logits.shape)

    # advance the cursor and prove the traced shape does not move with it
    token = ttnn.argmax(logits, dim=-1)
    lm_stage.advance(decode, token)
    with observe_host_ops() as ops:
        logits2 = lm_stage.decode_step(decode)
    decode_verdict2 = verdict(list(ops))
    print(f"decode_step verdict after advance = {decode_verdict2}", flush=True)
    assert tuple(logits2.shape) == shape_before, "traced shape moved with the cursor"
    assert tuple(decode["ids"].shape) == (1, decode["capacity"]), "id buffer left the pinned capacity"

    # `tt/pipeline.py::_pad_ids` hands pin_* a sequence ALREADY padded out to the
    # traced capacity plus a zero-tailed mask.  The mask -- not the tensor length
    # -- has to pick the last real row, so that form must agree with the unpadded
    # call, and the decode cursor must start on the prompt's last token.
    real = int(prompt_ids.shape[-1])
    cap = real + 4
    padded_ids = torch.zeros(1, cap, dtype=torch.int64)
    padded_ids[0, :real] = prompt_ids.reshape(-1)
    padded_mask = torch.zeros(1, cap, dtype=torch.int64)
    padded_mask[0, :real] = 1
    unpadded_row = to_host(lm_stage.prefill_logits(prompt_ids), mesh_device).reshape(1, -1)
    padded_resident = lm_stage.pin_prefill(padded_ids, padded_mask)
    assert padded_resident["cursor"][0] == real, "prefill cursor ignored the padding mask"
    padded_row = to_host(lm_stage.prefill_step(padded_resident), mesh_device).reshape(1, -1)
    pad_form_pcc = _pcc64(unpadded_row.to(torch.float32), padded_row.to(torch.float32))
    print(f"padded-vs-unpadded prefill PCC={pad_form_pcc:.8f}", flush=True)
    assert (
        lm_stage.pin_decode(padded_ids, padded_mask, cap)["cursor"][0] == real
    ), "decode cursor ignored the padding mask"

    _release(ledger, embed_stage, resident, other, first, short_out)
    _release(lm_ledger, lm_stage, decode, logits, logits2, token, padded_resident)
    assert pad_form_pcc > 0.999, f"padded input form disagrees with unpadded (PCC {pad_form_pcc})"

    assert repin_pcc > 0.9999, f"re-pinning corrupted an earlier resident (PCC {repin_pcc})"
    assert embed_verdict["on_device"], embed_verdict["reason"]
    assert decode_verdict["on_device"], decode_verdict["reason"]
    assert decode_verdict2["on_device"], decode_verdict2["reason"]
