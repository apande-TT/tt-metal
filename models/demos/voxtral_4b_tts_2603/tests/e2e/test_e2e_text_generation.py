# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gates 1/2/3 for Call 1 (`text_generation`) of `mistralai/Voxtral-4B-TTS-2603`.

    Gate 1  every routed block is the graduated Source-B stub, and every
            intermediate it produces is a `ttnn.Tensor`. No torch fallback.
    Gate 2  the graduated stubs are INVOKED by the real forward, at exactly the
            counts the wiring implies -- and "invoked" means "load-bearing":
            perturbing one split block's contribution moves the final output.
    Gate 3  e2e PCC = min over every sample AND every decode step of
            PCC(TT next-token logits, HF next-token logits) under the SAME
            teacher-forced HF-greedy prefix. Must be >= 0.99.

The batch driven is read off the pipeline object, never typed as a literal.

Everything expensive (the HF reference, the resident stack, the decode passes)
is built once in a module-scoped fixture; the test functions only assert on what
it captured, in a fixed order, so Gate 2's counts come from one clean forward.
"""
from __future__ import annotations

import math

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, generation
from models.demos.voxtral_4b_tts_2603.tt.pipeline import build_pipeline

# Each test here builds or drives a 26-layer 4B model on a device, with an HF golden on CPU next to
# it. The repo-wide 300s default in pytest.ini is a generic CI guard sized for unit tests: this
# module measured 99s idle and timed out at 300s on a loaded box, i.e. the default is a flake, not
# a budget. A bound is still enforced -- a genuine hang fails here instead of hanging the gate.
pytestmark = pytest.mark.timeout(1200)

PCC_TARGET = 0.99

# HF's own probability ratio below which a disagreement stops counting as a near-tie. The flat
# untrained head puts true probability ties around rank 7, so RANK is the wrong yardstick here.
NEAR_TIE_RATIO = 0.5

# The class each graduated stub is expected to instantiate, keyed by component name. Checked
# together with the defining module, so `decoder_layer` cannot pass as `layer`.
EXPECTED_STUB_CLASS = {
    "token_embed": "TtTokenEmbed",
    "rotary_embedding": "TtRotaryEmbedding",
    "r_m_s_norm": "TtRMSNorm",
    "attention": "TtAttention",
    "mlp": "TtMLP",
    "m_l_p": "TtMLP",
    "decoder_layer": "TtDecoderLayer",
    "layer": "TtDecoderLayer",
    "decoder_head": "TtDecoderHead",
}


class _Recorder:
    """Passes a stub's call straight through while recording that its output is a `ttnn.Tensor`.

    Wraps the counter-wrapped stub, so the Gate-2 counters keep counting underneath.
    """

    def __init__(self, label, inner, log):
        self.label = label
        self.inner = inner
        self.log = log

    def __call__(self, *a, **kw):
        out = self.inner(*a, **kw)
        flat = out if isinstance(out, (tuple, list)) else (out,)
        self.log.append((self.label, tuple(isinstance(t, ttnn.Tensor) for t in flat)))
        return out


class _Scaled:
    """Perturbs a stub's contribution. Used only to prove the stub is load-bearing."""

    def __init__(self, inner, factor):
        self.inner = inner
        self.factor = factor

    def __call__(self, *a, **kw):
        out = self.inner(*a, **kw)
        scaled = ttnn.multiply(out, self.factor)
        ttnn.deallocate(out)
        return scaled


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def hf_model():
    # ~13.7 GB of host RAM and a few seconds to load: built once for the whole module.
    model = common.load_reference_model()
    model.eval()
    return model


@pytest.fixture(scope="module")
def pipeline(device, hf_model):
    return build_pipeline(
        device,
        model=hf_model,
        heads=("text_generation",),
        batch=common.DEFAULT_BATCH,
    )


@pytest.fixture(scope="module")
def evidence(pipeline):
    """Run everything once, in order, and hand the assertions their evidence."""
    stack = pipeline.generation
    batch = pipeline.batch
    input_ids, prompt_texts = common.build_batch_inputs(batch=batch, seq_len=common.DEFAULT_SEQ_LEN)
    ev = {"input_ids": input_ids, "prompt_texts": prompt_texts, "batch": batch, "stack": stack}

    # ---- Gate 1: structure, captured before anything is instrumented -------------------
    structure = []
    for block in stack.layers:
        for name, role, wrapped in block.parts:
            inner = common.unwrap(wrapped)
            structure.append((block.index, block.kind, name, role, type(inner)))
    for name, wrapped in (
        ("token_embed", stack.token_embed),
        ("rotary_embedding", stack.rotary),
        ("r_m_s_norm", stack.final_norm),
        ("decoder_head", stack.head),
    ):
        structure.append((-1, "standalone", name, name, type(common.unwrap(wrapped))))
    ev["structure"] = structure
    ev["block_types"] = {type(b) for b in stack.layers}

    # ---- Gate 1 + Gate 2: ONE instrumented forward --------------------------------------
    log = []
    saved_blocks = [(block, list(block.parts)) for block in stack.layers]
    for block in stack.layers:
        for name, role, wrapped in list(block.parts):
            block.set_part(role, _Recorder(f"{block.index}:{name}", wrapped, log))
    saved_top = (stack.token_embed, stack.rotary, stack.final_norm, stack.head)
    stack.token_embed = _Recorder("token_embed", stack.token_embed, log)
    stack.rotary = _Recorder("rotary_embedding", stack.rotary, log)
    stack.final_norm = _Recorder("r_m_s_norm.final", stack.final_norm, log)
    stack.head = _Recorder("decoder_head", stack.head, log)

    stack.counter.reset()
    logits = stack.forward_logits(input_ids)
    ev["gate2_counts"] = dict(stack.counter.counts)
    ev["intermediate_log"] = list(log)
    ev["logits_is_ttnn"] = isinstance(logits, ttnn.Tensor)
    ev["logits_shape"] = tuple(int(d) for d in logits.shape)
    baseline = ttnn.to_torch(logits).to(torch.float32)
    ttnn.deallocate(logits)

    for block, parts in saved_blocks:
        for name, role, wrapped in parts:
            block.set_part(role, wrapped)
    stack.token_embed, stack.rotary, stack.final_norm, stack.head = saved_top
    ev["baseline_logits"] = baseline

    # ---- Gate 2 follow-up: is "invoked" the same as "load-bearing"? ---------------------
    split = next(b for b in stack.layers if b.kind.startswith("split_"))
    original = split.part("feed_forward")
    split.set_part("feed_forward", _Scaled(original, 0.5))
    perturbed_t = ttnn.to_torch(stack.forward_logits(input_ids)).to(torch.float32)
    split.set_part("feed_forward", original)
    ev["ablation"] = {
        "block_index": split.index,
        "kind": split.kind,
        "max_abs_delta": float((perturbed_t - baseline).abs().max()),
        "pcc_vs_baseline": common.pcc(perturbed_t, baseline),
    }

    # ---- Gate 3: the real run (free-running + teacher-forced) ---------------------------
    stack.counter.reset()
    ev["result"] = pipeline.run_text_generation(input_ids=input_ids, teacher_forced=True)
    return ev


# ---------------------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------------------


def test_gate1_every_routed_block_is_the_graduated_stub(evidence):
    """Nothing in the chain is a torch fallback, and nothing it emits leaves the device."""
    assert evidence["block_types"] == {
        generation.VoxtralBlock
    }, f"the stack must be a plain list of ONE wrapper type; found {evidence['block_types']}"

    for index, kind, name, role, cls in evidence["structure"]:
        expected_module = f"{common.STUB_PKG}.{name}"
        assert cls.__module__ == expected_module, (
            f"block {index} ({kind}) role {role!r}: expected the graduated {name} stub from "
            f"{expected_module}, got {cls.__module__}.{cls.__name__}"
        )
        assert (
            cls.__name__ == EXPECTED_STUB_CLASS[name]
        ), f"block {index} ({kind}) role {role!r}: expected {EXPECTED_STUB_CLASS[name]}, got {cls.__name__}"
        assert not issubclass(cls, torch.nn.Module), f"{cls} is a torch module, not a TTNN port"

    assert evidence["intermediate_log"], "no stub invocation was recorded"
    for label, flags in evidence["intermediate_log"]:
        assert all(flags), f"{label} returned a non-ttnn intermediate: {flags}"

    assert evidence["logits_is_ttnn"], "forward_logits must return a ttnn.Tensor"
    batch = evidence["batch"]
    assert evidence["logits_shape"] == (
        batch,
        1,
        evidence["stack"].config.vocab_size,
    ), f"the head must see only the last sequence row; got {evidence['logits_shape']}"
    print(
        f"gate1: {len(evidence['structure'])} routed graduated stubs, "
        f"{len(evidence['intermediate_log'])} ttnn intermediates, logits {evidence['logits_shape']}"
    )


# ---------------------------------------------------------------------------------------
# Gate 2
# ---------------------------------------------------------------------------------------


def test_gate2_invocation_counts(evidence):
    """The counts of ONE full forward, recomputed from the wiring that actually runs.

    At the full depth this is token_embed 1, rotary_embedding 1, decoder_layer 23, layer 1,
    attention 2, r_m_s_norm 5, mlp 1, m_l_p 1, decoder_head 1.

    `e2e_plan.json` writes decoder_layer 24, but the routing it declares in the same object --
    index 0, then indices 4..25 -- is 1 + 22 = 23 blocks. 23 is what the wiring runs; 24 is an
    arithmetic slip in the plan text.
    """
    stack = evidence["stack"]
    expected = generation.expected_invocation_counts(stack.n_layers)
    got = evidence["gate2_counts"]
    print(f"gate2: n_layers={stack.n_layers} counts={got}")
    assert got == expected, f"invocation counts {got} != expected {expected}"
    assert len(expected) == 9, f"Call 1 routes 9 graduated stubs, expected map has {len(expected)}"
    for name, count in expected.items():
        assert count > 0, f"{name} is routed but never invoked"


def test_gate2_invoked_means_load_bearing(evidence):
    """Perturbing one split block's feed-forward must move the final logits."""
    ab = evidence["ablation"]
    print(
        f"gate2/load-bearing: block {ab['block_index']} ({ab['kind']}) feed_forward x0.5 -> "
        f"max|delta|={ab['max_abs_delta']:.6f} pcc_vs_baseline={ab['pcc_vs_baseline']:.6f}"
    )
    assert (
        ab["max_abs_delta"] > 0.0
    ), "halving a split block's feed-forward changed nothing: the stub is invoked but not in the data path"
    assert ab["pcc_vs_baseline"] < 1.0


# ---------------------------------------------------------------------------------------
# Gate 3
# ---------------------------------------------------------------------------------------


def test_gate3_e2e_pcc(evidence):
    """e2e PCC = min over every sample and every decode step, teacher-forced on HF's own prefix."""
    result = evidence["result"]
    ref = result["reference"]
    batch = result["batch"]
    horizon = result["horizon"]

    tt_steps = result["tt_step_logits"]  # [B, H, V]
    hf_steps = ref["step_logits"]  # [B, H, V]
    assert tt_steps.shape == hf_steps.shape, f"{tuple(tt_steps.shape)} vs {tuple(hf_steps.shape)}"
    assert tuple(tt_steps.shape[:2]) == (batch, horizon)

    per_step = [[common.pcc(tt_steps[b, t], hf_steps[b, t]) for t in range(horizon)] for b in range(batch)]
    achieved_pcc = min(min(row) for row in per_step)

    prefill_hidden = [common.pcc(result["tt_prefill_hidden"][b], ref["prefill_hidden"][b]) for b in range(batch)]
    prefill_logits = [common.pcc(result["tt_prefill_logits"][b], ref["prefill_logits"][b]) for b in range(batch)]

    tt_tok, hf_tok = result["tt_step_tokens"], ref["step_tokens"]
    agree = (tt_tok == hf_tok).sum().item()
    total = batch * horizon

    print("")
    print("=" * 96)
    print(f"batch driven                : {batch}  (read from the pipeline object)")
    print(f"decoder depth               : {evidence['stack'].n_layers}")
    print(f"prompt_len / horizon        : {result['prompt_len']} / {horizon}  ({result['horizon_provenance']})")
    print(f"prefill hidden PCC per sample: min={min(prefill_hidden):.6f} max={max(prefill_hidden):.6f}")
    print(f"prefill logits PCC per sample: min={min(prefill_logits):.6f} max={max(prefill_logits):.6f}")
    print(f"per-step logits PCC          : min={achieved_pcc:.6f} max={max(max(r) for r in per_step):.6f}")
    print(f"token agreement (teacher-forced): {agree}/{total} = {agree / total:.4f}")
    print("=" * 96, flush=True)

    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_TARGET, f"e2e PCC {achieved_pcc} < {PCC_TARGET}"


def test_batch_drop_guard(evidence):
    """32 shape-supported samples that emit 32 identical tensors would be a dropped batch."""
    result = evidence["result"]
    batch = result["batch"]

    hidden = result["tt_prefill_hidden"]
    logits = result["tt_step_logits"][:, 0, :]
    distinct_hidden = len({hidden[b].numpy().tobytes() for b in range(batch)})
    distinct_logits = len({logits[b].numpy().tobytes() for b in range(batch)})
    print(f"batch-drop guard: distinct prefill hidden={distinct_hidden}/{batch} logits={distinct_logits}/{batch}")
    assert distinct_hidden == batch, f"only {distinct_hidden} distinct hidden states for {batch} samples"
    assert distinct_logits == batch, f"only {distinct_logits} distinct logit rows for {batch} samples"

    baseline = evidence["baseline_logits"]
    distinct_baseline = len({baseline[b, 0].numpy().tobytes() for b in range(batch)})
    assert distinct_baseline == batch, f"only {distinct_baseline} distinct rows in the Gate-1 forward"


def test_behavioral_free_running_report(evidence):
    """Behavioral evidence, NOT a gate: coherence is never asserted on this checkpoint.

    Disagreements are scored by HF's OWN probability ratio p_HF(tt)/p_HF(hf), computed exactly as
    exp(logit_tt - logit_hf) over HF's step logits -- not by token rank, which a flat distribution
    makes meaningless (true probability ties land around rank 7 here).
    """
    result = evidence["result"]
    ref = result["reference"]
    batch, horizon = result["batch"], result["horizon"]

    tt_tok, hf_tok = result["tt_step_tokens"], ref["step_tokens"]
    hf_steps = ref["step_logits"]
    disagreements, near_ties, worst = 0, 0, 1.0
    for b in range(batch):
        for t in range(horizon):
            if int(tt_tok[b, t]) == int(hf_tok[b, t]):
                continue
            disagreements += 1
            delta = float(hf_steps[b, t, int(tt_tok[b, t])] - hf_steps[b, t, int(hf_tok[b, t])])
            ratio = math.exp(delta)
            worst = min(worst, ratio)
            near_ties += int(ratio >= NEAR_TIE_RATIO)

    prompt_len = result["prompt_len"]
    tt_all = result["generated_ids"][:, prompt_len:]
    hf_all = ref["generated_ids"][:, prompt_len:]
    shared = min(int(tt_all.shape[1]), int(hf_all.shape[1]))
    tt_free, hf_free = tt_all[:, :shared], hf_all[:, :shared]
    free_match = int((tt_free == hf_free).sum())

    print("")
    print("-" * 96)
    print("behavioral evidence (reported, not gated)")
    print(f"  teacher-forced disagreements : {disagreements}/{batch * horizon}")
    print(f"  of which near-ties (>= {NEAR_TIE_RATIO}): {near_ties}/{disagreements} (worst ratio {worst:.4f})")
    print(f"  free-running token match     : {free_match}/{tt_free.numel()}")
    for i in range(min(3, batch)):
        print(f"  sample {i}: TT {result['texts'][i]!r}")
    print("-" * 96, flush=True)

    assert tt_free.shape == hf_free.shape
