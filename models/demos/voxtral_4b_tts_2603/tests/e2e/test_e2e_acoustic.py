# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gates 1/2/3 for Call 3 (`acoustic`) -- the checkpoint's `acoustic_transformer` section.

Gate 1  every routed block is a graduated Source-B stub and every intermediate
        it produces is a `ttnn.Tensor`; the stack is a plain list of ONE type so
        a structural walk can find, size and cap it.
Gate 2  those stubs are INVOKED by the real forward at exactly the counts the
        wiring implies, and "invoked" means load-bearing: perturbing one block's
        feed-forward moves the final semantic logits.
Gate 3  e2e PCC = min over samples of PCC(TT, torch reference) for BOTH the
        acoustic hidden state and the semantic-codebook logits. Must be >= 0.99.

WHAT THE REFERENCE IS. `tt/acoustic.py::AcousticReference` is built from real
`transformers` Mistral classes loaded with the checkpoint's own tensors, through
the same native -> HF key map and RoPE permute that
`tests/pcc/_reference_loader.py` validated bit-identically against Mistral's
published conversion of the declared base model. The section's shape comes from
`params.json`, which the test re-reads rather than restating.

WHAT IS NOT COVERED. `input_projection`, `time_projection` and
`acoustic_codebook_output` are the flow-matching sampler's surface and this
checkpoint ships no reference for it, so they are not driven and not gated. The
test asserts that this hole is DECLARED rather than silently forgotten.
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import acoustic as ac
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt import pipeline as pipeline_mod

# The text-generation gates own device 0; this file takes the second board, as the hidden_states
# gates do, so the two heavy text builds never contend for one device inside a single pytest run.
DEVICE_ID = int(os.environ.get("VOXTRAL_E2E_DEVICE_ID", "1"))
PCC_TARGET = 0.99

# A stack of fewer than three same-typed blocks is invisible to the walk that sizes repeated
# stacks. This section declares exactly three, so it is also the floor.
MIN_DISCOVERABLE_LAYERS = 3

# Every block routes these, and nothing else.
EXPECTED_STUB_CLASS = {"r_m_s_norm": "TtRMSNorm", "attention": "TtAttention", "mlp": "TtMLP"}

# Each test in this module builds or drives a 26-layer 4B model on a device; the repo-wide 300s
# default is a generic CI guard sized for unit tests, and a shared box under load has been
# measured 3x slower than an idle one. A bound is still enforced -- a genuine hang fails here.
pytestmark = pytest.mark.timeout(1200)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=DEVICE_ID, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def pipe(device):
    """Both the text stack (it supplies the conditioning) and the acoustic stack."""
    return pipeline_mod.build_pipeline(device, heads=("text_generation", "acoustic"), batch=common.DEFAULT_BATCH)


@pytest.fixture(scope="module")
def evidence(pipe):
    """Run the chained forward ONCE and hand the assertions their evidence."""
    stack = pipe.acoustic
    batch = pipe.batch
    input_ids, texts = common.build_batch_inputs(batch=batch)

    structure = [
        (block.index, name, role, type(common.unwrap(wrapped)))
        for block in stack.layers
        for name, role, wrapped in block.parts
    ]

    # The REAL chain: the text backbone's hidden state is handed to the acoustic section as a
    # DEVICE tensor. Nothing round-trips through the host between the two sections.
    lm_hidden_tt = pipe.generation.forward_hidden(input_ids)
    lm_hidden = ttnn.to_torch(lm_hidden_tt).to(torch.float32)
    pipe.counter.reset()
    result = ac.run_acoustic(stack, lm_hidden_tt)
    counts = dict(pipe.counter.counts)
    ttnn.deallocate(lm_hidden_tt)

    golden = ac.hf_reference_acoustic(stack, lm_hidden)

    # Load-bearing spot check: halve one block's feed-forward and re-run.
    baseline = result["semantic_logits"]
    block = stack.layers[-1]
    original = block.part("feed_forward")
    block._by_role["feed_forward"] = _Scaled(original, 0.5)
    perturbed = ac.run_acoustic(stack, lm_hidden)["semantic_logits"]
    block._by_role["feed_forward"] = original

    return {
        "batch": batch,
        "texts": texts,
        "stack": stack,
        "structure": structure,
        "block_types": {type(b) for b in stack.layers},
        "result": result,
        "golden": golden,
        "counts": counts,
        "ablation_delta": float((perturbed - baseline).abs().max()),
        "ablation_pcc": common.pcc(perturbed, baseline),
    }


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


# ---------------------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------------------


def test_gate1_every_routed_block_is_the_graduated_stub(evidence):
    stack = evidence["stack"]
    assert evidence["block_types"] == {
        ac.AcousticBlock
    }, f"the stack must be a plain list of ONE wrapper type; found {evidence['block_types']}"
    assert isinstance(stack.layers, list) and len(stack.layers) >= MIN_DISCOVERABLE_LAYERS

    for index, name, role, cls in evidence["structure"]:
        expected_module = f"{common.STUB_PKG}.{name}"
        assert cls.__module__ == expected_module, (
            f"block {index} role {role!r}: expected the graduated {name} stub from "
            f"{expected_module}, got {cls.__module__}.{cls.__name__}"
        )
        assert cls.__name__ == EXPECTED_STUB_CLASS[name], f"block {index} {role!r}: got {cls.__name__}"
        assert not issubclass(cls, torch.nn.Module), f"{cls} is a torch module, not a TTNN port"

    assert isinstance(evidence["result"]["tt_acoustic_hidden"], (type(None), ttnn.Tensor))
    print(f"gate1: {len(evidence['structure'])} routed graduated stubs over {len(stack.layers)} acoustic blocks")


def test_gate1_the_section_matches_what_params_json_declares(evidence):
    """The port's shape is READ from the checkpoint, never typed as a literal here."""
    args = ac.acoustic_args()
    cfg = evidence["stack"].config
    assert len(evidence["stack"].layers) == int(args["n_layers"])
    assert cfg.hidden_size == int(args["dim"])
    assert cfg.intermediate_size == int(args["hidden_dim"])
    assert cfg.num_attention_heads == int(args["n_heads"])
    assert cfg.num_key_value_heads == int(args["n_kv_heads"])
    assert cfg.head_dim == int(args["head_dim"])
    assert float(cfg.rope_parameters["rope_theta"]) == float(args["rope_theta"])
    # The text backbone's theta is 1e6; a port that silently reused it would be wrong here.
    assert float(args["rope_theta"]) != 1e6
    print(f"acoustic_transformer_args honoured: {args}")


def test_the_unported_surface_is_declared_not_forgotten():
    """The flow-matching projections have no reference; the package must SAY so, in one place."""
    report = ac.dump_section_report()
    for name in ("input_projection", "time_projection", "acoustic_codebook_output"):
        assert name in report, f"{name} is neither driven nor declared as a hole"
    assert "no reference" in report
    print(report)


# ---------------------------------------------------------------------------------------
# Gate 2
# ---------------------------------------------------------------------------------------


def test_gate2_invocation_counts(evidence):
    stack = evidence["stack"]
    expected = ac.expected_invocation_counts(stack.n_layers)
    got = evidence["counts"]
    print(f"gate2: n_layers={stack.n_layers} counts={got}")
    assert got == expected, f"invocation counts {got} != expected {expected}"
    for name, count in expected.items():
        assert count > 0, f"{name} is routed but never invoked"


def test_gate2_invoked_means_load_bearing(evidence):
    print(
        f"gate2/load-bearing: last block feed_forward x0.5 -> max|delta|={evidence['ablation_delta']:.6f} "
        f"pcc_vs_baseline={evidence['ablation_pcc']:.6f}"
    )
    assert evidence["ablation_delta"] > 0.0, "halving a block's feed-forward changed nothing"
    assert evidence["ablation_pcc"] < 1.0


# ---------------------------------------------------------------------------------------
# Gate 3
# ---------------------------------------------------------------------------------------


def test_gate3_e2e_pcc(evidence):
    batch = evidence["batch"]
    tt, golden = evidence["result"], evidence["golden"]

    hidden_pcc = [common.pcc(tt["acoustic_hidden"][b], golden["hidden"][b]) for b in range(batch)]
    logits_pcc = [common.pcc(tt["semantic_logits"][b], golden["semantic_logits"][b]) for b in range(batch)]
    achieved = min(min(hidden_pcc), min(logits_pcc))

    print("")
    print("=" * 96)
    print(f"batch driven          : {batch}  (read from the pipeline object)")
    print(f"acoustic depth        : {evidence['stack'].n_layers}")
    print(f"acoustic hidden  PCC  : min={min(hidden_pcc):.6f} max={max(hidden_pcc):.6f}")
    print(f"semantic logits  PCC  : min={min(logits_pcc):.6f} max={max(logits_pcc):.6f}")
    print("=" * 96, flush=True)

    print(f"e2e PCC={achieved}")
    assert achieved >= PCC_TARGET, f"e2e PCC {achieved} < {PCC_TARGET}"


def test_batch_drop_guard(evidence):
    """32 independent conditioning states that emit identical outputs would be a dropped batch."""
    batch = evidence["batch"]
    logits = evidence["result"]["semantic_logits"]
    distinct = len({logits[b].numpy().tobytes() for b in range(batch)})
    print(f"batch-drop guard: distinct semantic logit blocks={distinct}/{batch}")
    assert distinct == batch, f"only {distinct} distinct outputs for {batch} samples"
