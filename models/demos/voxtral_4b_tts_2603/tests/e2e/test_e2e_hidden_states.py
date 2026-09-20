# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gates 1/2/3 for Call 2 (`hidden_states`) of the Voxtral-4B-TTS-2603 TTNN pipeline.

Gate 1  the routed block is the GRADUATED stub class and its forward runs on
        `ttnn.Tensor` intermediates -- no torch fallback anywhere.
Gate 2  the graduated `model` stub is INVOKED by the real data path, proved by
        the invocation counter; there is no coverage-sweep helper in the package.
Gate 3  e2e PCC = min over the samples of PCC(TT last_hidden_state[b],
        HF last_hidden_state[b]) >= 0.99.

Everything is driven through `tt.pipeline.build_pipeline(...)` /
`pipeline.run_hidden_states(...)`, which is exactly what the demo calls.

Device: this file opens device_id=1 explicitly (see `DEVICE_ID`) rather than
using the repo `device` fixture, because device 0 is owned elsewhere on this box.

    PYTHONPATH=$PWD ./python_env/bin/python -m pytest \\
        models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_hidden_states.py -s
"""
from __future__ import annotations

import ast
import os
import pathlib

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt import hidden_states as hs
from models.demos.voxtral_4b_tts_2603.tt import pipeline as pipeline_mod

# Each test here builds or drives a 26-layer 4B model on a device, with an HF golden on CPU next to
# it. The repo-wide 300s default in pytest.ini is a generic CI guard sized for unit tests: this
# module measured 99s idle and timed out at 300s on a loaded box, i.e. the default is a flake, not
# a budget. A bound is still enforced -- a genuine hang fails here instead of hanging the gate.
pytestmark = pytest.mark.timeout(1200)

DEVICE_ID = int(os.environ.get("VOXTRAL_E2E_DEVICE_ID", "1"))
PCC_TARGET = 0.99
SEQ_LEN = common.DEFAULT_SEQ_LEN

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
STUB_PATH = pathlib.Path(common.BRINGUP_ROOT) / "_stubs" / "model.py"

# A helper that calls stubs purely so a counter ticks would make Gate 2 meaningless; the plan
# forbids one outright, so the test scans for the names such a helper would have.
SWEEP_HELPER_NAMES = ("coverage_step", "invoke_all_stubs", "_touch_all_graduated")


@pytest.fixture(scope="module")
def hf_model():
    """The HF reference, loaded ONCE (13.7 GB of host fp32 weights)."""
    return common.load_reference_model()


@pytest.fixture(scope="module")
def tt_device():
    device = ttnn.open_device(device_id=DEVICE_ID, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def pipeline(tt_device, hf_model):
    """The resident pipeline with only the hidden_states head built."""
    return pipeline_mod.build_pipeline(tt_device, model=hf_model, heads=("hidden_states",))


@pytest.fixture(scope="module")
def forward(pipeline, hf_model):
    """ONE real forward over the full declared batch, shared by every gate below."""
    batch = pipeline.batch
    input_ids, texts = common.build_batch_inputs(batch=batch, seq_len=SEQ_LEN)
    pipeline.counter.reset()
    result = pipeline.run_hidden_states(input_ids)
    golden = hs.hf_reference_hidden_states(hf_model, input_ids)
    return {
        "input_ids": input_ids,
        "texts": texts,
        "result": result,
        "golden": golden,
        "counts": dict(pipeline.counter.counts),
    }


# ------------------------------------------------------------------------------------------
# S1: the input really is a batch of independent samples
# ------------------------------------------------------------------------------------------


def test_batch_inputs_are_independent(pipeline):
    batch = pipeline.batch
    input_ids, texts = common.build_batch_inputs(batch=batch, seq_len=SEQ_LEN)
    print(f"batch driven={batch} seq_len={SEQ_LEN} input_ids={tuple(input_ids.shape)}")
    assert tuple(input_ids.shape) == (batch, SEQ_LEN)
    assert len({tuple(row.tolist()) for row in input_ids}) == batch, "prompt rows are not pairwise distinct"
    assert len(set(texts)) == batch


# ------------------------------------------------------------------------------------------
# GATE 1 -- the routed block is the graduated stub, running on ttnn tensors
# ------------------------------------------------------------------------------------------


def test_gate1_routed_block_is_the_graduated_stub(pipeline, forward):
    stub_module = common.import_stub("model")
    stack = pipeline.hidden_states
    routed = common.unwrap(stack.stub)

    assert isinstance(routed, stub_module.TtModel), f"routed block is {type(routed)}, not the graduated TtModel"
    assert stack.layers is routed.layers, "stack.layers must surface the stub's OWN repeated blocks"
    assert isinstance(stack.layers, list) and stack.layers, "the stack must be a plain non-empty python list"
    assert {type(b) for b in stack.layers} == {stub_module.TtDecoderLayer}, "blocks are not all the graduated class"
    assert stack.n_layers == len(pipeline.reference_model.model.layers), "full build must match the HF depth"

    tt_out = forward["result"]["tt_last_hidden_state"]
    assert isinstance(tt_out, ttnn.Tensor), f"forward output is {type(tt_out)}, not a ttnn.Tensor"

    # Prove the INTERMEDIATES are ttnn tensors too: stage one real input and step a single routed
    # block, checking the staged tensors and the block's output are all on device.
    staged = stack.prepare_inputs(forward["input_ids"])
    embedded = ttnn.embedding(staged["input_ids"], routed.embed_weight, layout=ttnn.TILE_LAYOUT)
    hidden = ttnn.typecast(embedded, stub_module._ACT_DTYPE)
    ttnn.deallocate(embedded)
    block_out = stack.layers[0](
        hidden, position_embeddings=staged["position_embeddings"], attention_mask=staged["attention_mask"]
    )
    norm_out = routed.norm(block_out)
    for name, t in (
        ("input_ids", staged["input_ids"]),
        ("cos", staged["position_embeddings"][0]),
        ("sin", staged["position_embeddings"][1]),
        ("mask", staged["attention_mask"]),
        ("hidden", hidden),
        ("block0_out", block_out),
        ("final_norm_out", norm_out),
    ):
        assert isinstance(t, ttnn.Tensor), f"intermediate {name} is {type(t)}, not a ttnn.Tensor"
    print(f"gate1 intermediates on device: block0_out={tuple(block_out.shape)} norm_out={tuple(norm_out.shape)}")
    for t in (hidden, block_out, norm_out, staged["input_ids"], staged["attention_mask"]):
        ttnn.deallocate(t)
    for t in staged["position_embeddings"]:
        ttnn.deallocate(t)


def test_gate1_stub_forward_has_no_torch_fallback():
    """Every `__call__` in the graduated stub must be pure ttnn (this is what keeps torch_ops 0)."""
    tree = ast.parse(STUB_PATH.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__call__":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Attribute):
                continue
            base = sub.value
            if isinstance(base, ast.Name) and base.id in ("torch", "F"):
                offenders.append(f"{node.name}: torch call {base.id}.{sub.attr}")
            if isinstance(base, ast.Name) and base.id == "ttnn" and sub.attr == "from_torch":
                offenders.append(f"{node.name}: host staging ttnn.from_torch inside the forward")
    assert not offenders, f"torch on the hot path: {offenders}"

    probe = STUB_PATH.with_suffix(".py.native_probe.json")
    if probe.exists():
        import json

        recorded = json.loads(probe.read_text())
        print(f"native_probe: {recorded}")
        assert int(recorded.get("torch_ops", 0)) == 0, "native probe recorded torch ops in the stub forward"


# ------------------------------------------------------------------------------------------
# GATE 2 -- the graduated stub was invoked by the REAL data path
# ------------------------------------------------------------------------------------------


def test_gate2_graduated_model_stub_is_invoked(pipeline, forward):
    counts = forward["counts"]
    print(f"invocation counts after one real forward: {counts}")
    assert counts.get("model", 0) >= 1, f"the graduated `model` stub never ran: {counts}"
    assert counts["model"] == 1, f"one forward must be exactly one stub invocation, got {counts['model']}"

    # The count came from the data path: the counted call is what produced the tensor under test.
    tt_hidden = forward["result"]["last_hidden_state"]
    assert tt_hidden.shape == forward["golden"].shape
    assert torch.isfinite(tt_hidden).all(), "the counted forward did not produce a usable tensor"


def test_gate2_there_is_no_coverage_sweep_helper():
    scanned = 0
    hits = []
    for root in (PACKAGE_ROOT, pathlib.Path(common.BRINGUP_ROOT) / "_stubs"):
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            text = path.read_text()
            for name in SWEEP_HELPER_NAMES:
                if name in text and "SWEEP_HELPER_NAMES" not in text:
                    hits.append(f"{path}: {name}")
    print(f"scanned {scanned} files for coverage-sweep helpers, found {len(hits)}")
    assert not hits, f"a coverage sweep would void Gate 2: {hits}"


# ------------------------------------------------------------------------------------------
# S7 -- batch-drop guard
# ------------------------------------------------------------------------------------------


def _pairwise_min_distance(x: torch.Tensor) -> float:
    flat = x.reshape(x.shape[0], -1).to(torch.float64)
    worst = float("inf")
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            worst = min(worst, float((flat[i] - flat[j]).abs().max()))
    return worst


def test_batch_drop_guard(pipeline, forward):
    tt_hidden = forward["result"]["last_hidden_state"]
    golden = forward["golden"]
    batch = pipeline.batch
    assert tt_hidden.shape[0] == batch and golden.shape[0] == batch

    tt_gap = _pairwise_min_distance(tt_hidden)
    hf_gap = _pairwise_min_distance(golden)
    print(f"pairwise distinctness over {batch} samples: TT min max|diff|={tt_gap:.6f}  HF min max|diff|={hf_gap:.6f}")
    assert hf_gap > 0.0, "the HF goldens are not pairwise distinct -- the batch is not independent"
    assert tt_gap > 0.0, "the TT outputs are pairwise identical -- samples were silently dropped"


# ------------------------------------------------------------------------------------------
# GATE 3 -- e2e PCC
# ------------------------------------------------------------------------------------------


def test_gate3_e2e_pcc(pipeline, forward):
    tt_hidden = forward["result"]["last_hidden_state"]
    golden = forward["golden"]
    batch = pipeline.batch

    per_sample = [common.pcc(tt_hidden[b], golden[b]) for b in range(batch)]

    print(f"\nbatch driven={batch}  seq_len={SEQ_LEN}  layers={pipeline.hidden_states.n_layers}")
    print(f"{'sample':>6} {'PCC':>12} {'TT L2':>12} {'HF L2':>12}  prompt")
    for b, sample_pcc in enumerate(per_sample):
        print(
            f"{b:>6} {sample_pcc:>12.6f} {float(tt_hidden[b].norm()):>12.4f} "
            f"{float(golden[b].norm()):>12.4f}  {forward['texts'][b][:48]!r}"
        )
    print(f"per-sample PCC spread: min={min(per_sample):.6f} max={max(per_sample):.6f}")

    achieved_pcc = min(per_sample)
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_TARGET, f"e2e PCC={achieved_pcc} below target {PCC_TARGET}"


def test_captured_bringup_golden(pipeline):
    """Direct check against the bring-up tool's OWN recorded golden for `model`."""
    cache = common.captured_golden_cache("model")
    input_ids = cache["kwargs"]["input_ids"]
    result = pipeline.run_hidden_states(input_ids)
    achieved = common.pcc(result["last_hidden_state"], cache["golden"])
    if result["tt_last_hidden_state"] is not None:
        ttnn.deallocate(result["tt_last_hidden_state"])
    print(f"captured golden input={tuple(input_ids.shape)} golden={tuple(cache['golden'].shape)} PCC={achieved}")
    assert achieved >= PCC_TARGET, f"captured-golden PCC={achieved} below target {PCC_TARGET}"
