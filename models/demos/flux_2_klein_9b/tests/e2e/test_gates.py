# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate 1 (static) for the FLUX.2-klein-9B e2e pipeline -- host only, no device.

Three things are checked here, all cheaply and before any 9 B checkpoint is
touched:

1. Source B is intact and the routed set is the graduated set: every component
   with a `.last_good_*` snapshot is still byte-identical to that snapshot, so the
   pipeline is composing the graduated body and not a rewrite.  A sharded body
   counts as native; nothing is downgraded to replication.
2. The pipeline is a pure TTNN forward path: no `model.generate`, no HF submodule
   call, no assignment to a `.forward` attribute, and none of the forbidden torch
   compute ops inside any hot-path function of `tt/`.
3. No coverage-sweep shortcut exists.

Gate 2 (all 43 stubs invoked in a real forward) and Gate 3 (final PCC) need the
device and live in `test_e2e_pipeline.py`.
"""

from __future__ import annotations

import ast
import os

import pytest

from models.demos.flux_2_klein_9b.tt import stubs

TT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tt")

#: The forbidden torch compute ops, verbatim from the TT-only contract.
FORBIDDEN_TORCH = {
    "matmul",
    "mm",
    "bmm",
    "einsum",
    "softmax",
    "log_softmax",
    "layer_norm",
    "rms_norm",
    "batch_norm",
    "group_norm",
    "embedding",
    "embedding_bag",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "scaled_dot_product_attention",
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "argmax",
    "topk",
    "multinomial",
    "dropout",
}

#: `reference.py` is the golden helper: HF calls belong there by construction.
NOT_A_FORWARD_PATH = {"reference.py"}

#: Functions that are setup, not forward: weight staging, buffer pinning, goldens.
SETUP_PREFIXES = (
    "__init__",
    "build",
    "load",
    "pin",
    "hf_",
    "_hf_reference",
    "release",
    "capacity",
    "_depth",
    "_bn_",
    "_unpatchify_helpers",
    "_patchify_permutation",
    "_observable_heads",
    "host_op_selftest",
    "trace_capture_selftest",
)
SETUP_SUFFIXES = ("_trace_setup", "_trace_inputs", "_trace_items", "_stage", "_layers")

#: Receivers whose `.generate(...)` would be HF orchestration rather than our loop.
HF_RECEIVERS = {
    "model",
    "self.model",
    "hf_model",
    "self.hf_model",
    "hf",
    "self.hf",
    "text_encoder",
    "self.text_encoder",
    "hf_text_encoder",
    "self.hf_text_encoder",
    "transformer",
    "self.transformer",
    "vae",
    "self.vae",
    "pipe",
    "self.pipe",
}

#: How `tt/reference.py` is imported; its `hf_*` functions are the GOLDENS.
REFERENCE_ALIASES = {"R", "reference"}

#: Names a coverage sweep would use -- forbidden outright.
COVERAGE_SWEEP_NAMES = {
    "coverage_step",
    "coverage_sweep",
    "invoke_all_stubs",
    "_touch_all_graduated",
    "touch_all_stubs",
    "_invoke_every_stub",
    "exercise_all_stubs",
}


def _tt_sources():
    for name in sorted(os.listdir(TT_DIR)):
        if name.endswith(".py") and name not in NOT_A_FORWARD_PATH:
            yield name, os.path.join(TT_DIR, name)


def _is_setup(fn_name: str) -> bool:
    return fn_name.startswith(SETUP_PREFIXES) or fn_name.endswith(SETUP_SUFFIXES)


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# --------------------------------------------------------------------- gate 1a


def test_source_b_is_intact_and_fully_routed():
    graduated = stubs.all_graduated()
    assert {k: len(v) for k, v in graduated.items()} == {
        "text_encoder": 10,
        "transformer": 18,
        "vae": 15,
    }, graduated
    assert sum(len(v) for v in graduated.values()) == 43


@pytest.mark.parametrize("stage", sorted(stubs.BRINGUP_DIRS))
def test_every_graduated_body_is_the_graduated_body(stage):
    for name in stubs.graduated_components(stage):
        snap = stubs.assert_body_is_graduated(stage, name)
        assert snap.endswith((".last_good_sharded", ".last_good_native"))


@pytest.mark.parametrize("stage", sorted(stubs.BRINGUP_DIRS))
def test_the_batch32_delta_is_declared_and_batch_only(stage):
    """Gate 1, part 1b: the pipeline runs 32 independent samples per call, and the
    graduated bodies wrote B=1 into some of their slice/reshape bounds.  Every such
    bound that had to be re-read off the tensor is DECLARED in
    `tt/batch_patches/<stage>.json`, and each declared edit must normalise back to
    the graduated text -- so the delta is a leading-axis bound and nothing else.

    A hardcoded leading 1 is the reason this is a gate rather than a comment: at
    B=32 it does not raise, it silently keeps sample 0 and drops the other 31.
    """
    declared = stubs.batch_patches(stage)
    graduated = set(stubs.graduated_components(stage))
    assert (
        set(declared) <= graduated
    ), f"batch_patches/{stage}.json names a non-graduated stub: {sorted(set(declared) - graduated)}"
    for name in sorted(declared):
        stubs.assert_patches_are_batch_only(stage, name, FORBIDDEN_TORCH)
        # and the declared edit is the ONLY difference from the graduated snapshot
        stubs.assert_body_is_graduated(stage, name)


@pytest.mark.parametrize("stage", sorted(stubs.BRINGUP_DIRS))
def test_no_stub_still_hardcodes_a_leading_one(stage):
    """No routed stub may still bound its LEADING axis with a literal 1.

    `ttnn.slice(x, [0, ...], [1, ...])` is the shape this takes; at B=1 it is
    correct and at B=32 it silently drops 31 samples, so it is checked statically
    rather than left to a PCC that would still look fine on sample 0.
    """
    offences = []
    for name in stubs.graduated_components(stage):
        path = stubs.stub_path(stage, name)
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _dotted(node.func) != "ttnn.slice":
                continue
            if len(node.args) < 3 or not isinstance(node.args[2], (ast.List, ast.Tuple)):
                continue
            lead = node.args[2].elts[0]
            if isinstance(lead, ast.Constant) and lead.value == 1:
                offences.append(f"{stage}/{name}:{node.lineno}: ttnn.slice end bound starts at literal 1")
    assert not offences, (
        "a graduated stub still bounds its leading axis with 1 -- at B=32 this keeps "
        "sample 0 and drops the rest:\n" + "\n".join(offences)
    )


@pytest.mark.parametrize("stage", sorted(stubs.BRINGUP_DIRS))
def test_graduated_stub_bodies_are_ttnn_not_torch(stage):
    """A sharded body counts as native; what must not appear is torch compute."""
    offences = []
    for name in stubs.graduated_components(stage):
        path = stubs.stub_path(stage, name)
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            head, _, tail = dotted.rpartition(".")
            if head in ("torch", "torch.nn.functional", "F") and tail in FORBIDDEN_TORCH:
                offences.append(f"{stage}/{name}: {dotted}")
    assert not offences, "torch compute inside a graduated stub:\n" + "\n".join(offences)


# --------------------------------------------------------------------- gate 1b


def test_pipeline_has_no_hf_orchestration():
    offences = []
    for name, path in _tt_sources():
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                dotted = _dotted(node.func)
                # `<stage>.generate(...)` is OUR greedy loop; `<hf model>.generate(...)`
                # would be HF orchestration.  `_dotted` returns a bare "generate" when
                # the receiver is itself a call (`self.causal_lm_stage().generate`), which
                # can never be an HF model attribute, so only dotted receivers matter.
                receiver = dotted.rpartition(".")[0]
                if dotted.rpartition(".")[2] == "generate" and receiver in HF_RECEIVERS:
                    offences.append(f"{name}: {dotted}()")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "forward":
                        offences.append(f"{name}: assignment to {_dotted(target)}")
    assert not offences, "HF orchestration in the pipeline:\n" + "\n".join(offences)


def test_pipeline_hot_paths_have_no_torch_compute():
    offences = []
    for name, path in _tt_sources():
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if _is_setup(fn.name):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                dotted = _dotted(node.func)
                head, _, tail = dotted.rpartition(".")
                if head in ("torch", "torch.nn.functional", "F") and tail in FORBIDDEN_TORCH:
                    offences.append(f"{name}::{fn.name}: {dotted}")
    assert not offences, "torch compute in a hot path:\n" + "\n".join(offences)


def test_pipeline_hot_paths_do_not_call_the_golden_helpers():
    """`reference.py` may call HF freely, but a head must never reach a golden."""
    offences = []
    for name, path in _tt_sources():
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if _is_setup(fn.name):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    dotted = _dotted(node.func)
                    head, _, tail = dotted.rpartition(".")
                    if head in REFERENCE_ALIASES and tail.startswith("hf_"):
                        offences.append(f"{name}::{fn.name}: {dotted}()")
    assert not offences, "a golden helper is reachable from a hot path:\n" + "\n".join(offences)


def test_no_coverage_sweep_shortcut():
    offences = []
    for name, path in _tt_sources():
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if fn.name in COVERAGE_SWEEP_NAMES:
                offences.append(f"{name}::{fn.name}")
    assert not offences, "coverage-sweep shortcut present:\n" + "\n".join(offences)


DEMO_DIR = os.path.join(os.path.dirname(TT_DIR), "demo")

#: the head entry points; a demo must call one of these, not re-implement it
HEAD_ENTRIES = {"run_text_to_image", "run_image_edit", "run_text_generation", "run_vae_roundtrip"}


def test_demo_and_tests_share_one_pipeline():
    """The wiring lives ONCE, in `tt/`, and both `demo/` and `tests/e2e/` call it.

    A demo carrying its own copy of the chain drifts from the tested one and ships a
    broken pipeline behind a green test, so this is checked structurally: every demo
    entry point must build through `tt.pipeline.build_pipeline`, must call one of the
    head entry points, and must contain no `ttnn.` call of its own -- device work
    belongs to `tt/`, and a demo that does any is by definition a second copy.
    """
    offences = []
    demos = [n for n in sorted(os.listdir(DEMO_DIR)) if n.startswith("demo_") and n.endswith(".py")]
    assert demos, f"no demo entry points under {DEMO_DIR}"
    for name in demos:
        path = os.path.join(DEMO_DIR, name)
        with open(path) as f:
            source = f.read()
        tree = ast.parse(source, filename=path)
        calls = {_dotted(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if "build_pipeline" not in calls:
            offences.append(f"{name}: does not call build_pipeline")
        if not any(c.rpartition(".")[2] in HEAD_ENTRIES for c in calls):
            offences.append(f"{name}: calls no head entry point of tt/pipeline.py")
        ttnn_calls = sorted(c for c in calls if c.startswith("ttnn."))
        if ttnn_calls:
            offences.append(f"{name}: does device work of its own: {ttnn_calls}")
        if "__main__" not in source or "argparse" not in source:
            offences.append(f"{name}: not runnable as a __main__ + argparse entry point")
    assert not offences, "demo/ and tests/ are not sharing one pipeline:\n" + "\n".join(offences)


def test_every_head_has_a_demo_and_a_batched_gate():
    """Each of the four heads is reachable from a demo AND from the e2e gate, and the
    gate covers it at BATCH=32 as well as at one sample."""
    demo_calls, test_calls = set(), set()
    for directory, sink in ((DEMO_DIR, demo_calls), (os.path.dirname(os.path.abspath(__file__)), test_calls)):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(directory, name)) as f:
                tree = ast.parse(f.read(), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    tail = _dotted(node.func).rpartition(".")[2]
                    if tail in HEAD_ENTRIES:
                        sink.add(tail)
    assert demo_calls == HEAD_ENTRIES, f"heads without a demo: {sorted(HEAD_ENTRIES - demo_calls)}"
    assert test_calls == HEAD_ENTRIES, f"heads without an e2e gate: {sorted(HEAD_ENTRIES - test_calls)}"

    # the batched gates live in more than one module -- the VAE-decoding heads need
    # their own device (a larger conv halo), so they cannot share a fixture
    here = os.path.dirname(os.path.abspath(__file__))
    batched = set()
    for name in sorted(os.listdir(here)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        with open(os.path.join(here, name)) as f:
            tree = ast.parse(f.read(), filename=name)
        batched |= {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_") and "_batch" in n.name
        }
    assert len(batched) >= len(
        HEAD_ENTRIES
    ), f"{len(HEAD_ENTRIES)} heads but only {len(batched)} batched gates: {sorted(batched)}"


def test_pipeline_declares_its_stages():
    from models.demos.flux_2_klein_9b.tt import pipeline as P

    assert P.PIPELINE_STAGES == ["encode_text", "vae_encode", "denoise", "vae_decode", "prefill", "decode"]
    assert callable(P.build_pipeline)
