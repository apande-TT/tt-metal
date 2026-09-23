# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate 1 (native), Gate 2 (invoked), the anti-shortcut scan, and the `layers` knob proof.

Gate 3 (end-to-end PCC) lives in the two per-call e2e tests, because it needs the real chain.
"""
from __future__ import annotations

import ast
import os
import re

import pytest

from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

pytestmark = pytest.mark.timeout(1800)

# The repo's pytest.ini caps at 300 s, which this suite measured as a FLAKE rather than a budget:
# a loaded box runs ~3x slower than an idle one and the cap fires on a run that would have passed.

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Every torch symbol the strict TT-only contract forbids in the hot path. Shape and dtype ops
# (zeros, tensor, arange, cat, reshape, expand, repeat_interleave, full_like, manual_seed,
# no_grad, .to(dtype)) are ALLOWED -- they are preparation, not compute.
FORBIDDEN_TORCH = [
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
]

# Function names a coverage sweep would hide behind. Any of these in the package is an automatic
# Gate 2 failure: a stub called once to tick a counter is not a stub in the forward path.
# Names this package binds the HF reference to. `.generate` on one of these is HF orchestration;
# `.generate` on anything else is one of this package's own explicit loops.
_HF_MODEL_NAMES = {"hf_model", "hf", "model", "reference_model", "reference"}

SWEEP_NAMES = re.compile(
    r"\b(coverage_step|coverage_sweep|invoke_all_stubs|_touch_all_graduated|touch_all_stubs|"
    r"_sweep_stubs|exercise_all_stubs)\b"
)


def _routed_stubs():
    """The graduated modules each call routes, taken from the PLAN rather than from a local list."""
    import json

    with open(os.path.join(PKG_ROOT, "e2e_plan.json")) as f:
        plan = json.load(f)
    return {h["name"]: set(h["graduated_modules_routed"]) for h in plan["task_heads"]}


def test_source_b_integrity():
    """All 31 components still have a live stub AND a graduation snapshot, and all import."""
    status = common.bringup_status()
    recorded = {c["name"] for c in status["components"]}
    graduated = set(common.graduated_modules())
    assert graduated == recorded, f"not every recorded component is graduated: {recorded - graduated}"
    assert len(graduated) == 31, f"expected 31 graduated components, found {len(graduated)}"
    for name in sorted(graduated):
        module = common.import_stub(name)
        assert hasattr(module, "build"), f"{name} exposes no build(device, torch_module)"


def test_alias_pairs_are_detected_and_both_members_are_routed():
    """The 4 alias pairs are found by comparison, and NEITHER member is dropped.

    An alias is the same work product under a second name. The pipeline splits the real work so
    both members sit in the forward path; what must never happen is one of them going unrouted.
    """
    pairs = common.alias_pairs()
    assert len(pairs) == 4, f"expected 4 alias pairs, detected {len(pairs)}: {pairs}"
    routed = set().union(*_routed_stubs().values())
    for a, b in pairs:
        assert a in routed and b in routed, f"alias pair ({a}, {b}) is not fully routed"
    print(f"alias pairs (detected, not listed): {pairs}")
    print(f"distinct graduated bodies: {len(common.graduated_modules()) - len(pairs)}")


def test_plan_routes_every_graduated_module_exactly_once():
    """Gate 2, statically: 31 = 28 + 3, disjoint. The expected set is READ from Source B."""
    graduated = set(common.graduated_modules())
    routed = _routed_stubs()
    call1, call2 = routed["text_to_speech"], routed["text_continuation"]
    assert not (call1 & call2), f"the two calls share modules: {sorted(call1 & call2)}"
    union = call1 | call2
    assert union == graduated, (
        f"missing from the pipeline: {sorted(graduated - union)}; "
        f"routed but not graduated: {sorted(union - graduated)}"
    )
    print(f"routed: text_to_speech={len(call1)} + text_continuation={len(call2)} = {len(union)}")


def test_gate1_every_routed_stub_is_still_real_ttnn():
    """Gate 1: no torch compute wrapper inside any routed stub, and every stub really uses ttnn.

    A sharded (TP>1) body would also count as native and must not be rewritten to replication --
    this bring-up is TP=1, so no `.last_good_sharded` snapshot exists and there is none to guard.
    """
    offenders = {}
    for name in sorted(common.graduated_modules()):
        path = os.path.join(common.BRINGUP_ROOT, "_stubs", f"{name}.py")
        with open(path) as f:
            src = f.read()
        tree = ast.parse(src)
        # Strip the docstrings: they legitimately NAME the torch ops the port replaces.
        hits = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr not in FORBIDDEN_TORCH:
                continue
            base = node.value
            root = base
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {"torch", "F", "nn"}:
                hits.add(f"{root.id}...{node.attr}")
        if hits:
            offenders[name] = sorted(hits)
        assert "ttnn." in src, f"{name} contains no ttnn call at all"
    assert not offenders, f"torch compute found inside graduated stubs: {offenders}"
    print(f"Gate 1: all {len(common.graduated_modules())} routed stubs are pure ttnn")


def test_no_coverage_sweep_anywhere_in_the_package():
    """Gate 2's anti-shortcut rule: a stub touched for the counter does not count."""
    found = {}
    for root, _dirs, files in os.walk(PKG_ROOT):
        if "__pycache__" in root:
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            if os.path.samefile(path, os.path.abspath(__file__)):
                continue  # this file NAMES the banned spellings in order to scan for them
            with open(path) as f:
                hit = SWEEP_NAMES.findall(f.read())
            if hit:
                found[os.path.relpath(path, PKG_ROOT)] = sorted(set(hit))
    assert not found, f"coverage-sweep shortcut found: {found}"


def test_hot_path_has_no_torch_compute_and_no_hf_orchestration():
    """The strict TT-only contract, over everything the pipeline's hot path can reach.

    EVERY file in `tt/` is scanned -- there is no exemption any more. The reference chains moved
    to the sibling `reference/` package, which is where they belong: `tt/` is the device port and
    the golden is the torch thing it is measured against. Nothing in `tt/` may import it at module
    scope (asserted below).
    """
    hot = []
    tt_dir = os.path.join(PKG_ROOT, "tt")
    for fname in sorted(os.listdir(tt_dir)):
        if fname.endswith(".py"):
            hot.append(os.path.join(tt_dir, fname))

    offenders, generate_calls = {}, {}
    for path in hot:
        with open(path) as f:
            src = f.read()
        tree = ast.parse(src)
        hits = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_TORCH:
                root = node.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in {"torch", "F", "nn"}:
                    hits.add(f"{root.id}...{node.attr}")
            if isinstance(node, ast.Attribute) and node.attr == "generate":
                # `.generate` is only HF ORCHESTRATION when the receiver is the reference model.
                # `self.continuation.generate(...)` is this package's own explicit decode loop,
                # and flagging it would be a scanner bug rather than a finding.
                receiver = ast.unparse(node.value)
                if receiver.split(".")[-1] in _HF_MODEL_NAMES:
                    generate_calls.setdefault(os.path.basename(path), []).append(f"{receiver}.generate:{node.lineno}")
        if hits:
            offenders[os.path.basename(path)] = sorted(hits)
    assert not offenders, f"torch compute in the pipeline's hot path: {offenders}"
    assert not generate_calls, f"HF orchestration (.generate) in the hot path: {generate_calls}"

    # The forward path must not import the reference at MODULE scope. A function-local import
    # inside a trace seam or the host-side noise draw is setup and is allowed; a module-scope one
    # would put the golden on the import path of the device package.
    for path in hot:
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "")] + [a.name for a in node.names]
            assert not any(
                "golden" in n or "reference" in n for n in names
            ), f"{os.path.basename(path)} imports the reference at module scope: {names}"


def test_layers_knob_is_not_inert(device, hf_model):
    """Prove the cap actually caps -- per stack, and with the stage-named overrides.

    `optimize` sets TT_PERF_LAYERS, and a builder that ignores it fails SILENTLY: the cap does
    nothing and the tool reports the knob inert. Accepting `layers` is what makes that check pass
    rather than merely be survived.
    """
    full = pipeline.build_pipeline(device, model=hf_model, heads=("text_to_speech",))
    full_depths = {stage: len(blocks) for stage, blocks in full.stacks().items()}
    print("full depths:", full_depths)
    assert full_depths["prefill"] == len(hf_model.model.layers)
    assert full_depths["acoustic"] == len(hf_model.acoustic_transformer.layers)
    del full

    capped = pipeline.build_pipeline(device, model=hf_model, heads=("text_to_speech",), layers=4)
    capped_depths = {stage: len(blocks) for stage, blocks in capped.stacks().items()}
    print("capped(layers=4) depths:", capped_depths)
    assert capped_depths["prefill"] == 4, "the global `layers` did not cap the text stack"
    assert capped_depths["prefill"] < full_depths["prefill"]
    # A capped build is still a MODEL: every stage can still run.
    assert capped.text is not None and capped.acoustic is not None and capped.vocode is not None
    del capped

    per_stack = pipeline.build_pipeline(
        device, model=hf_model, heads=("text_to_speech",), layers=None, decode_layers=6, acoustic_layers=3
    )
    per_depths = {stage: len(blocks) for stage, blocks in per_stack.stacks().items()}
    print("per-stack(decode_layers=6, acoustic_layers=3) depths:", per_depths)
    assert per_depths["decode"] == 6, "decode_layers did not cap the text stack"
    assert per_depths["acoustic"] == 3
    assert per_depths["vocode"] == full_depths["vocode"], "vocode should be untouched at layers=None"


def test_prefill_and_decode_layers_may_not_disagree(device, hf_model):
    """They are two phases of ONE stack, so a silent last-wins would be a lie."""
    with pytest.raises(ValueError, match="disagree"):
        pipeline.build_pipeline(device, model=hf_model, heads=("text_to_speech",), prefill_layers=4, decode_layers=8)
