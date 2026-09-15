# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Structural gates for Call 1 (text generation) -- moved out of test_e2e_pipeline.py
so that file holds only the PCC gate (Gate 3). These check the CODE and the
forward path's coverage, never a numeric output, and share test_e2e_pipeline.py's
`device`/`pipe`/`run` fixtures via the sibling conftest.py.

  Gate 1  every routed graduated stub is still native ttnn, and every sharded
          body keeps its ShardTensor2dMesh + all_reduce (a TP=2 body is NOT
          rewritten to replication)
  Gate 2  all ten graduated modules actually executed inside the forward path

Run:  ./python_env/bin/python -m pytest \
        models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/other_tests/test_gate1_gate2.py -s
"""
from __future__ import annotations

import re
from pathlib import Path

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

DEMO_DIR = Path(P.__file__).resolve().parents[1]
STUBS = DEMO_DIR / "_stubs"

# Host-side torch COMPUTE ops that must not run in a stub's forward.
FORBIDDEN_TORCH_FNS = {
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
    "dropout",
    "argmax",
    "topk",
    "multinomial",
}
FORBIDDEN_HF = re.compile(r"\.generate\(|\.forward\s*=")
FORBIDDEN_SWEEP = re.compile(r"def\s+(coverage_step|coverage_sweep|invoke_all_stubs|_touch_all_graduated)\b")

# Functions that run on EVERY call -- the hot path the contract governs.
HOT_FNS = ("__call__", "forward", "_mix", "_route", "decode_step", "decode_prefill")
HOT_PREFIXES = ("_apply_", "run_", "_trace_step")

# One-time constant builders. They are memoised and only ever execute on the
# FIRST call for a given shape; the trace contract primes them from
# `<stage>_trace_setup`, i.e. outside the captured region. Building a constant
# with torch is explicitly allowed prep, not forward compute.
PREP_FNS = {"__init__", "_ensure_consts", "_ensure_seq", "_get_consts", "build", "_get_causal_mask"}


def _torch_compute_calls(src: str, hot_only: bool = True):
    """AST-walk `src` and yield forbidden torch COMPUTE calls in hot functions.

    AST rather than regex: the stubs quote the HF reference (`F.linear(...)`,
    `torch.einsum`) in their docstrings, and a text scan reports those as
    violations that do not exist in the code.
    """
    import ast

    tree = ast.parse(src)
    hits = []

    def is_hot(name):
        return name in HOT_FNS or any(name.startswith(p) for p in HOT_PREFIXES)

    def dotted(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name in PREP_FNS:
            continue
        if hot_only and not is_hot(fn.name):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = dotted(node.func)
            if not name:
                continue
            if name.startswith("F.") or name.startswith("torch.nn.functional."):
                hits.append((fn.name, name))
            elif name.startswith("torch.") and name.split(".")[-1] in FORBIDDEN_TORCH_FNS:
                hits.append((fn.name, name))
    return hits


# --------------------------------------------------------------------------- #
#  Gate 1 -- still real ttnn, still sharded
# --------------------------------------------------------------------------- #
def _routed_stub_files():
    return {n: STUBS / f"{n}.py" for n in P.GRADUATED_MODULES}


def test_graduated_set_matches_bringup():
    """S1/S2: the set this pipeline routes IS the set bring-up graduated."""
    on_disk = set()
    for snap in list(STUBS.glob("*.py.last_good_native")) + list(STUBS.glob("*.py.last_good_sharded")):
        on_disk.add(snap.name.split(".py.")[0])
    assert on_disk == set(
        P.GRADUATED_MODULES
    ), f"graduated-set drift: on disk {sorted(on_disk)} vs routed {sorted(P.GRADUATED_MODULES)}"


def test_gate1_stubs_are_native_ttnn():
    """No host torch compute op and no coverage sweep in any routed stub's forward."""
    offenders = []
    for name, path in _routed_stub_files().items():
        src = path.read_text()
        for fn, call in _torch_compute_calls(src):
            offenders.append(f"{name}.{fn}: torch compute op -> {call}")
        if FORBIDDEN_SWEEP.search(src):
            offenders.append(f"{name}: defines a coverage sweep")
    assert not offenders, "Gate 1 violated:\n" + "\n".join(offenders)


def test_gate1_pipeline_has_no_shortcut():
    src = (DEMO_DIR / "tt" / "pipeline.py").read_text()
    assert not FORBIDDEN_SWEEP.search(src), "pipeline defines a coverage sweep"

    offenders = [f"{fn}: {call}" for fn, call in _torch_compute_calls(src)]
    assert not offenders, "pipeline hot path uses torch compute:\n" + "\n".join(offenders)

    # HF orchestration: allowed ONLY inside the golden helper and trace setup.
    # AST-based, because the module docstring legitimately says the words
    # "model.generate()" while promising NOT to call it.
    import ast

    allowed = {"_hf_reference_text_generation", "prefill_trace_setup", "decode_trace_setup"}
    bad = []
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef) or fn.name in allowed:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
                bad.append(f"{fn.name}: .generate()")
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == "forward":
                        bad.append(f"{fn.name}: monkey-patches .forward")
    assert not bad, "pipeline uses HF orchestration outside the golden helper:\n" + "\n".join(bad)


def test_gate1_sharded_bodies_kept_their_sharding():
    """No graduated body may LOSE sharding it had in its snapshot.

    The rule is "did it regress", not "does it shard": `nemotron_h_topk_router`
    is REPLICATED by design even in its sharded snapshot (a 2688x128 gate whose
    full-width logits every chip needs), so demanding a collective of it would
    be demanding a split that would be wrong.
    """
    bad = []
    for name in P.GRADUATED_MODULES:
        snap = STUBS / f"{name}.py.last_good_sharded"
        if not snap.exists():
            continue
        was, live = snap.read_text(), (STUBS / f"{name}.py").read_text()
        for marker in ("ShardTensor2dMesh", "ShardTensorToMesh", "all_reduce", "all_gather"):
            if marker in was and marker not in live:
                bad.append(f"{name}: lost {marker}")
    assert not bad, "sharded stubs rewritten to replication:\n" + "\n".join(bad)


def test_gate1_sharding_is_live_on_device(pipe):
    ev = pipe.sharding_evidence()
    print(f"[gate1] sharded stub instances: {ev}")
    assert pipe.sharded, "pipeline did not take its sharded branch"
    assert ev, "no built stub took its TP-sharded branch -- this is a pure-replication pipeline"


# --------------------------------------------------------------------------- #
#  Gate 2 -- every graduated module invoked in the real forward path
# --------------------------------------------------------------------------- #
def test_gate2_all_graduated_modules_invoked(run):
    invoked = run["invoked"]
    print(f"[gate2] invoked ({len(invoked)}): {sorted(invoked)}")
    missing = set(P.GRADUATED_MODULES) - invoked
    assert not missing, f"graduated modules never invoked: {sorted(missing)}"
