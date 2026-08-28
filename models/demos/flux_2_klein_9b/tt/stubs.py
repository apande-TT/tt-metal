# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Loader + invocation ledger for the FLUX.2-klein-9B graduated bring-up stubs.

Source B is three bring-up directories (one per checkpoint sub-folder of the same
snapshot).  This module is the single place that

* knows where those directories are,
* loads a graduated ``_stubs/<name>.py`` by path under a unique module name (three
  of the names collide across the directories -- ``layer``, ``mlp``,
  ``encoder_stack``, ``decoder_head``, ``patch_embed``, ``self_attention``),
* proves the live stub body is still the graduated body (Gate 1), and
* records every port invocation so the e2e test can prove all 43 graduated
  modules ran inside the real forward path (Gate 2).

Nothing here does model maths.  The stub bodies are used verbatim.
"""

from __future__ import annotations

import filecmp
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))

#: Source B: bring-up directory per checkpoint sub-folder, keyed by stage.
BRINGUP_DIRS = {
    "text_encoder": "models/tt_transformers/demo/flux_2_klein_9b_text_encoder",
    "transformer": "models/tt_dit/pipelines/flux_2_klein_9b_transformer",
    "vae": "models/tt_dit/pipelines/flux_2_klein_9b_vae",
}

#: Snapshot suffixes that mark a component as GRADUATED, in preference order.
_GRADUATED_SUFFIXES = (".last_good_sharded", ".last_good_native")


def bringup_dir(stage: str) -> str:
    return os.path.join(REPO_ROOT, BRINGUP_DIRS[stage])


def stub_path(stage: str, name: str) -> str:
    return os.path.join(bringup_dir(stage), "_stubs", f"{name}.py")


def captured_dir(stage: str, name: str) -> str:
    return os.path.join(bringup_dir(stage), "_captured", name)


def graduated_components(stage: str) -> list[str]:
    """Names in this stage's ``bringup_status.json`` that have a graduated snapshot."""
    with open(os.path.join(bringup_dir(stage), "bringup_status.json")) as f:
        status = json.load(f)
    out = []
    for comp in status["components"]:
        name = comp["name"]
        base = stub_path(stage, name)
        if not os.path.isfile(base):
            continue
        if any(os.path.isfile(base + s) for s in _GRADUATED_SUFFIXES):
            out.append(name)
    return sorted(out)


def all_graduated() -> dict[str, list[str]]:
    return {stage: graduated_components(stage) for stage in BRINGUP_DIRS}


def snapshot_for(stage: str, name: str) -> str:
    base = stub_path(stage, name)
    for suffix in _GRADUATED_SUFFIXES:
        if os.path.isfile(base + suffix):
            return base + suffix
    raise FileNotFoundError(f"{stage}/{name} has no graduated snapshot")


# ------------------------------------------------------- the batch-32 delta
#
# The bring-up graduated every stub at B=1, and some of them wrote that 1 into a
# `ttnn.slice` end bound or into the rank-4 view they hand to
# `nlp_create_qkv_heads`.  At B=32 such a bound does not fail -- it silently keeps
# sample 0 and drops samples 1..31 -- so the pipeline cannot carry a leading batch
# axis until those bounds come from the tensor itself.
#
# Rewriting a graduated body is exactly what Gate 1 exists to catch, so the delta is
# DECLARED rather than merely applied: `tt/batch_patches/<stage>.json` lists, per
# stub, the exact substrings replaced.  Gate 1 then no longer byte-compares -- it
# re-derives, requiring
#
#     live_body == apply_batch_patches(graduated_snapshot)
#
# so the live file is provably the graduated body plus exactly the declared edits,
# and the `.last_good_*` snapshots stay untouched as the record of what bring-up
# produced.  `assert_patches_are_batch_only` (used by the Gate 1 test) additionally
# requires every declared edit to be a batch generalisation and nothing else.

BATCH_PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_patches")

#: What a batch generalisation is allowed to introduce.  Anything else in a declared
#: `new` string fails Gate 1: the delta must be a bound, not new maths.
_BATCH_PATCH_TOKENS = re.compile(r"^[\s\w\.\(\)\[\],\-*+/:=]*$")


def batch_patches(stage: str) -> dict:
    """The declared batch-32 delta for this stage, keyed by stub name."""
    path = os.path.join(BATCH_PATCH_DIR, f"{stage}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def apply_batch_patches(text: str, edits: list) -> str:
    """Apply the declared edits, in order, requiring each to match exactly once."""
    for edit in edits:
        old, new = edit["old"], edit["new"]
        count = text.count(old)
        expect = int(edit.get("count", 1))
        if count != expect:
            raise AssertionError(
                f"batch patch {edit.get('why', '')!r}: expected {expect} occurrence(s) of "
                f"{old!r} in the graduated snapshot, found {count}"
            )
        text = text.replace(old, new)
    return text


def assert_body_is_graduated(stage: str, name: str) -> str:
    """Gate 1, part 1: the live stub body IS the graduated body plus the DECLARED
    batch-32 delta -- nothing else."""
    base = stub_path(stage, name)
    snap = snapshot_for(stage, name)
    edits = batch_patches(stage).get(name, [])
    if not edits:
        if not filecmp.cmp(base, snap, shallow=False):
            raise AssertionError(
                f"Gate 1: {stage}/_stubs/{name}.py differs from its graduated snapshot "
                f"{os.path.basename(snap)} -- the pipeline must compose the graduated body as-is"
            )
        return snap

    with open(snap) as f:
        want = apply_batch_patches(f.read(), edits)
    with open(base) as f:
        live = f.read()
    if live != want:
        raise AssertionError(
            f"Gate 1: {stage}/_stubs/{name}.py is not its graduated snapshot "
            f"{os.path.basename(snap)} plus the {len(edits)} declared batch edit(s) in "
            f"tt/batch_patches/{stage}.json -- an undeclared change to a graduated body"
        )
    return snap


def assert_patches_are_batch_only(stage: str, name: str, forbidden: set) -> None:
    """Gate 1, part 1b: every declared edit is a BATCH generalisation.

    A batch edit replaces a hard-coded leading `1` with a bound read off a tensor,
    or moves a rank-promoting `unsqueeze`/`squeeze` off axis 0 onto axis 1.  It may
    not introduce a torch compute op, an import, an HF call or an `if`.
    """
    for edit in batch_patches(stage).get(name, []):
        new, old = edit["new"], edit["old"]
        assert edit.get("why"), f"{stage}/{name}: every batch patch must say why"
        for bad in ("import ", "torch.", "def ", "class ", "lambda", "if "):
            assert bad not in new, f"{stage}/{name}: batch patch introduces {bad!r}: {new!r}"
        for op in forbidden:
            assert f".{op}(" not in new, f"{stage}/{name}: batch patch introduces torch.{op}"
        assert _BATCH_PATCH_TOKENS.match(new), f"{stage}/{name}: batch patch is not a plain bound: {new!r}"
        # a batch edit only ever touches the leading axis: normalising the new bound
        # back to `1` must reproduce the graduated text.
        assert _normalise_batch_bound(new) == _normalise_batch_bound(old), (
            f"{stage}/{name}: batch patch changes more than the leading axis\n"
            f"  graduated: {old!r}\n  live     : {new!r}"
        )


#: `<expr>.shape[0]` (the bound read off the tensor) normalises back to the
#: graduated `1`; `unsqueeze(x, 1)` / `squeeze(x, 1)` back to axis 0.
_SHAPE0 = re.compile(r"[A-Za-z_][A-Za-z_0-9\.]*\.shape\[0\]")
_UNSQ = re.compile(r"((?:un)?squeeze)\(([^()]*(?:\([^()]*\))?[^()]*), 1\)")


def _normalise_batch_bound(text: str) -> str:
    text = _SHAPE0.sub("1", text)
    return _UNSQ.sub(r"\1(\2, 0)", text)


_MODULE_CACHE: dict[tuple[str, str], object] = {}


def load_stub_module(stage: str, name: str):
    """Import ``<bringup_dir>/_stubs/<name>.py`` under a collision-free module name."""
    key = (stage, name)
    hit = _MODULE_CACHE.get(key)
    if hit is not None:
        return hit

    assert_body_is_graduated(stage, name)

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    mod_name = f"_f2k_stub__{stage}__{name}"
    spec = importlib.util.spec_from_file_location(mod_name, stub_path(stage, name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    _MODULE_CACHE[key] = module
    return module


# --------------------------------------------------------------------------- ledger


@dataclass
class _Record:
    stage: str
    name: str
    positions: set = field(default_factory=set)
    calls: int = 0
    downstream: bool = False
    #: keep the most recent output alive so its id() cannot be recycled by a
    #: later tensor and produce a false "downstream" hit.
    last_outputs: list = field(default_factory=list)
    ports: list = field(default_factory=list)


class Ledger:
    """Records which graduated stub ran and at which position.

    `calls` and `positions` are mechanical and exact.  `downstream` is a
    best-effort signal: it flips when a port's returned tensor object is later
    handed straight to another ledger-bound port (or marked as the head's output),
    which catches the obvious dead-end but NOT the common case where a plain ttnn
    op sits between two ports and the object identity changes.  So `downstream` is
    reported, never asserted.

    What actually proves a stub is load-bearing is the pair of gates around this
    ledger: the head's final PCC (a dropped block collapses it) and the ablation
    test, which neutralises a routed port through `ports()` and requires the final
    PCC to fall.  That is why there is no "call each stub once" shortcut here.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], _Record] = {}
        self._pending: dict[int, tuple[str, str]] = {}

    # ------------------------------------------------------------------ binding
    def bind(self, stage: str, name: str, position: str, port):
        rec = self._records.setdefault((stage, name), _Record(stage, name))
        rec.positions.add(position)
        wrapper = _LedgerPort(self, rec, position, port)
        rec.ports.append(wrapper)
        return wrapper

    def ports(self, stage: str, name: str) -> list:
        """Every live wrapper bound for this stub -- the seam the ablation test uses."""
        rec = self._records.get((stage, name))
        return list(rec.ports) if rec else []

    # ------------------------------------------------------------- bookkeeping
    def _note_inputs(self, obj) -> None:
        for tid in _tensor_ids(obj):
            owner = self._pending.pop(tid, None)
            if owner is not None:
                self._records[owner].downstream = True

    def _note_output(self, rec: _Record, out) -> None:
        rec.last_outputs = [out]
        for tid in _tensor_ids(out):
            self._pending[tid] = (rec.stage, rec.name)

    def mark_final(self, obj) -> None:
        """The head's real output; whatever produced it was consumed."""
        self._note_inputs(obj)

    # ---------------------------------------------------------------- reporting
    def routed(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {stage: [] for stage in BRINGUP_DIRS}
        for stage, name in self._records:
            out[stage].append(name)
        return {k: sorted(v) for k, v in out.items()}

    def rows(self) -> list[dict]:
        return [
            {
                "stage": r.stage,
                "name": r.name,
                "calls": r.calls,
                "positions": sorted(r.positions),
                "downstream": r.downstream,
            }
            for r in sorted(self._records.values(), key=lambda r: (r.stage, r.name))
        ]

    def table(self) -> str:
        lines = [f"{'stage':<13} {'stub':<36} {'calls':>6}  {'dstr':<5} positions"]
        for row in self.rows():
            lines.append(
                f"{row['stage']:<13} {row['name']:<36} {row['calls']:>6}  "
                f"{'yes' if row['downstream'] else '-':<5} {', '.join(row['positions'])}"
            )
        return "\n".join(lines)

    def missing(self) -> dict[str, list[str]]:
        routed = self.routed()
        return {stage: sorted(set(graduated_components(stage)) - set(routed[stage])) for stage in BRINGUP_DIRS}

    def no_downstream(self) -> list[str]:
        """Best-effort: ports whose returned object was never handed to another port.
        Reported, not asserted -- see the class docstring."""
        return [f"{r['stage']}/{r['name']}" for r in self.rows() if not r["downstream"]]

    def release(self) -> None:
        for rec in self._records.values():
            rec.last_outputs = []
        self._pending.clear()

    def drop_ports(self, *stages: str) -> None:
        """Let go of the built ports (and their DEVICE WEIGHTS) for these bring-up
        stages, keeping the counts and positions the gate reports.

        Without this the ledger is a memory leak: every `_LedgerPort` holds the
        graduated port object, which holds ~2.5 GB/chip of staged weights, so a
        pipeline that "released" a stage still could not build the next one.
        """
        for (stage, _), rec in self._records.items():
            if stages and stage not in stages:
                continue
            rec.ports = []
            rec.last_outputs = []
        self._pending.clear()

    def restore_all(self) -> None:
        for rec in self._records.values():
            for port in rec.ports:
                port.restore()


class _LedgerPort:
    """Call-through wrapper: forwards to the graduated port and records the call."""

    __slots__ = ("_ledger", "_record", "_position", "_port", "_override")

    def __init__(self, ledger: Ledger, record: _Record, position: str, port) -> None:
        self._ledger = ledger
        self._record = record
        self._position = position
        self._port = port
        self._override = None

    def override(self, fn) -> None:
        """Ablation seam: run `fn` instead of the graduated port, so a test can prove
        the port's result actually reaches the head's output."""
        self._override = fn

    def restore(self) -> None:
        self._override = None

    def __call__(self, *args, **kwargs):
        self._ledger._note_inputs(args)
        self._ledger._note_inputs(kwargs)
        out = (self._override or self._port)(*args, **kwargs)
        self._record.calls += 1
        self._ledger._note_output(self._record, out)
        return out

    # a couple of the ports are used for their attributes as well as their call
    def __getattr__(self, item):
        return getattr(self._port, item)

    @property
    def port(self):
        return self._port

    @property
    def position(self) -> str:
        return self._position


def _tensor_ids(obj, _depth: int = 0):
    """ids of every ttnn.Tensor reachable in a shallow arg structure."""
    if _depth > 3:
        return
    import ttnn

    if isinstance(obj, ttnn.Tensor):
        yield id(obj)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _tensor_ids(item, _depth + 1)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _tensor_ids(item, _depth + 1)
