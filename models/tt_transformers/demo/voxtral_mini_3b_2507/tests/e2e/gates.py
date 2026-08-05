# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable gate machinery for the Voxtral-Mini-3B-2507 end-to-end suite.

Three gates guard the e2e pipeline (see ``e2e_plan.json::self_validation_plan``):

* **Gate 1 — nativeness.** :func:`gate1_static_scan` AST-walks every routed stub file plus
  ``tt/pipeline.py`` and flags torch compute ops, host readbacks and HF orchestration that occur
  inside a *hot-path* function (weight extraction in ``__init__`` is legitimate and is NOT
  flagged). :func:`gate1_runtime_probe` then measures what actually executes with
  ``models.common.native_probe.run_native_probe`` — a source scan can be evaded by aliasing, the
  runtime probe cannot.
* **Gate 2 — every graduated module does real work.** :func:`gate2_invoked` asserts that each
  routed stub's counting proxy fired at least once during the real forward pass.
* **Gate 3 — parity.** :func:`pcc` / :func:`report_pcc` compute and print the per-stream and
  aggregate PCC of the TT task output against the HF golden.

:func:`graduated_inventory` is the independent cross-check that no graduated module was silently
dropped from the pipeline: it re-derives the graduated list from ``bringup_status.json`` plus the
``_stubs/<name>.py.last_good_{native,sharded}`` snapshots, rather than trusting the pipeline's own
``ROUTED_STUBS``.

Everything in this module is host-only and needs no device, so the static parts can be run
stand-alone::

    ./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.tests.e2e.gates
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from models.common.utility_functions import comp_pcc

# --------------------------------------------------------------------------------------------
# Gate 1 — static scan
# --------------------------------------------------------------------------------------------

#: ``torch.<op>`` calls that constitute real compute. A pure-ttnn hot path must not contain any of
#: them. Deliberately does NOT include shape/dtype/allocation helpers (``torch.arange``,
#: ``torch.zeros``, ``torch.no_grad`` ...) — those appear in legitimate build-time code and, when
#: they do show up in a hot path, the runtime probe is the authoritative detector.
TORCH_COMPUTE_OPS = frozenset(
    {
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
)

#: Function names that start the hot path. Anything these call (within the same file) is hot too.
HOT_PATH_PATTERNS = (
    "__call__",
    "forward",
    "run_*",
    "_apply_*",
    "decode_step",
    "decode_prefill",
    "*_trace_step",
    "*_step",
)

#: ``self.model.<name>(...)`` calls that are ordinary build-time bookkeeping, not HF orchestration.
_HF_MODEL_METHOD_ALLOWLIST = frozenset(
    {
        "eval",
        "train",
        "to",
        "cpu",
        "cuda",
        "float",
        "half",
        "bfloat16",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "children",
        "named_children",
        "modules",
        "named_modules",
        "state_dict",
        "requires_grad_",
        "get_input_embeddings",
        "get_output_embeddings",
        "get_submodule",
    }
)


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, else ``None``."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _import_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound to ``torch``, names bound to ``torch.nn.functional``) in this file."""
    torch_names = {"torch"}
    functional_names = {"torch.nn.functional", "torch.functional", "F", "nn.functional"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    torch_names.add(alias.asname or "torch")
                elif alias.name in ("torch.nn.functional", "torch.functional"):
                    functional_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                bound = alias.asname or alias.name
                if mod == "torch" and alias.name == "nn":
                    functional_names.add(f"{bound}.functional")
                elif mod in ("torch.nn", "torch") and alias.name == "functional":
                    functional_names.add(bound)
    return torch_names, functional_names


def _is_hot_name(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in HOT_PATH_PATTERNS)


def _collect_functions(tree: ast.AST) -> dict[str, list[ast.AST]]:
    """simple function name -> every def with that name anywhere in the file."""
    out: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).append(node)
    return out


def _callee_names(fn: ast.AST) -> set[str]:
    """Names of same-file functions this def could call: ``helper(...)`` and ``self.helper(...)``."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            names.add(func.attr)
    return names


def _hot_defs(tree: ast.AST) -> list[ast.AST]:
    """Hot-path seeds plus the transitive closure of same-file functions they call."""
    by_name = _collect_functions(tree)
    hot_names = {name for name in by_name if _is_hot_name(name)}
    changed = True
    while changed:
        changed = False
        for name in list(hot_names):
            for fn in by_name.get(name, []):
                for callee in _callee_names(fn):
                    if callee in by_name and callee not in hot_names:
                        hot_names.add(callee)
                        changed = True
    defs: list[ast.AST] = []
    for name in sorted(hot_names):
        defs.extend(by_name[name])
    return defs


def _scan_hot_calls(fn: ast.AST, torch_names: set[str], functional_names: set[str]) -> list[dict]:
    """(a) torch compute ops / torch.nn.functional and (b) host readbacks, inside one hot def."""
    violations: list[dict] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else None

        if dotted:
            head, _, tail = dotted.rpartition(".")
            if head in functional_names or any(dotted.startswith(f + ".") for f in functional_names):
                violations.append({"line": node.lineno, "kind": "torch_functional", "detail": f"{dotted}(...)"})
                continue
            if head in torch_names and tail in TORCH_COMPUTE_OPS:
                violations.append({"line": node.lineno, "kind": "torch_compute_op", "detail": f"{dotted}(...)"})
                continue
            if tail == "to_torch":
                violations.append({"line": node.lineno, "kind": "host_readback", "detail": f"{dotted}(...)"})
                continue

        if attr == "numpy":
            violations.append({"line": node.lineno, "kind": "host_readback", "detail": ".numpy()"})
    return violations


def _scan_hf_orchestration(tree: ast.AST) -> list[dict]:
    """(c) HF orchestration anywhere in the file: generate(), forward monkey-patching, and direct
    calls into HF submodules of the reference model."""
    violations: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            dotted = _dotted(node.func)
            if node.func.attr == "generate":
                violations.append(
                    {
                        "line": node.lineno,
                        "kind": "hf_generate",
                        "detail": f"{dotted or '<expr>.generate'}(...)",
                    }
                )
                continue
            owner = _dotted(node.func.value)
            if owner in ("self.model", "hf_model", "self.hf_model") and node.func.attr not in (
                _HF_MODEL_METHOD_ALLOWLIST
            ):
                violations.append(
                    {
                        "line": node.lineno,
                        "kind": "hf_submodule_call",
                        "detail": f"{dotted}(...)",
                    }
                )
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "forward":
                    violations.append(
                        {
                            "line": node.lineno,
                            "kind": "forward_monkeypatch",
                            "detail": f"{_dotted(tgt) or '<expr>.forward'} = ...",
                        }
                    )
    return violations


#: Trailing-comment pragma that waives the output-boundary readback (and nothing else).
READBACK_WAIVER = "# gate1: allow-readback"


def _readback_waiver(lines: Sequence[str], lineno: int) -> str | None:
    """The waiver reason on source line ``lineno``, or ``None`` if the line carries no pragma."""
    if not (1 <= lineno <= len(lines)):
        return None
    text = lines[lineno - 1]
    idx = text.find(READBACK_WAIVER)
    if idx < 0:
        return None
    return text[idx + len(READBACK_WAIVER) :].strip() or "(no reason given)"


def gate1_static_scan(stub_paths: dict[str, str], extra_files: Sequence[str] | None = None) -> dict:
    """Gate 1 (static half): AST-scan every stub file plus ``extra_files``.

    Flags, **only inside hot-path functions**: torch compute ops, any ``torch.nn.functional.*`` /
    ``F.*`` call, ``ttnn.to_torch(...)`` and ``.numpy()``. Flags, anywhere in the file: HF
    orchestration (``*.generate(...)``, ``*.forward = ...``, ``self.model.<name>(...)``).

    Weight extraction inside ``__init__`` / ``build()`` is allowed by construction: those functions
    are not hot-path seeds and nothing hot calls them.

    One narrow escape hatch exists, for the single unavoidable readback at the *output boundary*
    (the final logits/token ids have to leave the device to become a ``TaskResult``): a
    ``# gate1: allow-readback <reason>`` trailing comment waives a ``host_readback`` on that line
    only. It never waives a torch compute op or HF orchestration, and every waiver is listed in
    ``result["waived"]`` and printed, so it stays visible instead of hiding.

    Returns ``{"ok": bool, "violations": [{"file","line","kind","detail"}, ...],
    "waived": [...], "scanned": [paths], "hot_functions": {path: [names]}}``.
    """
    files: list[str] = []
    for path in (stub_paths or {}).values():
        if path and str(path) not in files:
            files.append(str(path))
    for path in extra_files or []:
        if path and str(path) not in files:
            files.append(str(path))

    violations: list[dict] = []
    waived: list[dict] = []
    hot_functions: dict[str, list[str]] = {}
    scanned: list[str] = []

    for path in files:
        p = Path(path)
        if not p.is_file():
            violations.append({"file": str(p), "line": 0, "kind": "missing_file", "detail": "file does not exist"})
            continue
        source = p.read_text()
        try:
            tree = ast.parse(source, filename=str(p))
        except SyntaxError as exc:
            violations.append({"file": str(p), "line": exc.lineno or 0, "kind": "syntax_error", "detail": str(exc)})
            continue
        scanned.append(str(p))

        lines = source.splitlines()
        torch_names, functional_names = _import_aliases(tree)
        hot = _hot_defs(tree)
        hot_functions[str(p)] = sorted({getattr(fn, "name", "?") for fn in hot})

        found: list[dict] = []
        seen: set[tuple] = set()
        for fn in hot:
            for vio in _scan_hot_calls(fn, torch_names, functional_names):
                key = (vio["line"], vio["kind"], vio["detail"])
                if key in seen:
                    continue
                seen.add(key)
                found.append({"file": str(p), **vio})
        for vio in _scan_hf_orchestration(tree):
            key = (vio["line"], vio["kind"], vio["detail"])
            if key in seen:
                continue
            seen.add(key)
            found.append({"file": str(p), **vio})

        for vio in found:
            reason = _readback_waiver(lines, vio["line"]) if vio["kind"] == "host_readback" else None
            if reason is None:
                violations.append(vio)
            else:
                waived.append({**vio, "waiver": reason})

    violations.sort(key=lambda v: (v["file"], v["line"]))
    waived.sort(key=lambda v: (v["file"], v["line"]))
    return {
        "ok": not violations,
        "violations": violations,
        "waived": waived,
        "scanned": scanned,
        "hot_functions": hot_functions,
    }


def print_static_scan(result: dict) -> None:
    """Human-readable Gate-1 static scan report."""
    print(f"[gate1] static scan: {len(result.get('scanned', []))} file(s)")
    for vio in result.get("waived", []):
        print(f"[gate1] WAIVED    {vio['file']}:{vio['line']} {vio['kind']}: {vio['detail']} " f"-- {vio['waiver']}")
    for vio in result.get("violations", []):
        print(f"[gate1] VIOLATION {vio['file']}:{vio['line']} {vio['kind']}: {vio['detail']}")
    if any(v["kind"] == "host_readback" for v in result.get("violations", [])):
        print(
            f"[gate1] hint: if a flagged readback IS the output boundary (device logits/ids -> "
            f"TaskResult), annotate that line with `{READBACK_WAIVER} <reason>` and it is reported "
            f"as WAIVED instead of failing."
        )
    print(f"[gate1] static scan ok={result.get('ok')} violations={len(result.get('violations', []))}")


# --------------------------------------------------------------------------------------------
# Gate 1 — runtime probe
# --------------------------------------------------------------------------------------------


def gate1_runtime_probe(thunk: Callable[[], Any], stub_path_for_sidecar: str | Path) -> dict:
    """Gate 1 (runtime half): execute ``thunk`` under ``models.common.native_probe``.

    Counts torch compute ops (via a ``TorchFunctionMode``, so aliasing cannot evade it) and ttnn
    device dispatches. Returns the probe dict plus:

    * ``"ok"``    — ``torch_ops == 0 and ttnn_dispatch > 0``
    * ``"result"`` — whatever ``thunk()`` returned (the caller usually needs it)

    ``torch_ops == -1`` means the instrumentation itself failed; that is reported as NOT ok so a
    broken probe can never be mistaken for a clean run.
    """
    from models.common.native_probe import run_native_probe

    out, probe = run_native_probe(str(stub_path_for_sidecar), thunk)
    probe = dict(probe)
    torch_ops = int(probe.get("torch_ops", -1))
    dispatch = int(probe.get("ttnn_dispatch", 0))
    probe["ok"] = torch_ops == 0 and dispatch > 0
    probe["result"] = out
    print(
        f"[gate1] native_probe ttnn_dispatch={dispatch} torch_ops={torch_ops} "
        f"ok={probe['ok']} sidecar={stub_path_for_sidecar}.native_probe.json"
    )
    if torch_ops > 0:
        print(f"[gate1] torch ops that executed: {probe.get('torch_op_names')}")
    elif torch_ops < 0:
        print("[gate1] probe instrumentation failed (torch_ops=-1) — treated as NOT ok")
    return probe


# --------------------------------------------------------------------------------------------
# Gate 2 — every routed stub actually ran
# --------------------------------------------------------------------------------------------


def gate2_invoked(counts: dict[str, int], required: Iterable[str]) -> dict:
    """Gate 2: every required stub's counting proxy fired at least once during the real forward.

    A stub with count 0 is rejected — being wired into the object graph is not the same as being
    on the data path. Prints the invocation table and returns
    ``{"ok": bool, "missing": [...], "table": str}``.
    """
    required = list(required)
    counts = dict(counts or {})
    width = max([len(n) for n in required] + [len(n) for n in counts] + [len("stub")])

    lines = [f"{'stub'.ljust(width)}  {'calls':>7}  status"]
    lines.append("-" * (width + 20))
    missing: list[str] = []
    for name in required:
        n = int(counts.get(name, 0))
        ok = n >= 1
        if not ok:
            missing.append(name)
        lines.append(f"{name.ljust(width)}  {n:>7}  {'ok' if ok else 'NOT INVOKED'}")

    extra = [n for n in sorted(counts) if n not in required]
    for name in extra:
        lines.append(f"{name.ljust(width)}  {int(counts[name]):>7}  (not in required list)")

    table = "\n".join(lines)
    print("[gate2] stub invocation table (counts from the last real run):")
    print(table)
    print(f"[gate2] ok={not missing} required={len(required)} missing={missing}")
    return {"ok": not missing, "missing": missing, "table": table}


# --------------------------------------------------------------------------------------------
# Gate 3 — PCC
# --------------------------------------------------------------------------------------------


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two tensors via ``models.common.utility_functions.comp_pcc``.

    Returns ``-1.0`` (and prints a flag line) when either side carries nan/inf or the correlation
    is otherwise undefined, so a degenerate comparison can never read as a pass.
    """
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    if a.numel() != b.numel():
        print(f"[pcc] FLAG shape mismatch: {a.numel()} vs {b.numel()} -> -1.0")
        return -1.0
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        print("[pcc] FLAG non-finite value (nan/inf) in one of the tensors -> -1.0")
        return -1.0
    try:
        _, value = comp_pcc(a, b, 0.99)
        value = float(value)
    except Exception as exc:  # noqa: BLE001
        print(f"[pcc] FLAG comp_pcc raised {type(exc).__name__}: {exc} -> -1.0")
        return -1.0
    if not math.isfinite(value):
        print(f"[pcc] FLAG non-finite PCC ({value}) -> -1.0")
        return -1.0
    return value


def report_pcc(tag: str, per_stream: Sequence[float], aggregate: float, threshold: float) -> None:
    """Print one ``e2e PCC[<tag>][stream=i]=<v>`` line per stream plus the aggregate line."""
    for i, value in enumerate(per_stream):
        flag = "" if value >= threshold else "   <-- BELOW THRESHOLD"
        print(f"e2e PCC[{tag}][stream:{i}]={value}{flag}")
    worst = min(per_stream) if len(per_stream) else float("nan")
    print(f"e2e PCC[{tag}][aggregate]={aggregate} threshold={threshold} worst_stream={worst}")


# --------------------------------------------------------------------------------------------
# Independent graduated-module inventory
# --------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graduated_inventory(bringup_dir: str | Path) -> dict:
    """Re-derive the graduated-module list straight from the bring-up artefacts.

    For every component in ``bringup_status.json`` require a live ``_stubs/<name>.py`` AND a
    ``.py.last_good_native`` or ``.py.last_good_sharded`` snapshot, then sha256-compare live vs
    snapshot and sha256-group the live bodies to expose duplicates (which must be routed over
    disjoint data / distinct layer indices rather than deduplicated away).

    This is deliberately independent of ``tt/pipeline.ROUTED_STUBS``: the test compares the two so
    that an unrouted graduated module is a hard failure rather than an omission nobody notices.

    Returns ``{"graduated": [...], "live_equals_snapshot": {name: bool},
    "duplicate_groups": [[name, ...]], "snapshot_kind": {...}, "stub_paths": {...},
    "problems": [...]}``.
    """
    root = Path(bringup_dir).resolve()
    status_p = root / "bringup_status.json"
    stubs_dir = root / "_stubs"
    status = json.loads(status_p.read_text())

    graduated: list[str] = []
    live_equals_snapshot: dict[str, bool] = {}
    snapshot_kind: dict[str, str] = {}
    stub_paths: dict[str, str] = {}
    digests: dict[str, str] = {}
    problems: list[str] = []
    repairs: dict[str, str] = {}
    repairs_p = stubs_dir / "_e2e_repairs.json"
    declared_repairs = {}
    if repairs_p.is_file():
        try:
            declared_repairs = json.loads(repairs_p.read_text()).get("repairs", {})
        except Exception:  # noqa: BLE001
            declared_repairs = {}

    for comp in status.get("components", []):
        name = comp.get("name")
        if not name:
            continue
        live = stubs_dir / f"{name}.py"
        if not live.is_file():
            problems.append(f"{name}: no live _stubs/{name}.py")
            continue
        snap = None
        for kind in ("last_good_native", "last_good_sharded"):
            cand = stubs_dir / f"{name}.py.{kind}"
            if cand.is_file():
                snap = cand
                snapshot_kind[name] = kind
                break
        if snap is None:
            problems.append(f"{name}: no .last_good_native / .last_good_sharded snapshot")
            continue
        graduated.append(name)
        stub_paths[name] = str(live)
        live_sha = _sha256(live)
        digests[name] = live_sha
        same = live_sha == _sha256(snap)
        live_equals_snapshot[name] = same
        if not same:
            # A live body may legitimately diverge from its snapshot when a
            # graduated-stub DEFECT was repaired for the e2e chain -- but only if
            # the repair is declared in _stubs/_e2e_repairs.json with the exact
            # sha256 of the repaired body.  An undeclared divergence is still a
            # hard failure, so a silently edited (or reverted) stub cannot slip by.
            dec = declared_repairs.get(name)
            if dec and dec.get("repaired_sha256") == live_sha:
                repairs[name] = dec.get("reason", "declared repair")
            else:
                problems.append(
                    f"{name}: live body differs from its {snapshot_kind[name]} snapshot "
                    f"and is not declared in _stubs/_e2e_repairs.json with sha256={live_sha[:12]}"
                )

    by_digest: dict[str, list[str]] = {}
    for name, digest in digests.items():
        by_digest.setdefault(digest, []).append(name)
    duplicate_groups = [sorted(names) for names in by_digest.values() if len(names) > 1]
    duplicate_groups.sort()

    return {
        "graduated": sorted(graduated),
        "live_equals_snapshot": live_equals_snapshot,
        "duplicate_groups": duplicate_groups,
        "snapshot_kind": snapshot_kind,
        "stub_paths": stub_paths,
        "problems": problems,
        "declared_repairs": repairs,
    }


def print_inventory(inv: dict) -> None:
    """Human-readable graduated-module inventory."""
    graduated = inv.get("graduated", [])
    print(f"[inventory] graduated components: {len(graduated)}")
    for name, reason in sorted(inv.get("declared_repairs", {}).items()):
        print(f"[inventory] DECLARED REPAIR {name}: {reason[:160]}")
    for name in graduated:
        same = inv.get("live_equals_snapshot", {}).get(name)
        kind = inv.get("snapshot_kind", {}).get(name, "?")
        print(f"[inventory]   {name:<32} snapshot={kind:<18} live_equals_snapshot={same}")
    for group in inv.get("duplicate_groups", []):
        print(f"[inventory] duplicate body group (must be routed over disjoint data): {group}")
    for problem in inv.get("problems", []):
        print(f"[inventory] PROBLEM {problem}")


def _self_test() -> int:
    """Run the device-free halves against the real ``_stubs/`` tree (see module docstring)."""
    bringup_dir = Path(__file__).resolve().parents[2]
    inv = graduated_inventory(bringup_dir)
    print_inventory(inv)
    print()
    scan = gate1_static_scan(inv["stub_paths"], [])
    print_static_scan(scan)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
