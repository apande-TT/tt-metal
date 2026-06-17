"""Roofline oracle — the THEORETICAL hardware floor per op, so the loop can be
gap-driven (chase the floor) instead of knob-driven (try levers until none help).

For each hot op in a profile bucket's `top_ops`, compute ideal_ms = the minimum time
physics allows, and gap_ms = measured - ideal (the attainable speedup). Reuses the
data already in `top_ops` (op_code, shape "MxK @ KxN", count, device_ms, fidelity,
cores) and the arch peaks in environment.ARCH_FACTS. Adds NO new profiling.

SCOPE (v1): COMPUTE-bound roofline for matmul/linear only (FLOPs / peak_TFLOPs at the
op's fidelity, against the FULL grid so under-utilization shows up as gap). The
MEMORY-bound roofline (bytes / DRAM_bw) needs per-op dtype+tensor-size, which the
tracy CSV parse does not yet extract — that is the next increment; until then
non-matmul ops report ideal_ms=None (no model yet) rather than a wrong floor.
"""

from __future__ import annotations

import re
from typing import Any

from .environment import ARCH_FACTS

_SHAPE_RE = re.compile(r"^\s*(\d+)x(\d+)\s*@\s*(\d+)x(\d+)\s*$")
_MATMUL_OPS = ("matmul", "linear")


def _facts(env: dict[str, Any]) -> dict[str, Any]:
    """Arch facts from the run's env block, falling back to ARCH_FACTS by arch name
    (so the oracle works on older profiles whose env predates peak_tflops_per_core)."""
    arch = (env or {}).get("arch", "")
    base = ARCH_FACTS.get(arch, {})
    merged = dict(base)
    merged.update({k: v for k, v in (env or {}).items() if v is not None})
    if "peak_tflops_per_core" not in merged and base:
        merged["peak_tflops_per_core"] = base.get("peak_tflops_per_core", {})
    return merged


def parse_matmul_shape(shape: str):
    """'32x1024 @ 1024x1024' -> (M=32, K=1024, N=1024); None if not a matmul fingerprint
    or any dim is non-numeric ('?')."""
    mobj = _SHAPE_RE.match(shape or "")
    if not mobj:
        return None
    m, k0, k1, n = (int(g) for g in mobj.groups())
    k = k0 if k0 == k1 else max(k0, k1)  # contraction dim; tolerate pad mismatch
    return m, k, n


def matmul_flops(m: int, k: int, n: int) -> int:
    """2*M*N*K mul-adds for one C[M,N] = A[M,K] @ B[K,N]."""
    return 2 * m * k * n


def _full_grid_cores(facts: dict[str, Any]) -> int:
    if facts.get("worker_cores"):
        return int(facts["worker_cores"])
    gx, gy = facts.get("grid_x"), facts.get("grid_y")
    return int(gx) * int(gy) if gx and gy else 64


def ideal_ms_compute(flops_total: float, fidelity: str, facts: dict[str, Any]) -> float | None:
    """Theoretical min ms for `flops_total` against the FULL grid at `fidelity`.
    Full grid (not the op's current cores) so under-utilization surfaces as gap."""
    peaks = facts.get("peak_tflops_per_core") or {}
    per_core = peaks.get((fidelity or "").lower()) or peaks.get("hifi4")  # hifi4 = most conservative floor
    if not per_core:
        return None
    chip_flops_per_s = per_core * 1e12 * _full_grid_cores(facts)
    return (flops_total / chip_flops_per_s) * 1e3


def ideal_ms_memory(total_bytes: float, memory: str, facts: dict[str, Any]) -> float | None:
    """Theoretical min ms to move `total_bytes` at the relevant bandwidth tier.
    DRAM-resident -> DRAM bw; L1/sharded -> L1 bw if known else DRAM bw (conservative)."""
    if not total_bytes:
        return None
    dram_bw = facts.get("dram_bw_gbps")
    l1_bw = facts.get("l1_bw_gbps")
    bw = dram_bw
    if (memory or "") in ("l1_interleaved", "sharded") and l1_bw:
        bw = l1_bw
    if not bw:
        return None
    return (total_bytes / (bw * 1e9)) * 1e3


def annotate_op(op: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Attach ideal_ms / gap_ms / bound_by onto a single top_ops entry (in place).
    ideal = max(compute floor, memory floor) — the op can't beat its tightest tier.
    Matmul gets a compute floor (FLOPs); ANY op with byte info gets a memory floor."""
    facts = _facts(env)
    op_code = str(op.get("op_code", "")).lower()
    measured = float(op.get("device_ms") or 0.0)

    compute = None
    if any(t in op_code for t in _MATMUL_OPS):
        parsed = parse_matmul_shape(op.get("shape", ""))
        if parsed:
            m, k, n = parsed
            flops = matmul_flops(m, k, n) * int(op.get("count") or 1)  # device_ms is total over count
            compute = ideal_ms_compute(flops, op.get("fidelity", ""), facts)

    memory = ideal_ms_memory(float(op.get("bytes") or 0.0), op.get("memory", ""), facts)

    candidates = [(c, lbl) for c, lbl in ((compute, "compute"), (memory, "memory")) if c is not None]
    if candidates:
        ideal, bound_by = max(candidates, key=lambda t: t[0])
        op["ideal_ms"] = round(ideal, 4)
        op["gap_ms"] = round(max(0.0, measured - ideal), 4)
        op[
            "bound_by"
        ] = bound_by  # which floor dominates -> hints the knob (compute->grid/fidelity, memory->dtype/shard)
    else:
        op["ideal_ms"] = None
        op["gap_ms"] = None
        op["bound_by"] = None
    return op


def annotate_profile(profile: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Annotate every bucket's top_ops with ideal_ms/gap_ms/bound_by IN PLACE, and set
    bucket['gap_ms'] = Σ modeled gap (None if no op in the bucket is modeled). This is the
    hook ROUTE reads to rank buckets by ATTAINABLE speedup instead of raw device_ms."""
    for b in profile.get("buckets") or []:
        if b.get("id") == "host_overhead":
            b.setdefault("gap_ms", None)
            continue
        gaps = []
        for op in b.get("top_ops") or []:
            annotate_op(op, env)
            if op.get("gap_ms") is not None:
                gaps.append(op["gap_ms"])
        b["gap_ms"] = round(sum(gaps), 4) if gaps else None
    return profile


def compute_rooflines(profile: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Annotate every bucket's top_ops with ideal_ms/gap_ms and summarize.
    Returns {total_device_ms, modeled_device_ms, total_ideal_ms, total_gap_ms,
    ops:[...]} — ops sorted by gap_ms desc (the gap-driven work order)."""
    annotated: list[dict[str, Any]] = []
    for b in profile.get("buckets") or []:
        if b.get("id") == "host_overhead":
            continue
        for op in b.get("top_ops") or []:
            annotate_op(op, env)
            row = dict(op)
            row["bucket"] = b.get("id")
            annotated.append(row)
    modeled = [o for o in annotated if o.get("ideal_ms") is not None]
    total_ideal = round(sum(o["ideal_ms"] for o in modeled), 4)
    modeled_device = round(sum(float(o.get("device_ms") or 0.0) for o in modeled), 4)
    total_gap = round(sum(o["gap_ms"] for o in modeled), 4)
    annotated.sort(key=lambda o: (o.get("gap_ms") is None, -(o.get("gap_ms") or 0.0)))
    return {
        "total_device_ms": round(profile.get("device_ms", 0.0), 4),
        "modeled_device_ms": modeled_device,  # device_ms of ops we have a roofline for
        "total_ideal_ms": total_ideal,  # Σ floors of modeled ops -> the gap-driven target
        "total_gap_ms": total_gap,
        "modeled_op_count": len(modeled),
        "unmodeled_op_count": len(annotated) - len(modeled),
        "ops": annotated,
    }
