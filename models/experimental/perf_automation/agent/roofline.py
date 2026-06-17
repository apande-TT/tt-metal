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
    DRAM-resident -> DRAM bw (mesh-aggregate when box_facts scaled it). L1/sharded ->
    L1 bw IF a peak is known; we do NOT have a sourced L1 peak, so this deliberately
    falls back to DRAM bw — which OVERSTATES an L1 op's floor (L1 is faster), i.e. a
    SAFE/conservative floor (never claims headroom that isn't there). Calibrate l1_bw_gbps
    later to tighten it."""
    if not total_bytes:
        return None
    dram_bw = facts.get("dram_bw_gbps")
    l1_bw = facts.get("l1_bw_gbps")  # intentionally usually absent -> conservative DRAM fallback below
    bw = dram_bw
    if (memory or "") in ("l1_interleaved", "sharded") and l1_bw:
        bw = l1_bw
    if not bw:
        return None
    return (total_bytes / (bw * 1e9)) * 1e3


def dispatch_floor_per_op(profile: dict[str, Any]) -> float | None:
    """Self-calibrated per-op DEVICE floor = the smallest observed per-op device_ms across
    all hot ops. A trivial op's device time is ~pure kernel-launch/granularity overhead, so
    this is the unavoidable cost every op pays regardless of FLOPs/bytes. Without it, a bucket
    of thousands of tiny ops gets ideal~0 and a fake huge gap, mis-ranking it #1 against an
    unreachable target. MEASURED from the profile (no hardcoded latency constant)."""
    pers: list[float] = []
    for b in profile.get("buckets") or []:
        if b.get("id") == "host_overhead":
            continue
        for op in b.get("top_ops") or []:
            c, d = int(op.get("count") or 0), float(op.get("device_ms") or 0.0)
            if c > 0 and d > 0:
                pers.append(d / c)
    return min(pers) if pers else None


def annotate_op(op: dict[str, Any], env: dict[str, Any], dispatch_per_op: float | None = None) -> dict[str, Any]:
    """Attach ideal_ms / gap_ms / bound_by onto a single top_ops entry (in place).
    ideal = max(compute, memory, dispatch) — the op can't beat its tightest floor. Matmul
    gets a compute floor (FLOPs); any op with byte info gets a memory floor; every op gets a
    dispatch floor (count × per-op launch overhead) when dispatch_per_op is supplied."""
    facts = _facts(env)
    op_code = str(op.get("op_code", "")).lower()
    measured = float(op.get("device_ms") or 0.0)
    count = int(op.get("count") or 1)

    compute = None
    if any(t in op_code for t in _MATMUL_OPS):
        parsed = parse_matmul_shape(op.get("shape", ""))
        if parsed:
            m, k, n = parsed
            flops = matmul_flops(m, k, n) * count  # device_ms is total over count
            compute = ideal_ms_compute(flops, op.get("fidelity", ""), facts)

    memory = ideal_ms_memory(float(op.get("bytes") or 0.0), op.get("memory", ""), facts)
    dispatch = (dispatch_per_op * count) if dispatch_per_op else None

    candidates = [
        (c, lbl) for c, lbl in ((compute, "compute"), (memory, "memory"), (dispatch, "dispatch")) if c is not None
    ]
    if candidates:
        ideal, bound_by = max(candidates, key=lambda t: t[0])
        op["ideal_ms"] = round(ideal, 4)
        op["gap_ms"] = round(max(0.0, measured - ideal), 4)
        op[
            "bound_by"
        ] = bound_by  # which floor dominates -> the knob class (compute->grid/fidelity, memory->dtype/shard, dispatch->fuse/trace)
    else:
        op["ideal_ms"] = None
        op["gap_ms"] = None
        op["bound_by"] = None
    return op


def annotate_profile(profile: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Annotate every bucket's top_ops with ideal_ms/gap_ms/bound_by IN PLACE, and set
    bucket['gap_ms'] = Σ modeled gap (None if no op in the bucket is modeled). This is the
    hook ROUTE reads to rank buckets by ATTAINABLE speedup instead of raw device_ms."""
    disp = dispatch_floor_per_op(profile)  # self-calibrated per-op launch floor
    for b in profile.get("buckets") or []:
        if b.get("id") == "host_overhead":
            b.setdefault("gap_ms", None)
            continue
        gaps = []
        for op in b.get("top_ops") or []:
            annotate_op(op, env, disp)
            if op.get("gap_ms") is not None:
                gaps.append(op["gap_ms"])
        b["gap_ms"] = round(sum(gaps), 4) if gaps else None
    return profile


def compute_rooflines(profile: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """Annotate every bucket's top_ops with ideal_ms/gap_ms and summarize.
    Returns {total_device_ms, modeled_device_ms, total_ideal_ms, total_gap_ms,
    ops:[...]} — ops sorted by gap_ms desc (the gap-driven work order)."""
    disp = dispatch_floor_per_op(profile)  # self-calibrated per-op launch floor
    annotated: list[dict[str, Any]] = []
    for b in profile.get("buckets") or []:
        if b.get("id") == "host_overhead":
            continue
        for op in b.get("top_ops") or []:
            annotate_op(op, env, disp)
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


_AT_FLOOR_ABS_MS = 0.05  # gap within measurement noise
_AT_FLOOR_FRAC = 0.10  # ...or within 10% of the floor -> effectively at the ttnn floor


def residual_report(profile: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """END-OF-RUN certification: how far the final profile is from its ttnn-reachable
    floor, and where the remaining gap lives. This is what lets the tool SAY 'nothing
    ttnn-reachable is left' (every hot op at_floor) vs merely 'ran out of levers'.

    For each hot op: measured vs ideal_ms, gap, gap%, bound_by, at_floor. Summary gives
    the residual gap + the biggest open op. Ops still OPEN (gap above the floor) with a
    known bound_by are the ttnn-reachable work the tool should keep attacking; an op whose
    remaining gap needs a new kernel is the honest out-of-scope residual — distinguishing
    those two automatically needs an op catalog (follow-on), so v1 reports gap + cause and
    flags at_floor, leaving the reachable-vs-kernel call to the agent/human."""
    r = compute_rooflines(profile, env)
    rows = []
    for o in r["ops"]:
        ideal, gap = o.get("ideal_ms"), o.get("gap_ms")
        if ideal is None:
            at_floor = None
        else:
            at_floor = gap <= _AT_FLOOR_ABS_MS or gap <= _AT_FLOOR_FRAC * ideal
        rows.append(
            {
                "bucket": o.get("bucket"),
                "op_code": o.get("op_code"),
                "shape": o.get("shape"),
                "count": o.get("count"),
                "device_ms": o.get("device_ms"),
                "ideal_ms": ideal,
                "gap_ms": gap,
                "gap_pct": (round(100.0 * gap / ideal, 1) if ideal else None),
                "bound_by": o.get("bound_by"),
                "at_floor": at_floor,
            }
        )
    modeled = [x for x in rows if x["ideal_ms"] is not None]
    open_ops = [x for x in modeled if x["at_floor"] is False]
    open_ops.sort(key=lambda x: -(x["gap_ms"] or 0.0))
    return {
        "total_device_ms": r["total_device_ms"],
        "modeled_floor_ms": r["total_ideal_ms"],  # Σ ttnn-reachable floor of modeled ops
        "residual_gap_ms": r["total_gap_ms"],  # Σ measured - floor still on the table
        "at_floor": len(open_ops) == 0 and len(modeled) > 0,  # nothing ttnn-reachable left (modeled)
        "n_open": len(open_ops),
        "n_modeled": len(modeled),
        "n_unmodeled": r["unmodeled_op_count"],
        "open_ops": open_ops[:10],  # the ttnn-reachable work still on the table, biggest gap first
        "rows": rows,
    }
