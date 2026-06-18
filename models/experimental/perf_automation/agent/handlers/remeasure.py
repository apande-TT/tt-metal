"""REMEASURE handler (PLAN 8.7) — REAL median + variance + iter profile.

Re-profiles the edited model (measurement injectable via ctx.deps["measure_runner"]),
takes the MEDIAN device_ms over the runs, records the run-to-run spread (for the
deferred noise-floor decision), and writes the re-bucketed iter profile that COMMIT
promotes to current_profile. A measurement crash is infra (post-PCC) -> discard
reason measure_failed (the one crash path that does NOT go to REPAIR).
"""

from __future__ import annotations

import json
import statistics

from .. import states
from ..opclass import STRUCTURAL_OP_CLASSES
from ..probes import PerfRunFailed


def _op_count(profile):
    return sum(int(b.get("count", 0)) for b in (profile.get("buckets") or []))


def _comparable(baseline, iter_profile, tol=0.25):
    """Is the iter profile structurally comparable to the baseline? Guards
    against trusting a partial/garbage capture (e.g. tracy logging 27 ops
    instead of 308 -> a false 22x 'win'). Returns (ok, reason)."""
    b_ops = _op_count(baseline)
    if b_ops == 0:
        return True, None  # no baseline op count -> nothing to compare against
    i_ops = _op_count(iter_profile)
    ratio = i_ops / b_ops
    if not (1 - tol) <= ratio <= (1 + tol):
        return False, f"op_count_mismatch: iter {i_ops} vs baseline {b_ops} ops ({ratio:.2f}x)"
    bbuckets = baseline.get("buckets") or []
    if bbuckets:
        dom = max(bbuckets, key=lambda b: b.get("device_ms", 0)).get("id")
        iter_ids = {b.get("id") for b in (iter_profile.get("buckets") or [])}
        if dom and dom not in iter_ids:
            return False, f"dominant_bucket_missing: baseline '{dom}' absent in iter profile"
    # Per-bucket vanish: a STRUCTURAL compute class (matmul/attention/embedding/conv) that
    # ran in the baseline but runs ZERO times now is a partial/crashed capture, not a win --
    # you cannot make a model do zero matmuls. This catches what the total-count (+/-25%) and
    # dominant-bucket checks miss: a LOW-count but essential op (a single SDPA, count 1)
    # silently dropping to 0 while the total stays within tolerance (the nemotron
    # attn-score-dtype bug: attention 1->0, total 3443->3436 = 0.998x, banked as a -2.2% win).
    icounts = {b.get("id"): int(b.get("count", 0)) for b in (iter_profile.get("buckets") or [])}
    for b in baseline.get("buckets") or []:
        bid = b.get("id")
        if bid in STRUCTURAL_OP_CLASSES and int(b.get("count", 0)) > 0 and icounts.get(bid, 0) == 0:
            return False, (
                f"structural_bucket_vanished: '{bid}' ran {int(b.get('count', 0))}x in baseline, "
                f"0x now -- partial/crashed capture, not an optimization"
            )
    return True, None


def _same_op_graph(before: dict, after: dict) -> bool:
    """True iff two profiles have a byte-identical op-class signature (same
    buckets, same op counts, same per-bucket device_ms to 4 dp). Device kernel
    time is deterministic, so an EFFECTIVE edit always shifts at least one
    bucket's count or time; if nothing moves, the edited code was never
    exercised by the perf workload (the edit targets dead/un-run code)."""

    def sig(p):
        return sorted(
            (str(b.get("id")), int(b.get("count", 0)), round(float(b.get("device_ms", 0.0)), 4))
            for b in (p.get("buckets") or [])
            if b.get("id")
            != "host_overhead"  # host time is non-deterministic; only the device op graph defines identity
        )

    bsig, asig = sig(before), sig(after)
    return bool(bsig) and bsig == asig


def _op_delta_evidence(before: dict, after: dict) -> str:
    """Per-bucket count/time before->after, as measured ground truth for the
    inert-repair agent. The agent has no device; this is the ONLY way it learns
    whether its edit changed the op graph (re-reading the file can't tell it)."""

    def by_id(p):
        return {b.get("id"): b for b in (p.get("buckets") or []) if b.get("id") != "host_overhead"}

    bb, ab = by_id(before), by_id(after)
    rows = []
    for bid in sorted(set(bb) | set(ab)):
        bc = bb.get(bid, {}).get("count", 0)
        ac = ab.get(bid, {}).get("count", 0)
        bm = round(float(bb.get(bid, {}).get("device_ms", 0.0)), 3)
        am = round(float(ab.get(bid, {}).get("device_ms", 0.0)), 3)
        tag = "" if (bc == ac and bm == am) else "  <-- CHANGED"
        rows.append(f"  {bid:12s} count {bc}->{ac}   device_ms {bm}->{am}{tag}")
    return "\n".join(rows)


def remeasure(ctx) -> str:
    before = ctx.state["metric"]["current"]
    runner = ctx.deps.get("measure_runner") or _default_runner()

    try:
        profiles = runner(ctx)
    except PerfRunFailed as exc:  # the EDIT crashed the perf run (device-op TT_FATAL) -> repairable
        # Not a flaky measurement: the edit produced code that crashes at runtime (tracy exits 0
        # even so, leaving a partial CSV that would be misread as op_count_mismatch). Route to
        # REPAIR_CODE with the real device error so the agent fixes its own edit — exactly like a
        # GATE_PCC crash. Capped by MAX_CODE_FIX; on exhaustion, discard (edit_failed) -> REVERT.
        ctx.state["last_verdict"] = {"status": "crash", "error": exc.error}
        if ctx.state.get("code_fix_attempts", 0) < states.MAX_CODE_FIX:
            ctx.log_event(states.REMEASURE, "warn", f"perf run crashed (repairable): {exc.error}")
            return states.REPAIR_CODE
        ctx.state["last_decision"] = {
            "result": "discard",
            "reason": "edit_failed",
            "before": before,
            "error": exc.error,
        }
        ctx.log_event(states.REMEASURE, "warn", f"perf run crashed, repair budget exhausted: {exc.error}")
        return states.REVERT
    except Exception as exc:  # infra flake, not an edit bug
        ctx.state["last_decision"] = {
            "result": "discard",
            "reason": "measure_failed",
            "before": before,
            "error": str(exc),
        }
        ctx.log_event(states.REMEASURE, "warn", f"measure failed: {exc}")
        return states.REVERT
    if not profiles:
        ctx.state["last_decision"] = {"result": "discard", "reason": "measure_failed", "before": before}
        return states.REVERT

    # Track the CONFIGURED metric (device_ms | wall_ms | host_ms), not always device.
    # wall_ms/host_ms let the loop optimize generation-loop wins (trace/2-CQ/bucketing)
    # that don't move the device floor. Unknown metrics (fps/throughput) fall back to device.
    metric_name = (ctx.state.get("metric") or {}).get("name", "device_ms")
    vals = [p.get(metric_name, p["device_ms"]) for p in profiles]
    median_val = statistics.median(vals)
    after = round(median_val, 4)
    spread = round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0
    rep = min(profiles, key=lambda p: abs(p.get(metric_name, p["device_ms"]) - median_val))  # representative profile

    rel = f"profiles/iter_{ctx.state.get('iteration', 0):02d}_profile.json"
    (ctx.run.dir / rel).write_text(json.dumps(rep, indent=2, sort_keys=True))

    # comparability guard: a profile structurally unlike the baseline (op count
    # collapsed, dominant bucket vanished) is an untrustworthy capture, not a win.
    measurement_ok, measurement_reason = True, None
    try:
        measurement_ok, measurement_reason = _comparable(ctx.baseline_profile(), rep)
    except Exception:  # baseline unreadable -> skip the guard, don't block
        pass

    # op_graph_identical => edit_inert ONLY for the device-floor metric. A trace /
    # 2-CQ / bucketed-decode edit LEGITIMATELY leaves the device op graph byte-identical
    # while improving wall/host time, so flagging it inert would wrongly discard a real
    # generation-loop win. For wall/host metrics, let DECIDE judge on the measured number.
    op_graph_identical = False
    op_delta = None
    if metric_name == "device_ms":
        try:
            op_graph_identical = _same_op_graph(ctx.current_profile(), rep)
            op_delta = _op_delta_evidence(ctx.current_profile(), rep)  # measured proof for the inert-repair agent
        except Exception:  # missing/odd profile -> don't false-flag
            pass

    ctx.state["last_decision"] = {
        "before": before,
        "after": after,
        "spread": spread,
        "runs": len(vals),
        "pcc": (ctx.state.get("last_verdict") or {}).get("pcc"),
        "profile": rel,
        "measurement_ok": measurement_ok,
        "measurement_reason": measurement_reason,
        "op_graph_identical": op_graph_identical,
        "op_delta": op_delta,
    }
    if not measurement_ok:
        ctx.log_event(states.REMEASURE, "warn", f"profile not comparable to baseline: {measurement_reason}")
    _counts = {b.get("id"): b.get("count") for b in (rep.get("buckets") or [])}
    ctx.log_event(
        states.REMEASURE,
        "info",
        f"after={after} spread={spread} runs={len(vals)} "
        f"counts(matmul={_counts.get('matmul')},datamove={_counts.get('datamove')},eltwise={_counts.get('eltwise')})",
    )
    return states.DECIDE


def _default_runner():
    from ..measure import measure_runs

    return measure_runs
