from agent import states
from agent.handlers import decide as decide_mod
from agent.handlers.remeasure import _comparable


def _prof(buckets):
    return {"buckets": buckets}


def test_comparable_ok():
    base = _prof([{"id": "matmul", "count": 96, "device_ms": 6.7}, {"id": "reduction", "count": 50, "device_ms": 2.0}])
    it = _prof([{"id": "matmul", "count": 94, "device_ms": 6.0}, {"id": "reduction", "count": 50, "device_ms": 2.0}])
    assert _comparable(base, it) == (True, None)


def test_comparable_opcount_collapse():
    base = _prof([{"id": "matmul", "count": 96, "device_ms": 6.7}, {"id": "reduction", "count": 50, "device_ms": 2.0}])
    it = _prof([{"id": "other", "count": 1, "device_ms": 0.1}, {"id": "datamove", "count": 9, "device_ms": 0.09}])
    ok, reason = _comparable(base, it)
    assert not ok and "op_count_mismatch" in reason


def test_comparable_dominant_missing():
    base = _prof([{"id": "matmul", "count": 50, "device_ms": 6.7}, {"id": "eltwise", "count": 50, "device_ms": 0.1}])
    it = _prof([{"id": "eltwise", "count": 95, "device_ms": 0.1}])
    ok, reason = _comparable(base, it)
    assert not ok and "dominant_bucket_missing" in reason


def test_comparable_structural_bucket_vanished():
    # The nemotron attn-score-dtype bug: the single SDPA op (attention, count 1) crashes out,
    # total op count barely moves (3443->3436 = 0.998x, within +/-25%) and the dominant bucket
    # (datamove) is still present -> the old guard PASSED it as a -2.2% "win". Must now reject.
    base = _prof(
        [
            {"id": "datamove", "count": 1742, "device_ms": 40.0},
            {"id": "matmul", "count": 559, "device_ms": 30.0},
            {"id": "eltwise", "count": 1124, "device_ms": 20.0},
            {"id": "attention", "count": 1, "device_ms": 0.7},
        ]
    )
    it = _prof(
        [
            {"id": "datamove", "count": 1740, "device_ms": 39.0},
            {"id": "matmul", "count": 557, "device_ms": 29.0},
            {"id": "eltwise", "count": 1123, "device_ms": 20.0},
            # attention vanished (1 -> 0)
        ]
    )
    ok, reason = _comparable(base, it)
    assert not ok and "structural_bucket_vanished" in reason and "attention" in reason


def test_comparable_fusable_bucket_drop_not_rejected():
    # A LEGIT fusion can zero out a FUSABLE class (e.g. all reductions fused into matmul
    # epilogues). That must NOT be rejected -- only structural classes are guarded.
    base = _prof([{"id": "matmul", "count": 96, "device_ms": 6.7}, {"id": "reduction", "count": 20, "device_ms": 2.0}])
    it = _prof([{"id": "matmul", "count": 96, "device_ms": 6.0}])  # reduction fused away
    assert _comparable(base, it) == (True, None)


class _Ctx:
    def __init__(self, last_decision, direction="min"):
        self.state = {"last_decision": last_decision, "metric": {"direction": direction}}
        self.events = []

    def log_event(self, *a):
        self.events.append(a)


def test_decide_discards_untrusted_measurement():
    ctx = _Ctx({"before": 12.10, "after": 0.55, "measurement_ok": False, "measurement_reason": "op_count_mismatch: x"})
    nxt = decide_mod.decide(ctx)
    assert ctx.state["last_decision"]["result"] == "discard"
    assert "op_count_mismatch" in ctx.state["last_decision"]["reason"]
    assert nxt == states.REVERT


def test_decide_keeps_real_gain():
    ctx = _Ctx({"before": 12.10, "after": 11.50, "measurement_ok": True})
    nxt = decide_mod.decide(ctx)
    assert ctx.state["last_decision"]["result"] == "keep" and nxt == states.COMMIT


def test_decide_flags_suspicious_but_keeps():
    ctx = _Ctx({"before": 12.10, "after": 3.0, "measurement_ok": True})
    decide_mod.decide(ctx)
    d = ctx.state["last_decision"]
    assert d["result"] == "keep" and d.get("suspicious_gain") is not None
