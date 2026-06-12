"""Bucket diversification + bottleneck-specific lever ranking."""

import json

from agent import router
from agent.handlers import route
from agent.handlers.log_exit import log
from agent.loop_context import LoopContext
from agent.run import Run


def _lever(anchor, **dims):
    entry = {dim: [router.WILDCARD] for dim in router.DIMENSIONS}
    entry["id"] = anchor
    for key, value in dims.items():
        entry[key] = value
    return entry


def test_rank_surfaces_grid_specific_lever_first():
    fidelity_lever = _lever("norm-fidelity", fidelity=["hifi4"])
    grid_lever = _lever("fnd-core-grid", op_class=["reduction"], grid=["partial", "tiny"])
    query = {"op_class": "reduction", "bound": "slow", "grid": "tiny", "fidelity": "hifi4"}
    ranked = route._rank_hits([fidelity_lever, grid_lever], query)
    assert ranked[0]["id"] == "fnd-core-grid"


def _log_ctx(tmp_path):
    run = Run.create(tmp_path / "runs", config={"config": {}, "pathmap": {}}, run_id="X")
    run.state_path.write_text(
        json.dumps(
            {
                "state": "LOG",
                "metric": {"name": "device_ms", "direction": "min", "current": 100.0},
                "current_bucket": "reduction",
                "candidates": ["l1", "l2", "l3", "l4", "l5"],
                "tried": [],
            }
        )
    )
    return LoopContext.from_run(run, index=[])


def test_log_exhausts_bucket_after_repeated_no_gain(tmp_path):
    ctx = _log_ctx(tmp_path)
    for lever in ("l1", "l2", "l3"):
        ctx.state["selected_lever"] = lever
        ctx.state["last_decision"] = {"result": "discard", "reason": "no_gain", "before": 100.0, "after": 100.0}
        log(ctx)
    assert ctx.state["bucket_misses"]["reduction"] == 3
    assert "reduction" in ctx.state.get("exhausted_buckets", [])


def test_log_keep_resets_misses_and_does_not_exhaust(tmp_path):
    ctx = _log_ctx(tmp_path)
    for lever in ("l1", "l2"):
        ctx.state["selected_lever"] = lever
        ctx.state["last_decision"] = {"result": "discard", "reason": "no_gain", "before": 100.0, "after": 100.0}
        log(ctx)
    ctx.state["selected_lever"] = "l3"
    ctx.state["last_decision"] = {"result": "keep", "reason": None, "before": 100.0, "after": 90.0}
    log(ctx)
    assert ctx.state["bucket_misses"]["reduction"] == 0
    assert "reduction" not in ctx.state.get("exhausted_buckets", [])
