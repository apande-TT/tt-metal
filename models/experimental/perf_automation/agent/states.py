"""Canonical state names + loop constants for the Agent Loop (PLAN section 8).

Single source of truth so handlers never typo a transition. Every handler
returns one of these strings; the engine (agent/engine.py) dispatches on it.
"""

from __future__ import annotations

# Entry (written by the Before Loop) ----------------------------------------
BEFORE_LOOP_DONE = "BEFORE_LOOP_DONE"

# Outer loop: decide & act (Member 1) ----------------------------------------
ROUTE = "ROUTE"
SELECT = "SELECT"
PLAN = "PLAN"
APPLY = "APPLY"
VERIFY = "VERIFY"
REPAIR_CODE = "REPAIR_CODE"
REPAIR_PCC = "REPAIR_PCC"

# Evaluate & record (Member 2) -----------------------------------------------
GATE_PCC = "GATE_PCC"
REMEASURE = "REMEASURE"
DECIDE = "DECIDE"
COMMIT = "COMMIT"
REVERT = "REVERT"
LOG = "LOG"
CHECK_EXIT = "CHECK_EXIT"

# Terminals ------------------------------------------------------------------
DONE = "DONE"
STOPPED = "STOPPED"
FAILED = "FAILED"
TERMINAL = frozenset({DONE, STOPPED, FAILED})

# Repair budgets (decided 2026-06-11) ----------------------------------------
MAX_CODE_FIX = 5  # parse / import / run-crash repairs before ABANDON
MAX_CODE_FIX_PRINCIPLES = 8  # off-menu invents WHAT *and* places WHERE in one budget -> larger
MAX_PCC_FIX = 2  # PCC-below-threshold repairs before DISCARD
MAX_INERT_RETRY = 6  # off-path (edit_inert) retries before giving up steering a lever
JUDGE_STREAK_THRESHOLD = 3  # consecutive measured no-gains in a bucket before the agentic waste-judge weighs in
MAX_STRUCT_FIX = 3  # FIXER: re-invoke the structural agent on an inert (op-graph-unchanged) shard, up to N times

# Sentinel lever id used when a hot bucket has NO matching playbook lever: instead of
# skipping the bucket (which left conv/scan/moe/other un-optimized), ROUTE emits this as
# the candidate and APPLY routes it to the THINKING structural agent to optimize the
# bucket's hottest op from first principles (roofline gap + primitive menu). This is the
# model-agnostic path — the playbook becomes a prior, not a requirement.
FROM_PRINCIPLES = "auto-principles"


def code_fix_budget(lever: str | None) -> int:
    """Repair budget for the selected lever. From-principles (off-menu) must INVENT the fix
    (WHAT) and PLACE it (WHERE) within one budget, whereas a known lever only places a proven
    recipe — so off-menu gets a larger budget so WHAT-discovery doesn't starve WHERE-placement.
    (Phase-1 of the two-phase 'B' design; the diagnose/place split is a later refinement.)"""
    return MAX_CODE_FIX_PRINCIPLES if lever == FROM_PRINCIPLES else MAX_CODE_FIX


# Reference transition map (documentation + a test can assert handlers conform).
# Conditional edges (verdicts, counters) are decided INSIDE the handler; this
# lists every state a handler may legally return.
TRANSITIONS = {
    BEFORE_LOOP_DONE: [ROUTE],
    ROUTE: [SELECT],
    SELECT: [PLAN],
    PLAN: [APPLY, REVERT],
    APPLY: [VERIFY],
    VERIFY: [GATE_PCC, REPAIR_CODE, REVERT],
    REPAIR_CODE: [VERIFY],
    REPAIR_PCC: [VERIFY],
    GATE_PCC: [REMEASURE, REPAIR_PCC, REPAIR_CODE, REVERT],
    REMEASURE: [DECIDE, REVERT, REPAIR_CODE],  # REPAIR_CODE = the edit crashed the perf run (device-op TT_FATAL)
    DECIDE: [COMMIT, REVERT, APPLY],  # APPLY = FIXER: iterate on an inert structural shard
    COMMIT: [LOG],
    REVERT: [LOG],
    LOG: [CHECK_EXIT],
    CHECK_EXIT: [ROUTE, DONE, STOPPED],
}
