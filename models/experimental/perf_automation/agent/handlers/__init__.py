"""Handler registry — the ONE shared file where members wire in their work.

To integrate: replace a `mocks.X` entry with your real module's handler, ONE
line at a time, and re-run `tests/test_engine.py`. The engine and every other
handler stay untouched.

  REAL today:  BEFORE_LOOP_DONE, ROUTE, SELECT, PLAN, APPLY, VERIFY (M1)
               GATE_PCC, REMEASURE, DECIDE, COMMIT, REVERT, LOG, CHECK_EXIT (M2)
  MOCK (TODO): REPAIR_CODE, REPAIR_PCC  (M1)
"""

from __future__ import annotations

from .. import states
from . import apply as _apply
from . import log_exit, mocks, plan as _plan, route
from . import select as _select
from . import gate_pcc as _gate_pcc
from . import decide as _decide
from . import remeasure as _remeasure
from . import verify as _verify
from . import commit as _commit
from . import revert as _revert


def build_handlers() -> dict:
    return {
        states.BEFORE_LOOP_DONE: lambda ctx: states.ROUTE,
        # --- Member 1: decide & act ---
        states.ROUTE: route.route,  # REAL
        states.SELECT: _select.select,  # REAL
        states.PLAN: _plan.plan,  # REAL
        states.APPLY: _apply.apply,  # REAL
        states.VERIFY: _verify.verify,  # REAL
        states.REPAIR_CODE: mocks.repair_code,
        states.REPAIR_PCC: mocks.repair_pcc,
        # --- Member 2: evaluate & record ---
        states.GATE_PCC: _gate_pcc.gate_pcc,  # REAL
        states.REMEASURE: _remeasure.remeasure,  # REAL
        states.DECIDE: _decide.decide,  # REAL
        states.COMMIT: _commit.commit,  # REAL
        states.REVERT: _revert.revert,  # REAL
        states.LOG: log_exit.log,  # REAL
        states.CHECK_EXIT: log_exit.check_exit,  # REAL
    }
