"""Env-compat gate: pin-respecting remedy + mesh-aware grid checks.

Regression tests for the 2026-07 fix to the `ENVIRONMENT INCOMPATIBLE`
gate. Two bugs were fixed:

  1. The remedy path treated source-level issues as a transformers
     version mismatch and let an LLM propose a `pip install` that
     DOWNGRADED transformers BELOW the repo's pin (5.10.2 -> 4.49.0),
     landing on a version too old to even load the target model. The
     gate must never go below the repo pin; if the installed version
     already matches the pin, remaining problems are source-level and
     pip cannot help.

  2. The dispatch-core grid checks (`find_grid`, `_dispatch_safe_grid`,
     hard-coded grid literals) fired for every target, even a canonical
     single-chip 1x1 mesh (e.g. p150) where the risk they describe
     (dispatch-core placement on NON-canonical meshes) does not apply.
"""

from __future__ import annotations


# ─── pin parsing ───────────────────────────────────────────────────


def test_pinned_transformers_version_matches_requirements() -> None:
    """The pin helper must read the exact `transformers == X` version
    from tt_metal/python_env/requirements-dev.txt (currently 5.10.2)."""
    from scripts.tt_hw_planner.cli import _pinned_transformers_version

    assert _pinned_transformers_version() == "5.10.2"


# ─── remedy logic (pure, no pip / no re-exec) ──────────────────────


def test_remedy_installs_pin_when_env_drifted_below() -> None:
    """Env downgraded below the pin -> restore the pinned version."""
    from scripts.tt_hw_planner.cli import _env_gate_remedy

    action, target = _env_gate_remedy(installed="4.49.0", pinned="5.10.2")
    assert action == "install_pin"
    assert target == "5.10.2"


def test_remedy_never_proposes_below_pin_even_from_newer() -> None:
    """Even from a newer version, the remedy targets the pin exactly —
    never a version below it."""
    from scripts.tt_hw_planner.cli import _env_gate_remedy

    action, target = _env_gate_remedy(installed="5.20.0", pinned="5.10.2")
    assert action == "install_pin"
    assert target == "5.10.2"


def test_remedy_source_abort_when_on_pin() -> None:
    """Already on the pinned version -> any remaining problem is
    source-level; pip must NOT run."""
    from scripts.tt_hw_planner.cli import _env_gate_remedy

    action, _ = _env_gate_remedy(installed="5.10.2", pinned="5.10.2")
    assert action == "source_abort"


def test_remedy_source_abort_when_pin_unknown() -> None:
    """No pin discoverable -> be conservative: treat as source, never
    guess a pip downgrade."""
    from scripts.tt_hw_planner.cli import _env_gate_remedy

    action, _ = _env_gate_remedy(installed="5.10.2", pinned=None)
    assert action == "source_abort"


# ─── mesh-aware grid checks ────────────────────────────────────────


def _grid_problems(problems):
    return [
        p
        for p in problems
        if "model_config.py" in p and ("_dispatch_safe_grid" in p or "find_grid" in p or "hard-codes a grid" in p)
    ]


def test_grid_checks_fire_by_default() -> None:
    """Sanity: on this checkout (model_config.py lacks _dispatch_safe_grid)
    the grid checks fire when no mesh context is given."""
    from scripts.tt_hw_planner.cli import _check_demo_environment_compat

    _ok, problems = _check_demo_environment_compat()
    assert _grid_problems(problems), "grid checks should fire by default on this repo"


def test_grid_checks_skipped_for_single_chip_1x1_mesh() -> None:
    """A canonical single-chip 1x1 mesh (e.g. p150) must NOT trip the
    dispatch-core grid checks — their risk is scoped to non-canonical
    meshes."""
    from scripts.tt_hw_planner.cli import _check_demo_environment_compat

    _ok, problems = _check_demo_environment_compat(mesh="1,1")
    assert not _grid_problems(problems), (
        "grid checks must be skipped for a single-chip 1x1 mesh; " f"got: {_grid_problems(problems)}"
    )


def test_grid_checks_still_fire_for_multichip_mesh() -> None:
    """Multi-chip / non-canonical meshes keep the grid checks."""
    from scripts.tt_hw_planner.cli import _check_demo_environment_compat

    _ok, problems = _check_demo_environment_compat(mesh="1,8")
    assert _grid_problems(problems), "grid checks must still fire for a multi-chip mesh"
