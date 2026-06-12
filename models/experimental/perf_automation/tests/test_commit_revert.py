"""COMMIT / REVERT handlers (PLAN 8.9) — real git against a throwaway repo.

No API key, no hardware. Each test builds a tiny git repo with a model dir, points
the manifest's model_root at it, and drives the handler the way DECIDE would.
"""

import json
import subprocess

from agent import gitio, states
from agent.handlers.commit import commit
from agent.handlers.revert import revert
from agent.loop_context import LoopContext
from agent.run import Run


def _init_repo(d):
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    model = d / "models" / "demo"
    model.mkdir(parents=True)
    (model / "stub.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=d, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=d, check=True)
    return d, model


def _ctx(tmp_path, repo, model, state):
    run = Run.create(
        tmp_path / "runs",
        config={"config": {"model_root": str(model)}, "pathmap": {}},
        run_id="CR",
    )
    run.state_path.write_text(json.dumps(state))
    return LoopContext.from_run(run, index=[])


def _base_state(sha, result, **extra):
    s = {
        "state": "COMMIT",
        "iteration": 3,
        "selected_lever": "qkv-fuse",
        "git_sha_clean": sha,
        "metric": {"name": "device_ms", "unit": "ms", "direction": "min", "current": 12.0},
        "tried": [],
        "last_decision": {"result": result, "before": 12.0, "after": 11.0},
    }
    s["last_decision"].update(extra)
    return s


def test_revert_restores_model_dir_to_clean_sha(tmp_path):
    repo, model = _init_repo(tmp_path)
    sha = gitio.head_sha(repo)
    (model / "stub.py").write_text("x = 999  # discarded edit\n")
    ctx = _ctx(tmp_path, repo, model, _base_state(sha, "discard", reason="no_gain"))

    assert revert(ctx) == states.LOG
    assert (model / "stub.py").read_text() == "x = 1\n"  # rolled back


def test_revert_is_scoped_unrelated_wip_survives(tmp_path):
    repo, model = _init_repo(tmp_path)
    sha = gitio.head_sha(repo)
    (repo / "wip.py").write_text("unrelated = True\n")  # WIP outside the model dir
    subprocess.run(["git", "-C", str(repo), "add", "wip.py"], check=True)  # tracked WIP edit
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "wip"], check=True)
    (repo / "wip.py").write_text("unrelated = False\n")  # uncommitted WIP change
    (model / "stub.py").write_text("x = 999\n")
    # git_sha_clean is the PRE-wip sha; a repo-wide reset would clobber wip.py.
    ctx = _ctx(tmp_path, repo, model, _base_state(sha, "discard"))

    assert revert(ctx) == states.LOG
    assert (model / "stub.py").read_text() == "x = 1\n"  # model restored
    assert (repo / "wip.py").read_text() == "unrelated = False\n"  # WIP untouched


def test_revert_no_sha_is_safe_noop(tmp_path):
    repo, model = _init_repo(tmp_path)
    state = _base_state(None, "discard")
    state["git_sha_clean"] = None
    ctx = _ctx(tmp_path, repo, model, state)
    assert revert(ctx) == states.LOG  # logs + continues, no crash


def test_commit_persists_edit_and_advances_clean_sha(tmp_path):
    repo, model = _init_repo(tmp_path)
    sha = gitio.head_sha(repo)
    (model / "stub.py").write_text("x = 2  # kept edit\n")
    ctx = _ctx(tmp_path, repo, model, _base_state(sha, "keep", profile="profiles/iter_03_profile.json"))

    assert commit(ctx) == states.LOG
    new_sha = gitio.head_sha(repo)
    assert new_sha != sha  # a commit happened
    assert ctx.state["git_sha_clean"] == new_sha  # next REVERT target advanced
    assert ctx.state["current_profile"] == "profiles/iter_03_profile.json"  # promoted
    # the kept edit is committed: nothing pending under the model dir vs the new HEAD
    # (whole-repo is_clean would be False here only because the test's runs/ dir lives
    #  inside tmp_path; that is unrelated to what COMMIT persisted).
    assert gitio.changed_files(repo, new_sha, pathspec="models/demo") == []


def test_commit_untracked_model_is_safe_noop(tmp_path):
    # model dir not under git at all -> nothing to commit, loop must not crash
    repo, model = _init_repo(tmp_path)
    sha = gitio.head_sha(repo)
    untracked = repo / "untracked_model"
    untracked.mkdir()
    (untracked / "stub.py").write_text("x = 1\n")
    state = _base_state(sha, "keep")
    ctx = _ctx(tmp_path, repo, untracked, state)
    assert commit(ctx) == states.LOG  # no crash; nothing committed
