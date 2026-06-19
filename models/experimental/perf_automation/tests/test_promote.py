"""Promote / learning loop — should_promote, prompt, provisional-lever write + graduate (no hw)."""

from pathlib import Path

from agent import states
from agent.promote import (
    build_promote_prompt,
    graduate_lever,
    promote_win,
    should_promote,
    write_provisional_lever,
)


class _Ctx:
    def __init__(
        self, lever, decision, *, top_ops=None, bucket="datamove", last_diff="@@ -1 +1 @@\n-x\n+y", root="nemotron"
    ):
        self.state = {
            "selected_lever": lever,
            "last_decision": decision,
            "current_bucket": bucket,
            "top_ops": top_ops or [{"op_code": "Tilize", "shape": "1024x6144"}],
            "last_diff": last_diff,
        }
        self.deps = {}
        self._root = root

    def model_root(self):
        return Path(self._root)

    def current_profile(self):
        return {"buckets": [{"id": "datamove", "tags": {"op_class": "datamove"}}]}


_WIN = {"result": "keep", "before": 90.7, "after": 88.0, "pcc": 0.999}


def test_should_promote_only_kept_offmenu_faster():
    assert should_promote(_Ctx(states.FROM_PRINCIPLES, _WIN)) is True
    # a known lever win is NOT promoted (only off-menu discoveries become new levers)
    assert should_promote(_Ctx("shard-activation-to-l1", _WIN)) is False
    # off-menu but not faster -> not promoted
    assert should_promote(_Ctx(states.FROM_PRINCIPLES, {"result": "keep", "before": 90.0, "after": 91.0})) is False


def test_prompt_carries_evidence_and_route_format():
    from agent.promote import _win_from_ctx

    p = build_promote_prompt(_win_from_ctx(_Ctx(states.FROM_PRINCIPLES, _WIN)))
    assert "datamove" in p and "Tilize" in p and "88.0" in p  # bucket, hot op, measured win
    assert "op_class:" in p and "lever_type: structural" in p and "{#" in p  # must ask for a route block


def test_write_provisional_lever_is_router_indexable(tmp_path):
    section = "## Learned: datamove coherence {#learned-datamove-coherence-x}\n<!-- route\nop_class: datamove\nlever_type: structural\n-->\n\n**Fires when:** redundant tilize.\nEmit the producer's tile layout."
    path = write_provisional_lever(section, "datamove-coherence-x", tmp_path, "nemotron")
    assert path.exists() and path.name == "LEARNED_datamove-coherence-x.md"
    text = path.read_text()
    assert "provisional: true" in text and "learned_on: nemotron" in text and "{#learned-datamove-coherence-x}" in text
    # the router's build_index globs *.md -> the learned lever is picked up
    from agent.router import build_index

    idx = build_index(tmp_path)
    assert any(e["id"] == "learned-datamove-coherence-x" for e in idx)


def test_graduate_flips_provisional(tmp_path):
    p = write_provisional_lever(
        "## x {#x}\n<!-- route\nop_class: datamove\nlever_type: structural\n-->\nbody", "x", tmp_path, "nemotron"
    )
    graduate_lever(p, "seamless")
    text = p.read_text()
    assert "provisional: false" in text and "graduated_on: seamless" in text


def test_promote_win_writes_lever_with_mock_runner(tmp_path):
    ctx = _Ctx(states.FROM_PRINCIPLES, _WIN)
    section = "## Learned: datamove coherence {#learned-dm}\n<!-- route\nop_class: datamove\nlever_type: structural\n-->\n\n**Fires when:** redundant tilize churn.\nMake the producer emit tile layout."
    ctx.deps["promote_runner"] = lambda prompt: section
    path = promote_win(ctx, guidelines_dir=tmp_path)
    assert path and path.exists() and "LEARNED_" in path.name


def test_promote_win_rejects_junk_section(tmp_path):
    # a response with no route anchor isn't a valid lever -> nothing written
    ctx = _Ctx(states.FROM_PRINCIPLES, _WIN)
    ctx.deps["promote_runner"] = lambda prompt: "sorry, I cannot do that"
    assert promote_win(ctx, guidelines_dir=tmp_path) is None


def test_promote_win_skips_non_offmenu(tmp_path):
    ctx = _Ctx("shard-activation-to-l1", _WIN)
    ctx.deps["promote_runner"] = lambda prompt: "## x {#x}\n<!-- route\nop_class: datamove\n-->\nbody"
    assert promote_win(ctx, guidelines_dir=tmp_path) is None  # only off-menu wins promote
