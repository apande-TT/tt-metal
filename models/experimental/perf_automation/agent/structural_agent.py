"""structural-edit sub-agent — applies a COORDINATED multi-op optimization
(sharding / full-grid / trace) for levers tagged `lever_type: structural`, where a
single config kwarg is not enough. Gets the per-op fingerprints + op->source
attribution, is fenced to the executed files, and self-verifies by re-reading.
Ground truth stays the deterministic GATE_PCC + REMEASURE. Same SDK seam as
edit_agent.make_edit_runner."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


STRUCTURAL_TEMPLATE = (
    "You are applying ONE *structural* performance optimization to a TTNN model, then stopping.\n\n"
    "Optimization (lever '{lever}'):\n{section}\n\n"
    "Hottest individual ops in this bucket:\n{top_ops}\n\n"
    "WHERE the hot ops are EMITTED (op->source attribution — edit THESE exact source lines,\n"
    "they are ranked by how much matmul work they emit; the top one is where the dominant\n"
    "matmuls actually execute — NOT necessarily the 'obvious' FFN/attention stub):\n{hot_sources}\n\n"
    "Edit ONLY these files (the executed path; Read them first):\n{files}\n\n"
    "THIS IS A COORDINATED, MULTI-SITE CHANGE — a single config kwarg is NOT enough:\n"
    "  1. Convert the INPUT activation tensor to a sharded L1 `memory_config` BEFORE the op\n"
    "     (e.g. `ttnn.to_memory_config(x, sharded_cfg)` / `ttnn.create_sharded_memory_config(...)`).\n"
    "     Setting only a `program_config` on a tensor that is still DRAM_INTERLEAVED does NOTHING\n"
    "     — the kernel graph is unchanged and the edit is inert. Sharding is a property of the\n"
    "     TENSOR, not the matmul call.\n"
    "  2. Give the op the matching `program_config` + full core grid\n"
    "     (`device.compute_with_storage_grid_size()` — never hard-code the grid).\n"
    "  3. Keep the OUTPUT sharded so the next op consumes it without a reshard back to DRAM.\n\n"
    "FORCE-TRY: make the full change; do NOT ship a partial 'safe' version (that is the inert\n"
    "no-op above). The downstream PCC gate will catch any correctness regression — rely on it.\n\n"
    "SELF-VERIFY before finishing (Read-only — do NOT run the model/device):\n"
    "  - Re-Read each file you edited and CONFIRM your edit added a tensor `memory_config`\n"
    "    conversion (to_memory_config / create_sharded_memory_config / interleaved_to_sharded)\n"
    "    ON THE EXECUTED CALL PATH (the method the forward actually runs), not merely a\n"
    "    `program_config=` kwarg and not in a dead/unused helper. If it only added a\n"
    "    program_config, or sits in code the forward doesn't call, the edit is INERT — fix it.\n\n"
    "When done, output exactly ONE JSON object and nothing else:\n"
    '  {{"files": [<repo-relative paths you changed>], "summary": <one sentence on the coordinated change>}}'
)


def _format_top_ops(top_ops: list[dict] | None) -> str:
    if not top_ops:
        return "  (per-op detail unavailable — target the bucket's dominant op)"
    lines = []
    for o in top_ops:
        extra = ""
        if o.get("ideal_ms") is not None:  # ROOFLINE evidence: how far off the hardware floor + which wall
            extra = f"  ROOFLINE: ideal={o.get('ideal_ms')}ms GAP={o.get('gap_ms')}ms " f"bound_by={o.get('bound_by')}"
        lines.append(
            f"  - {o.get('op_code','?')} [{o.get('shape','?')}] ×{o.get('count','?')}: "
            f"{o.get('device_ms',0):.3f}ms total, {o.get('cores','?')} cores ({o.get('grid','?')}), "
            f"mem={o.get('memory','?')}, fidelity={o.get('fidelity','?')}{extra}"
        )
    return "\n".join(lines)


# FROM-PRINCIPLES template: used when a hot bucket has NO playbook lever (conv/scan/moe/
# other/kv_cache/etc.). No prescribed recipe — give the agent the roofline gap + a MENU of
# ttnn primitives and make it DIAGNOSE from the gap which wall it's hitting, then close it.
# This is the model-agnostic, think-don't-pattern-match path.
FROM_PRINCIPLES_TEMPLATE = (
    "You are optimizing ONE hot operation on Tenstorrent hardware from FIRST PRINCIPLES,"
    " then stopping. There is no pre-written recipe — you must diagnose and decide.\n\n"
    "Hottest ops in this bucket (with ROOFLINE gap = how far above the hardware floor, and"
    " bound_by = which wall they hit):\n{top_ops}\n\n"
    "WHERE they are emitted (op->source attribution — edit THESE exact lines):\n{hot_sources}\n\n"
    "Edit ONLY these files (the executed path; Read them first):\n{files}\n\n"
    "DIAGNOSE from the gap + bound_by, then apply the matching ttnn primitive(s):\n"
    "  - bound_by=memory (DRAM/L1 bandwidth): shrink bytes moved -> lower-precision dtype"
    " (bfloat8_b/bfloat4_b weights via dtype/weights_dtype), shard the input into L1"
    " (ttnn.to_memory_config to a sharded config), keep outputs sharded to avoid reshards,"
    " remove redundant tilize/untilize/typecast.\n"
    "  - bound_by=compute (FLOPs): the op is far from its FLOP ceiling because the grid is"
    " under-occupied (cores << full grid) or fidelity is higher than needed -> occupy the"
    " full core grid (device.compute_with_storage_grid_size(), matching program_config) and"
    " drop MATH_FIDELITY to the lowest the PCC gate tolerates.\n"
    "  - tiny op / huge gap with near-zero ideal (dispatch/launch bound): FUSE it into a"
    " neighbor (activation into the matmul, elementwise chains, norm+residual) so it stops"
    " being a separate dispatched kernel.\n\n"
    "RULES: change the TENSOR's memory_config/dtype, not merely a program_config kwarg on a"
    " DRAM tensor (that is a no-op). Make the real change on the EXECUTED call path; the PCC"
    " gate will catch any correctness regression so FORCE-TRY rather than shipping a safe"
    " no-op. Read your edit back to confirm it adds a real device op (a new memory_config /"
    " dtype / fused op), not just a kwarg.\n\n"
    "When done, output exactly ONE JSON object and nothing else:\n"
    '  {{"files": [<repo-relative paths you changed>], "summary": <one sentence on what you did and why>}}'
)


def build_structural_prompt(
    lever: str, section: str, model_files: list, top_ops: list[dict] | None, hot_sources: list[dict] | None = None
) -> str:
    from .op_attribution import format_hot_sources

    files = "\n".join(f"  - {f}" for f in model_files)
    if not (section or "").strip():  # FROM_PRINCIPLES: no playbook section -> think from the roofline gap
        return FROM_PRINCIPLES_TEMPLATE.format(
            top_ops=_format_top_ops(top_ops),
            hot_sources=format_hot_sources(hot_sources or []),
            files=files,
        )
    return STRUCTURAL_TEMPLATE.format(
        lever=lever or "(unspecified)",
        section=section,
        top_ops=_format_top_ops(top_ops),
        hot_sources=format_hot_sources(hot_sources or []),
        files=files,
    )


def make_structural_runner(
    env_agent_path: str | Path = Path(__file__).parent.parent / ".env.agent",
    max_turns: int = 60,
) -> Callable[..., dict]:
    """Build the structural editor: runner(lever, section, model_files, top_ops) -> result.

    Mirrors edit_agent.make_edit_runner but (self-verify by re-reading, no device run),
    a bigger turn budget (coordinated edit + verify loop), and the structural model tier.
    Result parsing is LENIENT: APPLY uses git-diff as ground truth, so a missing/!JSON
    final message yields files=[] rather than raising.
    """
    from .config import apply_agent_env, get_model

    resolved = apply_agent_env(env_agent_path)
    model = get_model("structural", resolved)

    def runner(
        *,
        lever: str,
        section: str,
        model_files: list,
        top_ops: list[dict] | None = None,
        hot_sources: list[dict] | None = None,
        error: str | None = None,
        spec: dict | None = None,
        cwd: str | None = None,
    ) -> dict:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        from .probes import _extract_json_object, _usage_summary

        files = [str(p) for p in model_files]
        prompt = build_structural_prompt(lever, section, files, top_ops, hot_sources)
        if error:  # REPAIR: prepend the failure, keep the structural protocol
            prompt = f"Your previous structural edit for '{lever}' FAILED:\n{error}\n\n" + prompt

        opts: dict = dict(
            model=model,
            system_prompt=(
                "You apply exactly one STRUCTURAL optimization (sharding / grid / layout) to "
                "TTNN model source using Read, Edit, Grep, Glob. A structural edit is multi-site: "
                "you MUST change the tensor's memory_config, not just a program_config, and it must "
                "land on the EXECUTED call path. Self-verify by RE-READING (do NOT run the model or "
                "device — that deadlocks the profiler that holds it). Your FINAL message must be one "
                "JSON object, no prose. Stay inside the working directory."
            ),
            allowed_tools=["Read", "Edit", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            setting_sources=[],
            max_turns=max_turns,
            max_buffer_size=50 * 1024 * 1024,
        )
        if cwd:
            opts["cwd"] = cwd
        options = ClaudeAgentOptions(**opts)
        chunks: list[str] = []
        usage: dict = {}

        async def _go() -> None:
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            chunks.append(block.text)
                elif isinstance(msg, ResultMessage):
                    usage["u"] = _usage_summary(msg)

        from .sdk_retry import run_with_retry

        run_with_retry(_go, lambda: (chunks.clear(), usage.clear()))

        response = "\n".join(chunks)
        files_out, summary = [], ""
        try:  # LENIENT: APPLY falls back to git-diff if files is empty
            import json

            obj = json.loads(_extract_json_object(response))
            if isinstance(obj, dict):
                files_out = [str(f) for f in (obj.get("files") or [])]
                summary = str(obj.get("summary", ""))
        except Exception:
            pass
        return {
            "files": files_out,
            "summary": summary,
            "model": model,
            "usage": usage.get("u"),
            "prompt": prompt,
            "response": response,
        }

    return runner
