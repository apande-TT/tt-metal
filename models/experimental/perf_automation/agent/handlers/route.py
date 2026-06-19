"""ROUTE handler (Member 1 / lead) — REAL, deterministic. No agent, no device.

Picks the slowest bucket of the CURRENT profile, asks the router which playbook
levers are tagged for it, and appends a **route brief row** to route_briefs.jsonl — candidate metadata,
model map, and full extracted section texts. SELECT reads that row back (by
route_brief_id) as its decision material.

In:  ctx.current_profile() buckets + ctx.index (playbook).
Out: ctx.state["current_bucket"], ["candidates"] (ids), ["route_brief_id"]. -> SELECT
"""

from __future__ import annotations

from typing import Any, Callable

from .. import exec_scope, model_map, router, states


def _select_bucket(buckets: list[dict[str, Any]], exhausted: set[str], metric: str = "device_ms") -> dict[str, Any]:
    """Top remaining bucket by ms. For the device-floor metric, the host_overhead
    bucket is NOT routable (optimizing host can't move device_ms); for a wall/host
    metric it competes like any other (wall = device + host, biggest share wins)."""
    pool = [b for b in buckets if b["id"] not in exhausted]
    if metric == "device_ms":
        pool = [b for b in pool if b["id"] != "host_overhead"]
    pool = pool or [b for b in buckets if b["id"] != "host_overhead"] or buckets
    # GAP-DRIVEN: prefer the bucket with the most ATTAINABLE speedup (Σ gap-to-roofline,
    # set by roofline.annotate_profile) over raw device_ms — a bucket already near its
    # hardware floor has little to give even if its absolute ms is large. Fall back to
    # device_ms whenever rooflines are unavailable (older profiles / annotate failed).
    if metric == "device_ms" and any(b.get("gap_ms") is not None for b in pool):
        return max(pool, key=lambda b: (b.get("gap_ms") or 0.0))
    return max(pool, key=lambda b: b.get("device_ms", 0.0))


def _bucket_query(tags: dict[str, Any]) -> dict[str, Any]:
    """Project a bucket's tags onto the router's 8 dimensions (drop anything else)."""
    return {k: v for k, v in tags.items() if k in router.DIMENSIONS}


def build_route_brief(
    bucket: dict[str, Any], hits: list[dict[str, Any]], read_section: Callable[[str], str], skeleton: str = ""
) -> dict[str, Any]:
    """Assemble JSON decision material: bottleneck + candidates + section texts."""
    sections = []
    for h in hits:
        try:
            text = read_section(h["id"])
        except KeyError:
            text = ""
        sections.append({"id": h["id"], "file": h["file"], "title": h["title"], "text": text})
    return {
        "row_type": "route_brief",
        "bucket": bucket,
        "top_ops": bucket.get("top_ops", []),
        "candidate_count": len(hits),
        "candidates": [
            {
                "id": h["id"],
                "lever_type": h.get("lever_type", ""),
                "file": h["file"],
                "title": h["title"],
            }
            for h in hits
        ],
        "model_map": skeleton,
        "sections": sections,
    }


def route(ctx) -> str:
    exec_scope.ensure_scope(ctx)  # one-time: restrict edit targets to the executed path
    profile = ctx.current_profile()
    # Annotate buckets with gap-to-roofline so _select_bucket can rank by ATTAINABLE
    # speedup. Best-effort: any failure leaves gap_ms unset and routing falls back to device_ms.
    try:
        from .. import roofline

        env = (
            (ctx.manifest or {}).get("env", {})
            if isinstance(ctx.manifest, dict)
            else getattr(ctx, "manifest", {}).get("env", {})
        )
        roofline.annotate_profile(profile, env or {})
    except Exception as exc:  # roofline is an optimization, never a hard dependency
        ctx.log_event(states.ROUTE, "warn", f"roofline annotate skipped: {exc}")
    metric_name = (ctx.state.get("metric") or {}).get("name", "device_ms")
    all_ids = {b["id"] for b in (profile.get("buckets") or [])}
    exhausted = set(ctx.state.get("exhausted_buckets", []))
    # Pick the highest-gap (else slowest) bucket. If the playbook has a matching lever,
    # use it (the prior). If NOT — instead of skipping the bucket (which left conv/scan/
    # moe/other un-optimized and made the tool transformer-only) — emit the FROM_PRINCIPLES
    # sentinel so APPLY routes the bucket's hottest op to the THINKING structural agent.
    # The sentinel is a normal candidate: it gets tried, then marked tried, so the bucket
    # still exhausts and the loop advances (no livelock).
    bucket = _select_bucket(profile["buckets"], exhausted, metric_name)
    query = _bucket_query(bucket.get("tags", {}))
    hits = router.route(ctx.index, query)
    candidates = [h["id"] for h in hits]
    if not candidates:
        ctx.log_event(states.ROUTE, "info", f"bucket '{bucket['id']}' has no playbook lever -> from-principles only")
    # Off-menu (from-principles) is ALWAYS a standing option, not just an empty-bucket
    # fallback. The brain may reason from first principles even when the bucket HAS levers
    # but none targets the dominant op (e.g. datamove dominated by Tilize, which no
    # shard/dtype lever addresses). The playbook is a PRIOR, not a requirement.
    if states.FROM_PRINCIPLES not in candidates:
        candidates = candidates + [states.FROM_PRINCIPLES]

    ctx.state["current_bucket"] = bucket["id"]
    ctx.state["candidates"] = candidates
    # Per-op targets for the structural-edit agent (the hottest individual ops in
    # this bucket); empty on older profiles that predate top_ops.
    ctx.state["top_ops"] = bucket.get("top_ops", [])

    # deterministic model map, filtered to this bucket's op_class — where its ops live
    op_class = bucket.get("tags", {}).get("op_class")
    subs = model_map.OP_CLASS_SUBSTRINGS.get(op_class)
    try:
        mm = model_map.build_model_map(ctx.model_files(), root=ctx.model_root())
        skeleton = model_map.render_skeleton(mm, op_substrings=subs)
    except Exception:
        skeleton = ""

    # append structured decision material to the single route_briefs.jsonl stream;
    # SELECT reads its row back by route_brief_id (state below points at it).
    from ..events import append_jsonl

    route_brief_id = f"{ctx.run.run_id}:{ctx.state.get('iteration', 0)}:ROUTE"
    payload = build_route_brief(bucket, hits, router.read_section, skeleton)
    payload.update(
        {
            "route_brief_id": route_brief_id,
            "run_id": ctx.run.run_id,
            "iteration": ctx.state.get("iteration", 0),
            "stage": states.ROUTE,
            "query": query,
        }
    )
    append_jsonl(ctx.run.dir / "route_briefs.jsonl", payload)
    ctx.state["route_brief_id"] = route_brief_id

    ctx.log_event(states.ROUTE, "info", f"bucket={bucket['id']} candidates={len(hits)} brief_id={route_brief_id}")
    return states.SELECT
