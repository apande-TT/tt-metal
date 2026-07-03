# ACE-Step A→Z — Progress Log

Started: 2026-07-03

## Status

| Phase | Status | Commit | Notes |
|-------|--------|--------|-------|
| 1 G0 Baseline | ✅ soft PASS | — | tests 1–3 PASS; trace removed from gate |
| 2 Live inputs | ✅ done | — | text_encode + ref audio + pipeline wired; CPU+device tests pass |
| 3 Prod sampler | 🔄 prep done | — | apg_guidance.py + cfg helpers; loop wiring pending |
| 4 TT VAE | 🔄 code fixes | — | oobleck_layers fixes; device PCC gate pending |
| 5 Full A→Z demo | 🔄 skeleton | — | demo_acestep_az.py; live inputs wired |
| 6 Trace + perf | ⏳ deferred | — | last; use `docs/acestep-m0-trace-run.sh` |

## Revisit / blockers

- **Trace capture hang (Phase 6 only):** hangs at `capturing trace...` — do **not** block Phases 2–5. Debug in `m0-trace` window.
- **Device reset:** after killing hung trace test, run `tt-smi -r 0` before device tests.

## Plan update (2026-07-03)

- Decoupled trace from critical path per `docs/acestep-az-phases-summary.md`
- Phase 1 gate = G0 only (3 tests, no traced perf)
- Phase 6 added for trace + 2-CQ + A100 perf signoff

## Session log

### 2026-07-03 — Phase 1 kickoff

- Created tmux session `acestep-az` (7 windows)
- Docs: `docs/acestep-az-phases-summary.md`, `docs/acestep-az-progress.md`
- Killed stale pytest on device lock
- tmux Phase 1 script killed on test 1 — retry running in background

### 2026-07-03 — Session restart (post-reboot)

- Machine rebooted ~21:44; tmux session lost
- Recreated 7-window layout in `acestep-az`
- Board reset: `tt-smi -r 0`
- Phase 1 alternate path restarted: `bash docs/acestep-az-phase1-run.sh` (device window)
- Log: `/tmp/acestep_phase1.log`
- G0 tests 1–3 PASS; test 4/4 (traced perf) hung at trace capture → **removed from Phase 1 gate**

### 2026-07-03 — Plan refactor (trace last)

- Updated plan: Phases 1–5 functional (`traced=False`); Phase 6 = trace + perf
- Phase 1 script now 3 tests only; added `docs/acestep-m0-trace-run.sh`
- Killed stalled pytest (trace capture hang); device lock FREE
- tmux: added `m0-trace` window; refreshed all window banners

### 2026-07-03 — Parallel agents deployed

- 5 parallel agents: 2A, 2B, 3 prep, 4, 5 prep (no shared files)
- Integration agent: pipeline_acestep.py + test_e2e_live_inputs_acestep.py
- CPU tests pass; device live-input test pass (~77s)
- tmux windows tail `/tmp/acestep_agent_*.log` via docs/acestep-az-window-watch.sh
- See docs/acestep-az-agent-deploy.md
