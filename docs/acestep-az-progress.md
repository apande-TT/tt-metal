# ACE-Step A→Z — Progress Log

Started: 2026-07-03

## Status

| Phase | Status | Commit | Notes |
|-------|--------|--------|-------|
| 1 G0 Baseline | ✅ soft PASS | — | tests 1–3 PASS; trace removed from gate |
| 2 Live inputs | ✅ done | — | host text_encode + ref audio + pipeline |
| 2C TT text encode | ✅ done | — | Qwen3-Embedding-0.6B via `tt_transformers`; PCC ≥0.99 |
| 3 Prod sampler | ✅ done | — | 30-step CFG + APG; `/tmp/az_prod_30s_cfg7.wav` |
| 4 TT VAE | ✅ done | — | PCC gate + e2e TT VAE waveform test |
| 5 Full TT A→Z demo | ✅ signoff run | — | `bash docs/acestep-az-phase5-run.sh` → `/tmp/az_phase5_signoff.wav` |
| 6 Trace + perf (DiT) | ⏳ deferred | — | DiT stack only; use `docs/acestep-m0-trace-run.sh` |
| 7 5Hz LM planner | ✅ 7C signoff | `24de6caa37` + logits squeeze | `bash docs/acestep-az-phase7c-run.sh` — 3 passed 2026-07-04 |
| 8 Trace + perf (full) | ✅ smoke signoff | traced DiT + TT LM + TT VAE | `bash docs/acestep-az-phase8-run.sh` — e2e 70.2s @ 8s/4 steps |

## Revisit / blockers

- **Trace capture hang (Phase 6 only):** hangs at `capturing trace...` — do **not** block Phases 2–5. Debug in `m0-trace` window.
- **Device reset:** after killing hung trace test, run `tt-smi -r 0` before device tests.

**Agent resume guide:** `docs/acestep-az-agent-resume.md`

## Plan update (2026-07-03)

- Decoupled trace from critical path per `docs/acestep-az-phases-summary.md`
- Phase 1 gate = G0 only (3 tests, no traced perf)
- Phase 6 = DiT trace + 2-CQ + A100 perf signoff
- **Phase 7 added:** all three **`acestep-5Hz-lm-{0.6B,1.7B,4B}`** planners ([awesome-ace-step](https://github.com/ace-step/awesome-ace-step)) — reuse **`tt_transformers` Qwen3**; `--lm-model` selects variant
- Phase 8 = full-stack trace + perf (< 2 s with LM on TT)
- **Efficient full-TT plan:** reuse `tt_transformers` Qwen3; remaining order **4 → 3 → 2C → 5 → 7**; Phase 5 = TT text + TT DiT + TT VAE (host ref encode only)

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

### 2026-07-04 — Phase 7C device signoff PASS

- Fix: squeeze TT prefill logits `[B,1,V]` → `[B,V]` in `lm_planner_tt._next_token_from_logits`
- Gate: `ACESTEP_RUN_PHASE7C=1 bash docs/acestep-az-phase7c-run.sh` — **3 passed**, 3 skipped (~212s)
- Phase 8 traced full-stack gate started: `bash docs/acestep-az-phase8-run.sh`

### 2026-07-04 — Phase 8 traced full-stack PASS (smoke)

- Fix: set `quantized = lm_quantized` in traced+LM path (`pipeline.py`)
- Gate: `ACESTEP_RUN_PHASE8=1 bash docs/acestep-az-phase8-run.sh` — **1 passed** (~108s)
- Output: `/tmp/az_phase8_traced_lm.wav` — `PHASE8_SIGNOFF e2e_s=70.23 latent_gen_s=33.80 vae_decode_s=36.43`
- Perf target (<2s vs A100) not met; functional traced stack verified

- 5 parallel agents: 2A, 2B, 3 prep, 4, 5 prep (no shared files)
- Integration agent: pipeline_acestep.py + test_e2e_live_inputs_acestep.py
- CPU tests pass; device live-input test pass (~77s)
- tmux windows tail `/tmp/acestep_agent_*.log` via docs/acestep-az-window-watch.sh
- See docs/acestep-az-agent-deploy.md
