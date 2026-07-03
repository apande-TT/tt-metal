# ACE-Step A→Z TTNN — Condensed Plan

**Goal:** Text prompt + reference WAV → generated music (WAV). **Trace + 2-CQ and < 2 s e2e vs A100 come last** (Phase 6).

**Device:** p150 1×1 · serialize with `flock /tmp/tt_ace_device.lock`

**Principle:** Functional correctness first (`traced=False`). Do **not** block Phases 2–5 on trace capture stability.

| Phase | Deliverable | Parallel? | Gate |
|-------|-------------|-----------|------|
| **1** | TT latents → host VAE WAV (no trace) | No | Listen WAV; tests 1–3 pass |
| **2** | Live prompt + ref audio (not captures) | 2A text ∥ 2B ref | Prompt/ref change output |
| **3** | 30 steps + CFG + APG (`traced=False`) | Code ∥ Phase 4 | Quality vs HF @ 30 steps |
| **4** | TT Oobleck VAE decode PCC ≥ 0.99 | File work ∥ 2–3 | TT VAE WAV ≈ host |
| **5** | Full CLI demo (prompt + ref → music) | No | One-shot demo; listenable output |
| **6** | Trace + 2-CQ + perf vs A100 | No | Traced e2e stable; timing table |

**Per phase:** sanity tests → **commit + push if pass** → next phase.

**If stuck on device:** reset board → retry; else skip to parallel-safe file work; log in `docs/acestep-az-progress.md`.

**If stuck on trace (Phase 6 only):** do not hold the device window — continue Phases 2–5 with `traced=False`.

**tmux attach:** `tmux attach -t acestep-az`

**Progress log:** `docs/acestep-az-progress.md`

**Phase 1 gate (device window):** `bash docs/acestep-az-phase1-run.sh`

**Phase 6 trace debug (optional window `m0-trace`):** `bash docs/acestep-m0-trace-run.sh`

**Board reset if PCIe errors:** `tt-smi -r 0`
