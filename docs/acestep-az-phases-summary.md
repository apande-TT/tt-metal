# ACE-Step A→Z TTNN — Condensed Plan

**Goal:** Text prompt + reference WAV → generated music (WAV) on **full TT hot path** (TT Qwen3-Embedding + TT DiT + TT VAE decode).

**Model reference:** [awesome-ace-step](https://github.com/ace-step/awesome-ace-step)

**Principle:** **Qwen3 is already in tt-metal** (`tt_transformers`) — wire ACE-Step weights/heads; do not greenfield port. Do **not** block on one mega `auto-up --stack` or trace/LM for first full-TT WAV.

**Two finish lines:**
- **Phase 5** — full TT e2e: TT text + TT DiT + TT VAE (`demo_acestep_az.py`); host ref encode only
- **Phase 7** — production + **`acestep-5Hz-lm-1.7B`** on TT (same Qwen3 stack)

**Efficient order (remaining work):** **4 → 3 → 2C → 5 → 7 → 6/8**

```
TT Qwen3-Embedding ─┐
host ref VAE encode ┴→ TT DiT (+ CFG) → TT Oobleck decode → WAV
(optional Phase 7: TT 5Hz LM replaces Call B)
```

**Trace + 2-CQ:** Phase 6 (DiT), Phase 8 (full stack + LM). **< 2 s e2e vs A100** = Phase 8.

**Device:** p150 1×1 · `flock /tmp/tt_ace_device.lock`

| Phase | Deliverable | Parallel? | Gate |
|-------|-------------|-----------|------|
| **1** | TT latents → host VAE WAV | No | ✅ done |
| **2** | Live prompt + ref (host text) | 2A ∥ 2B | ✅ done |
| **2C** | TT Qwen3-Embedding-0.6B | After 4+3 OK | PCC vs host; pipeline flag |
| **3** | 30 steps + CFG + APG | Code ∥ 4 | Quality vs HF @ 30 steps |
| **4** | TT Oobleck decode PCC ≥ 0.99 | **Priority 1** | TT VAE WAV ≈ host |
| **5** | Full TT demo CLI | After 2C+3+4 | TT text+DiT+VAE; listenable WAV |
| **6** | Trace + perf (DiT) | No | Traced DiT e2e; timing table |
| **7** | 5Hz LM (Qwen3 ×3) | ∥ 6 | 1.7B on TT; 0.6B/4B validated |
| **8** | Trace + perf (full stack) | No | < 2 s target |

**Full plan:** `docs/superpowers/plans/2026-07-03-acestep-az-phases.md`
**Q&A (incl. auto-up scope):** `docs/acestep-session-qa-2026-07-03.md`
**Progress:** `docs/acestep-az-progress.md`

**Board reset if PCIe errors:** `tt-smi -r 0`
