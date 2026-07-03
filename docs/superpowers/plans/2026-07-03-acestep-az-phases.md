# ACE-Step v1.5 — Phased execution plan (A→Z on TTNN)

**Your goal:** Give **text prompt + reference audio** → get **generated music (WAV)**.

**Perf stretch goal (Phase 6, last):** **trace + 2-CQ** on hot TT paths, **end-to-end time vs A100 (< 2 s full stack)**.

**Hardware:** p150 Blackhole, 1×1 mesh. **One device job at a time** — use `flock /tmp/tt_ace_device.lock`.

**Diagram (status):** `docs/acestep_v15_e2e_dataflow_ttnn_status.png`

**Updated:** 2026-07-03 — trace decoupled from Phases 1–5; functional path uses `traced=False`.

---

## Two tracks

| Track | Phases | Blocks A→Z demo? |
|-------|--------|------------------|
| **Functional** | 1 → 2 → 3 → 4 → 5 | No — this is the main line |
| **Trace / perf** | 6 (M0) | Only blocks < 2 s signoff |

Do **not** wait on trace capture to start Phase 2 file work or non-traced device tests.

---

## Phase overview

| Phase | Name | You get at the end | Device? | Trace? | Parallel? |
|-------|------|--------------------|---------|--------|-----------|
| **1** | Functional baseline | TT latents → host VAE WAV | Yes (serial) | **No** | No |
| **2** | Live inputs | Real prompt + ref WAV (not captures) | Mixed | No | **Yes** (2A ∥ 2B) |
| **3** | Production sampler | 30 ODE steps + CFG + APG | Yes | No | Code ∥ Phase 4 |
| **4** | TT VAE decode | Oobleck decoder on TT, PCC ≥ 0.99 | Yes | No | **Yes** (file work during 2–3) |
| **5** | Full A→Z demo | CLI: prompt + ref → music | Yes | No | No |
| **6** | Trace + perf | trace + 2-CQ stable; timing vs A100 | Yes | **Yes** | No |

**Rule:** Finish each phase **manual gate** before starting the next **on the functional track**. File-level work across phases can overlap where the table says parallel.

---

## Shared setup (every tmux window)

```bash
export REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
PY="$REPO/python_env/bin/python"

# Check device lock before any pytest
flock -n /tmp/tt_ace_device.lock echo FREE || echo BUSY
```

**Default for Phases 1–5 device tests:** `traced=False` (explicit in pipeline call or test fixture).

---

## tmux session layout

```bash
tmux new-session -d -s acestep-az -n phase1
tmux new-window -t acestep-az -n phase2a-text
tmux new-window -t acestep-az -n phase2b-ref
tmux new-window -t acestep-az -n phase3-sampler
tmux new-window -t acestep-az -n phase4-vae
tmux new-window -t acestep-az -n phase5-demo
tmux new-window -t acestep-az -n device      # functional flock pytest (Phases 1–5)
tmux new-window -t acestep-az -n m0-trace    # Phase 6 only — never blocks 2–5
tmux attach -t acestep-az
```

- **device** — Phases 1–5 integration tests (`traced=False`).
- **phase2a-text, phase2b-ref, phase4-vae** — implementation + CPU/golden tests (no device lock).
- **m0-trace** — trace hang debug + `docs/acestep-m0-trace-run.sh` (optional, after G0 or idle).

---

## Phase 1 — Functional baseline (G0)

**Objective:** Prove TT latents → **listenable WAV** via host VAE. **No trace required.**

**Why first:** Confirms Calls A/B/D/C + host glue work on device before wiring live inputs.

**Not in scope:** Traced e2e, trace capture, perf baseline — moved to **Phase 6**.

### Tasks

| # | Task | Owner window | Needs device |
|---|------|--------------|--------------|
| 1.1 | Kill stale pytest/flock; confirm lock FREE | device | — |
| 1.2 | Host VAE golden (`-k golden_latents`) | device | No |
| 1.3 | Eager e2e latents (`test_e2e_generate_audio.py`) | device | Yes |
| 1.4 | TT latents + host VAE (`-k tt_latents_host_vae`) | device | Yes |
| 1.5 | Listen to output WAV; log G0 pass in progress doc | phase1 | — |

### Commands (device window)

```bash
bash docs/acestep-az-phase1-run.sh
```

Or individually:

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio.py \
  -s -v --timeout=900

flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_with_vae_host.py \
  -k tt_latents_host_vae -s -v --timeout=900
```

### Manual gate (you)

- [ ] Listen: output WAV — sounds like music, not noise.
- [ ] All three G0 tests PASS.

**Exit → start Phase 2** (and Phase 4 file work in parallel).

---

## Phase 2 — Live inputs (text + reference audio)

**Objective:** Replace `_captured/` tensors with **your actual inputs**: caption/lyrics + reference WAV.

**Why second:** Core of “give prompt + ref → music.” Host VAE decode OK; gate step count (4) until Phase 3.

**Trace:** Not required. Integration tests use `traced=False`.

### Parallel tracks

```
Phase 2A (text)          Phase 2B (ref audio)
     │                         │
     └──────────┬──────────────┘
                ▼
         pipeline_acestep.py
    prompts + reference_audio → generate
```

### Phase 2A — Live text (tmux: `phase2a-text`)

| # | Task |
|---|------|
| 2A.1 | Add `text_encode.py`: ACE-Step processor + Qwen3-Embedding-0.6B (torch reference first) |
| 2A.2 | PCC test: TT Qwen embeddings vs torch on fixed prompts |
| 2A.3 | Extend `build_inputs(..., use_captured=False, prompts=..., lyrics=...)` |
| 2A.4 | Wire `pipeline_acestep.py`: stop ignoring `prompts` |

**Key files:** `models/tt_dit/pipelines/acestep/text_encode.py`, `pipeline_acestep.py`, `tt/common.py`

### Phase 2B — Live reference audio (tmux: `phase2b-ref`)

| # | Task |
|---|------|
| 2B.1 | Add `encode_reference_audio(wav_path)` in `vae_host.py` (torch Oobleck **encoder**) |
| 2B.2 | Map encoder output → `refer_audio_acoustic_hidden_states_packed`, `src_latents`, masks |
| 2B.3 | Add `reference_audio: str | Tensor` to pipeline API; set `is_covers=1` when ref present |
| 2B.4 | Golden test: fixed WAV vs HF `generate_audio` latents (PCC or statistical sanity) |

**Key files:** `vae_host.py`, `pipeline_acestep.py`, new `test_e2e_live_inputs_acestep.py`

### Integration (device window, after 2A + 2B code lands)

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_live_inputs_acestep.py \
  -s -v --timeout=1800   # add this test in Phase 2; traced=False
```

Optional demo:

```bash
flock /tmp/tt_ace_device.lock $PY -m \
  models.demos.hf_eager.acestep_v15_base.demo.demo_generate_audio_wav \
  --prompt "upbeat electronic track" --reference /path/to/ref.wav --output /tmp/live.wav
```

### Manual gate (you)

- [ ] Change **prompt text** → output audibly changes.
- [ ] Change **reference WAV** → timbre/style shifts (cover mode).
- [ ] No dependency on `_captured/` for a normal run.

**Exit → start Phase 3** (Phase 4 file work can already be running in parallel).

---

## Phase 3 — Production sampler (30 steps + CFG + APG)

**Objective:** Match **production inference** (quality): ~**30 ODE steps**, **CFG** (2× DiT forward/step), **APG/ADG** on host.

**Trace:** Implement CFG against **eager / non-traced** DiT first. Traced CFG batching → Phase 6.

### Tasks

| # | Task | Window |
|---|------|--------|
| 3.1 | Default `num_inference_steps=30` (keep `ACESTEP_E2E_INFER_STEPS=4` for fast CI) | phase3-sampler |
| 3.2 | Wire CFG via `models/tt_dit/pipelines/cfg.py` — batched cond+uncond DiT | phase3-sampler |
| 3.3 | Port APG/ADG from HF `apg_guidance.py` (host velocity math) | phase3-sampler |
| 3.4 | HF parity test @ 30 steps (quality gate) | device |
| 3.5 | *(Phase 6)* Extend traced decoder for CFG batch dim or dual replay | m0-trace |

### Device command

```bash
export ACESTEP_USE_TT_VAE=0 ACESTEP_SAVE_WAV=1
# num_inference_steps=30 via pytest param or env; traced=False
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_live_inputs_acestep.py \
  -k "prod" -s -v --timeout=7200   # add prod profile when ready
```

### Manual gate (you)

- [ ] 30-step output quality acceptable vs HF reference on same prompt+ref.

**Exit → Phase 5 prep; Phase 4 device gate can run anytime after 4.x code ready.**

---

## Phase 4 — TT VAE decode (Oobleck on device)

**Objective:** Move **latents → WAV** from host PyTorch to **TT Oobleck decoder**; PCC ≥ 0.99.

**Why parallel with 2–3:** Decoder port is independent of text/ref wiring and trace.

**Today:** `OOBLECK_DECODER_PORT_COMPLETE = False` — scaffold runs, tests skip.

### Tasks (tmux: `phase4-vae`)

| # | Task |
|---|------|
| 4.1 | Fix ConvTranspose1d padding parity vs torch |
| 4.2 | Fix dilated ResUnit center-crop boundaries |
| 4.3 | Pass `test_vae_oobleck_decoder.py` (PCC ≥ 0.99) |
| 4.4 | Set `OOBLECK_DECODER_PORT_COMPLETE = True` |
| 4.5 | Run full pipeline with `ACESTEP_USE_TT_VAE=1`, `traced=False` |

### Device command

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_vae_oobleck_decoder.py \
  -s -v --timeout=900

export ACESTEP_USE_TT_VAE=1 ACESTEP_SAVE_WAV=1
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_live_inputs_acestep.py \
  -s -v --timeout=3600   # full stack with TT VAE; traced=False
```

### Manual gate (you)

- [ ] TT VAE WAV matches host VAE on same latents (listen + PCC).

**Exit → required before Phase 5 signoff if goal is full TTNN stack (no host VAE).

---

## Phase 5 — Full A→Z demo

**Objective:** Single entrypoint: **prompt + lyrics + reference WAV → music**. Functional demo **without trace**.

### Deliverables

| # | Deliverable |
|---|-------------|
| 5.1 | CLI: `demo_acestep_az.py --prompt ... --lyrics ... --reference ... --output out.wav` |
| 5.2 | All hot paths on TT (DiT + TT VAE); host glue for ODE/CFG |
| 5.3 | README section: how to run in tmux on lab machine |

### Device command (demo gate)

```bash
export ACESTEP_USE_TT_VAE=1 ACESTEP_SAVE_WAV=1
flock /tmp/tt_ace_device.lock $PY -m \
  models.tt_dit.pipelines.acestep.demo_acestep_az \
  --prompt "..." --reference /path/to/ref.wav --output /tmp/az_final.wav \
  --infer-steps 30
```

### Manual gate (you) — functional project done

- [ ] One command with **your** prompt + ref → listenable music you’d ship as a demo.
- [ ] No `_captured/` required for normal run.

**Exit → Phase 6 for trace + perf signoff.**

---

## Phase 6 — Trace + 2-CQ + performance (last)

**Objective:** Enable **trace + 2-CQ** on hot paths; record **perf vs A100**; target **< 2 s e2e**.

**Why last:** Trace capture currently hangs (`test_e2e_generate_audio_traced.py`, music perf warmup). Fixing this must not block the functional demo.

### Known issue

- Hang at `capturing trace...` during warmup (condition / audio / decoder regions).
- Metal warning: allocating buffers during active trace.

### Tasks (tmux: `m0-trace`)

| # | Task |
|---|------|
| 6.1 | Reproduce hang under flock; Tracy if needed |
| 6.2 | Verify `trace_region_size=50M`, `num_command_queues=2` |
| 6.3 | Fix trace capture stability (< 5 min per traced e2e) |
| 6.4 | Extend traced decoder for CFG (from Phase 3.5) |
| 6.5 | Run traced perf @ 4 steps, then @ 30 steps + CFG + TT VAE |
| 6.6 | Record `preflight | latent_gen | vae_decode | total_e2e` vs A100 |

### Commands (m0-trace window only)

```bash
bash docs/acestep-m0-trace-run.sh
```

Or individually:

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_traced.py \
  -s -v --timeout=900

export ACESTEP_USE_TT_VAE=0 ACESTEP_SAVE_WAV=1
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_music_perf_traced_acestep.py \
  -k 1x1 -s -v --timeout=3600
```

### Performance budget (p150 target, traced + prod settings)

| Stage | Budget |
|-------|--------|
| Text + ref encode | 0.15–0.30 s |
| Prefill (A+B+D traced) | ~0.05 s |
| Denoise ×30 + CFG | 0.9–1.4 s |
| TT VAE decode | 0.10–0.25 s |
| **Total** | **< 2.0 s** |

### Manual gate (you) — perf signoff

- [ ] Traced e2e completes reliably in **< 15 min** (not hung).
- [ ] Trace + 2-CQ confirmed in logs (`use_2cq=True`).
- [ ] `total_e2e` mean recorded; if > 2 s, optimization backlog listed.

---

## What is already done

- Calls A/B/D/C on TTNN (latents PCC 0.998 @ 4 steps, captured inputs)
- Trace + 2CQ **code paths exist** but capture is unstable — Phase 6
- Host Oobleck VAE (`vae_host.py`)
- TT Oobleck decoder **scaffold** (`vae_oobleck.py` — needs Phase 4)
- Phase 1 G0 tests 1–3 PASS (2026-07-03)

---

## Out of scope (later)

- `acestep-5Hz-lm` planner
- TT Oobleck **encoder** (host encode in Phase 2B is enough for A→Z)
- Multi-chip mesh
- Hard CI enforcement of < 2 s

---

## Execution checklist

```
[ ] tmux session acestep-az created (+ m0-trace window optional)
[ ] Phase 1 G0 complete — WAV listened, no trace required
[ ] Phase 2A complete — live text wired
[ ] Phase 2B complete — live ref audio wired
[ ] Phase 2 integration — prompt+ref e2e pass (traced=False)
[ ] Phase 3 complete — 30 steps + CFG + APG
[ ] Phase 4 complete — TT VAE PCC + WAV
[ ] Phase 5 complete — full demo CLI, listenable output
[ ] Phase 6 complete — trace stable + perf vs A100
```
