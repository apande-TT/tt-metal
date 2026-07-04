# ACE-Step v1.5 — Phased execution plan (A→Z on TTNN)

**Your goal:** Give **text prompt + reference audio** → get **generated music (WAV)** on the **full ACE-Step production stack**.

**Architecture target (three towers):**

| Tower | Model | Phase-5 status | Full-stack target |
|-------|--------|----------------|-------------------|
| **Text conditioning** | Qwen3-Embedding-0.6B | Host wired (Phase 2A) | **TT** via `tt_transformers` (Phase 2C) |
| **DiT + conditioning** | `acestep-v15-base` (Calls A/B/D/C) | **TT** (auto-up done) | TT + trace (Phase 6) |
| **5Hz LM planner** | `acestep-5Hz-lm-{0.6B,1.7B,4B}` | **Not wired** | All three via `tt_transformers` Qwen3 (Phase 7); **1.7B default** |
| **Audio VAE** | AutoencoderOobleck | Host encode; TT decode (Phase 4) | TT decode; host encode OK |

**Perf stretch goal (Phase 8, last):** **trace + 2-CQ** on all hot TT paths, **end-to-end time vs A100 (< 2 s full stack including LM)**.

**Hardware:** p150 Blackhole, 1×1 mesh. **One device job at a time** — use `flock /tmp/tt_ace_device.lock`.

**Diagram (status):** `docs/acestep_v15_e2e_dataflow_ttnn_status.png`

**Official model reference:** [awesome-ace-step](https://github.com/ace-step/awesome-ace-step) — curated ACE-Step v1.5 model list (LM + DiT variants).

**Updated:** 2026-07-03 — trace decoupled from Phases 1–5; functional path uses `traced=False`.
**Updated:** 2026-07-03 — Phase 7 added for `acestep-5Hz-lm` planner (removed from out-of-scope).
**Updated:** 2026-07-03 — Phase 7 scoped to all three official **Qwen3 LM planner** variants (0.6B / 1.7B / 4B).
**Updated:** 2026-07-03 — **Efficient full-TT path:** reuse existing `tt_transformers` Qwen3 for text + LM; finish TT VAE + CFG; gate on `demo_acestep_az.py` (not one mega `auto-up`).

### Official ACE-Step Qwen3 models (from [awesome-ace-step](https://github.com/ace-step/awesome-ace-step))

**Text conditioning (Phase 2A — not the LM planner):**

| Model | Base | Role in pipeline |
|-------|------|------------------|
| `Qwen3-Embedding-0.6B` | Qwen3 decoder (embedding head) | Caption + lyrics → `text_hidden` / `lyric_hidden` for Call A |

**5Hz LM planners (Phase 7 — audio-code generation):**

| Model | Qwen3 base | VRAM (HF ref) | Role | Bundle path |
|-------|------------|---------------|------|-------------|
| `acestep-5Hz-lm-0.6B` | Qwen3-0.6B | 6–8 GB | Lightweight planner | `ACE-Step/Ace-Step1.5/acestep-5Hz-lm-0.6B/` |
| **`acestep-5Hz-lm-1.7B`** | Qwen3-1.7B | 8–16 GB | **Default** — full features | `ACE-Step/Ace-Step1.5/acestep-5Hz-lm-1.7B/` |
| `acestep-5Hz-lm-4B` | Qwen3-4B | 16+ GB | Best quality / audio understanding | `ACE-Step/Ace-Step1.5/acestep-5Hz-lm-4B/` |

**Selection API (target):** `ACESTEP_LM_PLANNER_MODEL=acestep-5Hz-lm-1.7B` or `--lm-model {0.6B,1.7B,4B}` on CLI. Same `tt_transformers` `Qwen3ForCausalLM` stack; only weights + `config.json` dims change per variant.

**Related Qwen (out of ACE-Step hot path for A→Z):** `acestep-captioner` / `acestep-transcriber` (Qwen2.5 Omni) — annotation tools, not generation stack.

**Existing Qwen in tt-metal (reuse, do not greenfield):**

| Use case | Existing path | ACE-Step hook |
|----------|---------------|---------------|
| **Qwen3 causal LM decode** | `models/tt_transformers/` — `Qwen3ForCausalLM`, generator; `tt-train/sources/examples/qwen3/` (0.6B ref) | Phase 7: wire **all three** `acestep-5Hz-lm-{0.6B,1.7B,4B}` via HF `config.json` |
| **Qwen3-Embedding** | `tt_transformers` + `tt_hw_planner` Embed backend (`Qwen3-Embedding-8B` in `model_config.py`; ACE uses **0.6B**) | **Phase 2C:** wire `Qwen3-Embedding-0.6B` — integration, not new arch |
| **Galaxy Qwen decode** | `models/demos/llama3_70b_galaxy/` — `qwen_model_config.py`, `demo_qwen_decode.py` | Reference for decode perf patterns; p150 uses `tt_transformers` |

Phase 2C and Phase 7 are **integration + PCC**, not from-scratch Qwen ports. Core decoder layers come from `tt_transformers`. Use `tt_hw_planner auto-up` only for ACE-Step–specific deltas (embedding head, LM audio-code vocab); **do not** wait on a single mega `auto-up --stack` to finish A→Z.

### Efficient path — full TT e2e (3 towers + host ref glue)

**Target hot path on TT:**

```
TT Qwen3-Embedding (prompt/lyrics)     host ref VAE encode (once/gen)
              │                                    │
              └──────────────┬─────────────────────┘
                             ▼
              TT DiT (Calls A/B/D/C + CFG @ 30 steps)
                             ▼
                    TT Oobleck decode → WAV
```

**Optional (Phase 7, production):** TT `acestep-5Hz-lm-1.7B` via same `tt_transformers` Qwen3 stack — replaces Call B tokenizer.

**Recommended order for remaining work** (Phases 1–2 host path already done):

| Priority | Phase | Why first |
|----------|-------|-----------|
| 1 | **4** — TT VAE decode | Biggest missing TT piece; host decode blocks full-stack signoff |
| 2 | **3** — CFG @ 30 steps | Quality + production parity on DiT (already on TT) |
| 3 | **2C** — TT Qwen3-Embedding | Wire existing `tt_transformers`; replaces host `text_encode.py` on device |
| 4 | **5** — `demo_acestep_az.py` | Full TT signoff: TT text + TT DiT + TT VAE (host ref encode only) |
| 5 | **7** — LM planner | Production stack; same Qwen3 reuse pattern |
| 6 | **6 / 8** — trace + perf | Last; never block 3–5 |

**Not on the critical path:** TT Oobleck **encoder** (host ref encode once per gen is enough), `tt_hw_planner --stack` orchestration (defer until manual e2e green — see Q4 in `docs/acestep-session-qa-2026-07-03.md`).

---

## Two tracks

| Track | Phases | Blocks first WAV demo? | Blocks full production stack? |
|-------|--------|----------------------|-------------------------------|
| **Functional (full TT e2e)** | 2C → 3 → 4 → 5 | No — main line to first listenable WAV on TT | Yes — missing LM planner (Phase 7) |
| **LM planner** | 7 | No — starts after Phase 5 interim demo | No — completes production path |
| **Trace / perf (DiT)** | 6 (M0) | No | Only blocks < 2 s DiT-stack signoff |
| **Trace / perf (full stack)** | 8 | No | Blocks < 2 s with LM on TT |

Do **not** wait on trace capture or LM bring-up to start Phase 2–5 file work or non-traced device tests.

---

## Phase overview

| Phase | Name | You get at the end | Device? | Trace? | Parallel? |
|-------|------|--------------------|---------|--------|-----------|
| **1** | Functional baseline | TT latents → host VAE WAV | Yes (serial) | **No** | No |
| **2** | Live inputs | Real prompt + ref WAV (host text today) | Mixed | No | **Yes** (2A ∥ 2B); **2C** TT text after 4+3 |
| **2C** | TT text encode | Qwen3-Embedding-0.6B on TT via `tt_transformers` | Yes | No | After Phase 4+3 file work OK |
| **3** | Production sampler | 30 ODE steps + CFG + APG | Yes | No | Code ∥ Phase 4 |
| **4** | TT VAE decode | Oobleck decoder on TT, PCC ≥ 0.99 | Yes | No | **Priority 1** — start now |
| **5** | A→Z demo (full TT) | CLI: TT text + TT DiT + TT VAE → WAV | Yes | No | After 2C+3+4 |
| **6** | Trace + perf (DiT) | trace + 2-CQ on DiT hot path; timing vs A100 | Yes | **Yes** | No |
| **7** | 5Hz LM planner (Qwen3 ×3) | Qwen3 LM → `audio_codes` on TT; `--lm-model` selects variant | Yes | No | CPU/file work ∥ 6 |
| **8** | Trace + perf (full) | trace + 2-CQ including LM; < 2 s full stack | Yes | **Yes** | No |

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
tmux new-window -t acestep-az -n phase2c-tt-text  # TT Qwen3-Embedding (Phase 2C)
tmux new-window -t acestep-az -n phase4-vae
tmux new-window -t acestep-az -n phase5-demo
tmux new-window -t acestep-az -n phase7-lm      # 5Hz LM planner (after Phase 5 interim demo)
tmux new-window -t acestep-az -n device      # functional flock pytest (Phases 1–5, 7)
tmux new-window -t acestep-az -n m0-trace    # Phase 6/8 trace debug — never blocks 2–5
tmux attach -t acestep-az
```

- **device** — Phases 1–5 and 7 integration tests (`traced=False`).
- **phase2a-text, phase2c-tt-text, phase2b-ref, phase4-vae, phase7-lm** — implementation + CPU/golden tests (no device lock).
- **m0-trace** — trace hang debug + `docs/acestep-m0-trace-run.sh` (Phase 6 DiT, Phase 8 full stack).

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
| 2A.1 | Add `text_encode.py`: ACE-Step processor + Qwen3-Embedding-0.6B (torch reference) — **done** |
| 2A.2 | Extend `build_inputs(..., use_captured=False, prompts=..., lyrics=...)` — **done** |
| 2A.3 | Wire `pipeline_acestep.py`: stop ignoring `prompts` — **done** |
| 2A.4 | Host golden tests for fixed prompts — **done** |

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

**Exit → Phase 3 + Phase 4 in parallel; then Phase 2C (TT text).**

### Phase 2C — TT text encode (Qwen3-Embedding on device)

**Objective:** Move caption/lyrics encoding from host PyTorch to **TT** using existing **`models/tt_transformers/`** Qwen3-Embedding stack. Same weights as Phase 2A (`Qwen3-Embedding-0.6B` from Ace-Step1.5 bundle).

**Why not optional:** Qwen3 is already implemented in tt-metal — full TT e2e requires TT text encode, not a permanent host fallback.

| # | Task |
|---|------|
| 2C.1 | Add `text_encode_tt.py` (or TT branch in `text_encode.py`): load 0.6B via `tt_transformers` `ModelConfig` + HF `config.json` |
| 2C.2 | Map ACE-Step processor output → TT forward → `text_hidden_states`, `lyric_hidden_states` |
| 2C.3 | Device PCC vs host golden (`test_text_encode_acestep.py` extended for TT) |
| 2C.4 | Pipeline flag: `ACESTEP_USE_TT_TEXT_ENCODE=1` / `use_tt_text_encode=True`; default **on** after gate passes |

**Key reuse:** `models/tt_transformers/tt/{decoder,attention,mlp,embedding}.py`, `model_config.py` (Qwen3-Embedding patterns), `scripts/tt_hw_planner` Embed backend.

### Device command (Phase 2C gate)

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_text_encode_acestep.py \
  -k tt -s -v --timeout=900
```

### Manual gate (you)

- [ ] TT embeddings PCC ≥ 0.99 vs host on fixed prompts.
- [ ] Pipeline with `ACESTEP_USE_TT_TEXT_ENCODE=1` passes live-input e2e.

**Exit → required before Phase 5 full-TT signoff.**

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

## Phase 5 — A→Z demo (DiT + tokenizer path)

**Objective:** Single entrypoint: **prompt + lyrics + reference WAV → music**. Uses **Call B+D** (audio tokenizer + detokenizer) for `lm_hints_25Hz` in cover mode. Functional demo **without trace** and **without 5Hz LM planner** (Phase 7 adds the production LM path).

### Deliverables

| # | Deliverable |
|---|-------------|
| 5.1 | CLI: `demo_acestep_az.py --prompt ... --lyrics ... --reference ... --output out.wav` |
| 5.2 | DiT + TT VAE decode on TT; host glue for text encode, ref encode, ODE/CFG |
| 5.3 | README section: how to run in tmux on lab machine |

### Device command (demo gate)

```bash
export ACESTEP_USE_TT_VAE=1 ACESTEP_SAVE_WAV=1
read -r -d '' LYRICS <<'EOF' || true
[verse]
City lights are fading slow
Warm piano starts to glow
Soft drums keep the time so low
In this lounge where feelings flow
[chorus]
Stay with me tonight
Under neon light
Smooth jazz in the air
Like we haven't got a care
EOF
flock /tmp/tt_ace_device.lock $PY -m \
  models.tt_dit.pipelines.acestep.demo_acestep_az \
  --prompt "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm" \
  --lyrics "$LYRICS" \
  --reference /tmp/ref_kaazoom_25s.wav \
  --output /tmp/az_phase5_signoff.wav \
  --infer-steps 30 --guidance-scale 7.0 --shift 3.0 --audio-duration 30 --seed 42 \
  --use-tt-vae --use-tt-text-encode --no-traced
```

### Manual gate (you) — interim functional demo done

- [ ] One command with **your** prompt + ref → listenable music you’d ship as a demo.
- [ ] No `_captured/` required for normal run.
- [ ] Uses tokenizer path (Call B+D), not LM planner — OK for Phase 5 signoff.

**Exit → Phase 6 (DiT trace) can start; Phase 7 required for full production stack.**

---

## Phase 6 — Trace + 2-CQ + performance (DiT stack)

**Objective:** Enable **trace + 2-CQ** on DiT hot paths (Calls A/B/D/C); record **perf vs A100** for the **Phase 1–5 stack** (no LM planner yet).

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
- [ ] `total_e2e` mean recorded for DiT stack; if > 2 s, optimization backlog listed.

**Exit → Phase 8 after Phase 7 LM is on TT.**

---

## Phase 7 — 5Hz LM planner (Qwen3 variants on device)

**Objective:** Wire the official **`acestep-5Hz-lm-{0.6B,1.7B,4B}`** planners ([awesome-ace-step LM table](https://github.com/ace-step/awesome-ace-step#language-models-planner)) into the production path: text/structure prompt → **discrete `audio_codes`** → `lm_hints_25Hz` → DiT. **Reuse existing `tt_transformers` Qwen3 decode stack** — one integration path, three weight checkpoints.

**Why separate from Phase 2A:** Phase 2A uses **Qwen3-Embedding-0.6B** (text conditioning for Call A). Phase 7 is the **5Hz audio-code LM planner** — a different model family that can **replace Call B** (tokenizer) when generating cover hints. HF path: `audio_codes` → `tokenize.quantizer.get_output_from_indices()` → detokenize → `lm_hints_25Hz` (see `prepare_condition` in `modeling_acestep_v15_base.py`).

**Variant rollout (p150 1×1):**

| Order | Model | Why |
|-------|-------|-----|
| 1 | **`acestep-5Hz-lm-1.7B`** | Official default; full features |
| 2 | `acestep-5Hz-lm-0.6B` | Lightweight; good for CI / memory headroom on p150 |
| 3 | `acestep-5Hz-lm-4B` | Quality tier; validate p150 fits before signoff |

**Default for bring-up:** **1.7B**. Gate each variant with host golden → TT PCC → listen test before enabling the next size.

### Parallel tracks

```
Phase 7A (host LM)         Phase 7B (tt_transformers Qwen3)
     │                           │
     └──────────┬────────────────┘
                ▼
         pipeline_acestep.py
    --use-lm-planner → audio_codes → lm_hints
                │
                ▼
         Phase 7C device e2e + quality gate
```

### Phase 7A — Host LM reference (tmux: `phase7-lm`)

| # | Task |
|---|------|
| 7A.1 | Add `lm_planner.py`: load any `acestep-5Hz-lm-{0.6B,1.7B,4B}` from bundle; ACE-Step chat template; generate `audio_codes` |
| 7A.2 | Map LM output → `audio_codes` tensor compatible with `quantizer.get_output_from_indices` |
| 7A.3 | CPU golden per variant: LM `audio_codes` → `lm_hints_25Hz` matches HF on fixed prompts |
| 7A.4 | Add `--use-lm-planner` and `--lm-model {0.6B,1.7B,4B}` to `demo_acestep_az.py` (host LM first) |

**Key files:** new `models/tt_dit/pipelines/acestep/lm_planner.py`, `pipeline_acestep.py`, `tt/pipeline.py`

### Phase 7B — TTNN integration (reuse `tt_transformers`)

| # | Task |
|---|------|
| 7B.1 | Load each variant via **`models/tt_transformers/`** (`Qwen3ForCausalLM` + HF `config.json` per size; add `model_params` entries for p150 if needed) |
| 7B.2 | Shared ACE-Step chat template + audio-code vocab across variants; PCC per layer vs host golden (**1.7B first**, then 0.6B, 4B) |
| 7B.3 | Wire TT LM into `lm_planner.py`; `resolve_lm_planner_path(model=...)`; device PCC vs host per variant |
| 7B.4 | Pipeline: when `use_lm_planner=True`, skip Call B tokenizer; feed `audio_codes` into detokenizer path only |
| 7B.5 | Optional: `tt_hw_planner auto-up` for ACE-Step–specific head/output only if not covered by manual wiring |

**Key reuse paths:** `models/tt_transformers/tt/{decoder,attention,mlp,model,generator}.py`, `models/tt_transformers/demo/simple_text_demo.py`, `scripts/tt_hw_planner` Embed backend for Qwen3-Embedding patterns.

### Phase 7C — Integration (device window)

| # | Task |
|---|------|
| 7C.1 | E2e test per variant (parametrize): prompt + ref + `--use-lm-planner --lm-model …` → WAV on device (`traced=False`) |
| 7C.2 | Quality gate: each LM variant vs tokenizer-only on same prompt+ref (listen + latent PCC) |
| 7C.3 | Document variant selection (0.6B = fast/CI, 1.7B = default, 4B = quality) per [awesome-ace-step](https://github.com/ace-step/awesome-ace-step) |

### Device command

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_lm_planner_acestep.py \
  -s -v --timeout=3600   # add in Phase 7

bash docs/acestep-az-phase7-run.sh
# Or inline:
read -r -d '' LYRICS <<'EOF' || true
[verse]
City lights are fading slow
Warm piano starts to glow
Soft drums keep the time so low
In this lounge where feelings flow
[chorus]
Stay with me tonight
Under neon light
Smooth jazz in the air
Like we haven't got a care
EOF
export ACESTEP_USE_TT_VAE=1 ACESTEP_PIPELINE_DIR=/local/ttuser/gtobar/acestep_pipeline
flock /tmp/tt_ace_device.lock $PY -m \
  models.tt_dit.pipelines.acestep.demo_acestep_az \
  --prompt "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm" \
  --lyrics "$LYRICS" \
  --reference /tmp/ref_kaazoom_25s.wav \
  --output /tmp/az_lm_tt.wav \
  --use-lm-planner --use-tt-lm-planner --lm-model 1.7B \
  --infer-steps 30 --guidance-scale 7.0 --shift 3.0 --audio-duration 30 --seed 42 \
  --use-tt-vae --use-tt-text-encode --no-traced
```

### Manual gate (you) — full production stack done

- [ ] LM planner runs on **TT device** for **`acestep-5Hz-lm-1.7B`** (required) and at least one of **0.6B / 4B** validated.
- [ ] `--use-lm-planner --lm-model 1.7B` produces listenable output; quality acceptable vs HF LM path.
- [ ] Cover-mode hints audibly differ from tokenizer-only path where expected.

**Exit → Phase 8 for full-stack trace + perf (default LM = 1.7B).**

---

## Phase 8 — Trace + 2-CQ + performance (full stack)

**Objective:** Extend trace + 2-CQ to **LM planner + DiT + TT VAE**; target **< 2 s e2e** for the complete three-tower hot path.

### Tasks (tmux: `m0-trace`)

| # | Task |
|---|------|
| 8.1 | Trace capture for LM prefill (or batched LM+DiT if fused) |
| 8.2 | Traced e2e with `--use-lm-planner` + 30 steps + CFG + TT VAE |
| 8.3 | Record full timing table vs A100 |

### Performance budget (p150 target, full stack)

| Stage | Budget |
|-------|--------|
| Text + ref encode | 0.15–0.30 s |
| **LM planner (5Hz codes, 1.7B default)** | **0.10–0.25 s** (0.6B faster; 4B slower) |
| Prefill (A+D traced; B skipped if LM) | ~0.05 s |
| Denoise ×30 + CFG | 0.9–1.4 s |
| TT VAE decode | 0.10–0.25 s |
| **Total** | **< 2.0 s** |

### Manual gate (you) — full-stack perf signoff

- [ ] Traced full-stack e2e completes reliably (not hung).
- [ ] `total_e2e` mean recorded with LM on TT; optimization backlog if > 2 s.

---

## What is already done

- Calls A/B/D/C on TTNN (latents PCC 0.998 @ 4 steps, captured inputs)
- **Qwen3 on TT:** `tt_transformers` supports `Qwen2ForCausalLM` / `Qwen3ForCausalLM` (Qwen2.5–Qwen3 family); Qwen3-Embedding backend in `tt_hw_planner`
- Trace + 2CQ **code paths exist** but capture is unstable — Phase 6
- Host Oobleck VAE (`vae_host.py`)
- TT Oobleck decoder **scaffold** (`vae_oobleck.py` — needs Phase 4)
- Phase 1 G0 tests 1–3 PASS (2026-07-03)

---

## Out of scope (later)

- TT Oobleck **encoder** (host ref encode in Phase 2B is enough for A→Z)
- Multi-chip mesh
- Hard CI enforcement of < 2 s
- `tt_hw_planner auto-up --stack` full orchestration until manual Phases 3–5 e2e is green (see Q4 in session Q&A)
- `acestep-captioner` / `acestep-transcriber` (Qwen2.5 Omni annotation — not generation hot path)
- DiT turbo/sft variants (`acestep-v15-turbo`, `acestep-v15-sft`) — separate from base bring-up; see [awesome-ace-step DiT table](https://github.com/ace-step/awesome-ace-step#dit-models-diffusion-transformer)

---

## Execution checklist

```
[ ] tmux session acestep-az created (+ m0-trace window optional)
[ ] Phase 1 G0 complete — WAV listened, no trace required
[ ] Phase 2A complete — live text wired (host)
[ ] Phase 2B complete — live ref audio wired
[ ] Phase 2 integration — prompt+ref e2e pass (traced=False)
[ ] Phase 4 complete — TT VAE PCC + WAV                    ← priority 1
[ ] Phase 3 complete — 30 steps + CFG + APG                ← priority 2
[ ] Phase 2C complete — TT Qwen3-Embedding via tt_transformers
[ ] Phase 5 complete — full TT demo CLI (text+DiT+VAE on device)
[ ] Phase 6 complete — DiT trace stable + perf vs A100
[ ] Phase 7A complete — host LM reference + audio_codes golden (all 3 variants)
[ ] Phase 7B complete — TT LM via `tt_transformers` Qwen3; 1.7B + 0.6B/4B PCC
[ ] Phase 7C complete — LM wired in pipeline; quality gate vs tokenizer per variant
[ ] Phase 8 complete — full-stack trace + perf vs A100 (< 2 s target)
```
