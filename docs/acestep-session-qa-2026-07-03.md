# ACE-Step Session Q&A — 2026-07-03

Captured from Cursor agent session on `sjc-snva-tp100`.
Branch at time of session: `dvartanians/feature/tt-hw-planner` @ `1f7efe63d2`.

---

## Q1: Device vs host — which components run where?

**Question:** Give me a brief summary of which components are now running on device and which ones fall back to host?

**Answer:**

For the **current default path** (`traced=False`, `ACESTEP_USE_TT_VAE=0`):

### On device (TT / p150)

The core diffusion chain — verified in Phase 1–2 integration:

| Call | Component | What it does |
|------|-----------|--------------|
| **A** | `ConditionEncoderTT` | Encodes text/lyric/timbre conditioning for DiT |
| **B** | `AudioTokenizerTT` | Tokenizes `src_latents` patches |
| **D** | `DetokenizerTT` | Produces `lm_hints_25hz` from tokenizer output |
| **C** | `DecoderTT` | Flow-matching denoise loop → `target_latents` |

Host orchestrates the loop; the heavy matmuls run on TT.

### On host (CPU / PyTorch)

Everything **before and after** the DiT chain, plus glue:

| Component | What it does |
|-----------|--------------|
| **Text encode** (`text_encode.py`) | Qwen3-Embedding tokenizer + model → `text_hidden_states`, `lyric_hidden_states` |
| **Ref audio encode** (`vae_host.py`) | Load WAV → Oobleck **encoder** → `src_latents`, refer tensors, `is_covers=1` |
| **Input assembly** | ODE timesteps, noise init, `assemble_context_latents`, tokenize preprocess |
| **VAE decode** (default) | Oobleck **decoder** on CPU → stereo WAV (`latents_to_waveform`) |
| **CFG / APG** | Not wired yet; when added, guidance math stays on host after batched DiT forward |

### Optional / not active yet

| Component | Status |
|-----------|--------|
| **TT Oobleck VAE decode** | Code exists (`vae_oobleck.py`) but **`OOBLECK_DECODER_PORT_COMPLETE=False`** — falls back to host even if `ACESTEP_USE_TT_VAE=1` |
| **Trace + 2-CQ** | Code paths exist for Calls A/B/C/D; **disabled** by default (`traced=False`) — Phase 6 |

### One-line mental model

```
HOST:  prompt/lyrics → Qwen3          ref WAV → Oobleck encoder
         ↓                                      ↓
DEVICE: ConditionEncoder → AudioTokenizer → Detokenizer → DiT denoise (×N steps)
         ↓
HOST:  Oobleck decoder → WAV          (TT decoder when Phase 4 gate passes)
```

**Hot path on TT today:** DiT subsystem (Calls A/B/D/C).
**Still on host today:** all input encoding, VAE encode/decode, and sampler guidance.

---

## Q2: Why was VAE encode/decode left on host by default?

**Question:** Why was VAE encode/decode left on host by default?

**Answer:**

Short answer: **it was a deliberate phasing choice**, not because host VAE is better. The plan prioritizes getting **DiT + live inputs working first**, and uses host VAE as a **known-good reference** until the TT Oobleck port is verified.

### Encode (ref WAV → latents) — host by design

Reference-audio encoding was always scoped as **host PyTorch** in Phase 2B:

- Load WAV → run Oobleck **encoder** → produce `src_latents`, refer tensors, masks
- The plan explicitly says: *"TT Oobleck **encoder** (host encode in Phase 2B is enough for A→Z)"* — porting the encoder to TT is **out of scope** for now

**Why:**

- Ref-audio encode runs **once per generation**, not in the inner denoise loop
- Host encoder is easy to match HF/diffusers exactly (PCC/golden tests)
- Unblocks live reference conditioning without waiting on another big TT port (~169M-param decoder-scale work)
- DiT is the performance-critical path; VAE encode is comparatively cheap

### Decode (latents → WAV) — host by default *for now*

Decode defaults to host because **Phase 4 isn’t signed off yet**:

| Fact | Implication |
|------|-------------|
| `ACESTEP_USE_TT_VAE` defaults to `0` | Host Oobleck decoder used |
| `OOBLECK_DECODER_PORT_COMPLETE = False` | TT decoder exists as scaffold but is **gated off** |
| Phase 1 G0 proved TT latents → **host** VAE WAV | Established baseline before TT VAE risk |
| Phase 4 device gate not run | ConvTranspose/ResUnit fixes landed; PCC ≥ 0.99 not confirmed |

The code comments say this directly: host decode **unblocks e2e** before the TT Oobleck port is complete. Phase 1 used host VAE to answer “do TT latents sound like music?” without blocking on VAE port parity.

**Why defer TT decode:**

- Oobleck decoder is a large, fiddly port (ConvTranspose padding, dilated ResUnit crop boundaries)
- Wrong VAE decode = bad audio even if DiT latents are perfect — hard to debug in parallel with DiT work
- Phases 2–3 could run in parallel with Phase 4 **file work**; functional demo doesn’t need TT VAE on day one
- Host decode is the **golden reference** for Phase 4 PCC testing

### The strategy in one line

```
Get the hard part on TT first (DiT chain)  +  use host VAE as trusted glue
Then swap decode to TT once parity is proven (Phase 4)
Never port encoder unless needed (cost >> benefit for A→Z)
```

### When that changes

After Phase 4 gate passes:

```bash
export ACESTEP_USE_TT_VAE=1
# + set OOBLECK_DECODER_PORT_COMPLETE=True
```

Then decode moves to TT. Encode stays on host unless you explicitly scope a TT encoder port later.

**Bottom line:** Host VAE isn’t the long-term architecture — it’s a **staging default** so Phases 1–2 could ship verified while Phase 4 proves TT Oobleck decode matches host within PCC ≥ 0.99.

---

## Q3: Why did `tt_hw_planner auto-up` silently skip VAE on TTNN?

**Question:** I initially used `scripts.tt_hw_planner auto-up` for bring-up and it silently skipped VAE implementation on TTNN. Can you investigate why?

**Answer:**

It wasn’t a bug — **`auto-up` never had VAE in scope**, and it completed successfully against a task definition that stops at **`target_latents`**, not WAV.

### Root cause: two separate models

ACE-Step is split across repos:

| Piece | What it is | Loaded by `auto-up`? |
|-------|------------|----------------------|
| **DiT / conditioning** | `AceStepConditionGenerationModel` (`ACE-Step/acestep-v15-base`) | **Yes** — discovered, captured, graduated |
| **VAE (Oobleck)** | `AutoencoderOobleck` from diffusers (`ACE-Step/Ace-Step1.5/vae/…`) | **No** — separate weights, separate module tree |

`generate_audio()` on the HF model returns **`target_latents [B, L, 64]`**. It does **not** include waveform decode. Documented in `models/demos/hf_eager/acestep_v15_base/e2e_plan.json`:

> *"generate_audio returns the denoised acoustic latents [B, L, 64]; the waveform VAE that turns latents into a .wav is NOT part of this HF model (it lives in a separate acestep VAE repo) and has NO graduated component, so the faithful task output of THIS model is target_latents."*

From the planner’s perspective, bring-up **succeeded** when DiT latents matched HF at PCC ≥ 0.99. No VAE work was ever scheduled.

### Why `auto-up` never attempted VAE

**1. Component discovery only walks the main HF model**

`bringup_status.json` lists **17 components** — all from `AceStepConditionGenerationModel` (encoder, tokenizer, DiT decoder, etc.). No `oobleck`, `autoencoder`, or `vae` entry because `discover_components()` only calls `model.named_modules()` on the loaded checkpoint.

**2. Capture driver doesn’t touch VAE**

The ACE-Step capture driver (`scripts/tt_hw_planner/capture_drivers/acestep.py`) exercises `model.encoder`, `model.decoder` (DiT), `model.tokenizer`, `model.detokenizer` — never `AutoencoderOobleck`. No `_captured/` tensors exist for VAE layers.

**3. Success gate = latents only**

- Smoke test: `test_pipeline_acestep.py` asserts **latent shape** — not WAV
- E2e gate (`e2e_plan.json` Gate 3): `target_latents` PCC ≥ 0.99 — not VAE decode

**4. Backend scope is “flow-matching DiT”**

`family_backends.py` registers ACE-Step as *"Purpose-built tt_dit pipeline for ACE-Step flow-matching DiT"*. Zero references to `oobleck`, `vae_host`, or `audio_vae` under `scripts/tt_hw_planner/`.

### Why it felt “silent”

`auto-up` reports success when all discovered components graduate and e2e passes on **`target_latents`**. It does **not** warn that WAV generation is out of scope, that a second model is needed, or that VAE must stay on host unless ported separately. That detail lives in `e2e_plan.json` and the acestep README — not in main `auto-up` progress output.

### What was added later (outside `auto-up`)

| File | Origin |
|------|--------|
| `vae_host.py` | Manual Phase 2B — host PyTorch Oobleck encode/decode |
| `vae_oobleck.py` + `oobleck_layers.py` | Manual Phase 4 — TT decoder scaffold (still gated off) |

### Summary

| Question | Answer |
|----------|--------|
| Did `auto-up` skip VAE due to a failure? | **No** — VAE was never in the task graph |
| Is host VAE a planner default? | **Yes** — by omission; planner defines “done” as latents |
| Can `auto-up` port VAE today? | **Not as-is** — needs a second bring-up target or manual Phase 4 |

To automate TT VAE in the future, extend the planner with a **companion model** pass (e.g. `AutoencoderOobleck` from `ACE-Step/Ace-Step1.5/vae`) with its own component tree — similar to SD templates that include UNet **and** VAE separately. That path doesn’t exist for ACE-Step today.

---

## Q4: What would it take for `scripts.tt_hw_planner auto-up` to implement the full stack (all 3 components per the reference demo)?

**Question:** What would it take for `scripts.tt_hw_planner auto-up` to do the full implementation which would include all the 3 components according to our reference demo?

**Answer:**

Running:

```bash
python -m scripts.tt_hw_planner auto-up ACE-Step/acestep-v15-base --box p150 --mesh 1,1
```

only brings up **one HF repo**: `AceStepConditionGenerationModel` inside `acestep-v15-base`. To match the reference demo (`demo_acestep_az.py` + [awesome-ace-step](https://github.com/ace-step/awesome-ace-step)), the planner needs a **stack orchestrator above per-model `auto-up`**, not just an extension of the DiT run.

### What `auto-up` does today

| Aspect | Today |
|--------|--------|
| **Discovery** | Walks `named_modules()` on that single model (`module_tree.py` → `bringup_plan.py`) |
| **Components** | ~13 NEW + 4 REUSE (DiT, condition encoders, tokenizer/detokenizer, FSQ) |
| **E2E golden** | `target_latents` PCC ≥ 0.99 — **not WAV** |
| **Inputs** | Pre-captured `text_hidden` / `lyric_hidden` from `_captured/` — **not live Qwen3** |
| **Success =** | Latents match HF `generate_audio()` |

`e2e_plan.json` states this explicitly:

> *"generate_audio returns the denoised acoustic latents [B, L, 64]; the waveform VAE that turns latents into a .wav is NOT part of this HF model … so the faithful task output of THIS model is target_latents."*

So today's `auto-up` is **DiT-tower only**. It never sees the other checkpoints because they are **separate models under `ACE-Step/Ace-Step1.5/`**, not submodules of `acestep-v15-base`.

### The 3 components vs our reference demo

Two framings:

**A. Production hybrid ([awesome-ace-step](https://github.com/ace-step/awesome-ace-step)) — LM + DiT + VAE**

| Component | Official model | In `auto-up` today? |
|-----------|----------------|---------------------|
| **LM planner** | `acestep-5Hz-lm-{0.6B,1.7B,4B}` (Qwen3) | No — separate checkpoint |
| **DiT** | `acestep-v15-base` | Yes — this is all `auto-up` covers |
| **VAE** | Oobleck under `Ace-Step1.5/vae/` | No — loaded separately via `vae_host.py` |

**B. `demo_acestep_az.py` functional path — text + DiT + VAE**

| Component | Model | In `auto-up` today? |
|-----------|-------|---------------------|
| **Text encode** | `Qwen3-Embedding-0.6B` | No — host-only in `text_encode.py` |
| **DiT** | `acestep-v15-base` | Partially (latents-only gate) |
| **VAE decode** | Oobleck | No — host default; TT scaffold incomplete |

The reference demo wires all of these in `AceStepPipeline` → `demo_acestep_az.py`, but **`auto-up` never validates that end-to-end path**.

### Why the planner can't do it today (mechanism)

Four structural gaps, not just missing docs:

```
auto-up today:
  ACE-Step/acestep-v15-base → module_tree → capture_drivers/acestep.py → e2e_plan (target_latents)

Not in graph:
  Qwen3-Embedding-0.6B | acestep-5Hz-lm-* | Oobleck VAE
                              ↓
                    demo_acestep_az.py → WAV
```

1. **Single-model discovery** — `build_bringup_plan()` calls `discover_components_from_hf_id()` once; no multi-checkpoint bundle walk.
2. **E2E contract stops at latents** — no Call T (text), Call LM, Call VAE in `e2e_plan.json`.
3. **Capture drivers** — `capture_drivers/acestep.py` only exercises DiT subgraph; no text/LM/VAE drivers.
4. **Backend registry** — only `acestep-v15-base` has a TTS backend with `use_module_tree=True`; LM/VAE need their own entries in `family_backends.py`.

`BRING_UP_PLAN.md` also documents v0 scope as explicitly skipping LM and VAE:

> *"Text → music only. Skip LM hints, cover/reference-audio modes, and `acestep-5Hz-lm`."*

### What it would take: stack-level `auto-up`

#### 1. Stack manifest (new)

A single config describing all towers, dependency order, and the final gate:

```json
{
  "stack_id": "acestep-az-full",
  "reference": "models/tt_dit/pipelines/acestep/demo_acestep_az.py",
  "towers": [
    {"id": "text",  "model_id": "ACE-Step/Ace-Step1.5/Qwen3-Embedding-0.6B", "backend": "Qwen3-Embedding"},
    {"id": "lm",    "model_id": "ACE-Step/Ace-Step1.5/acestep-5Hz-lm-1.7B", "backend": "Qwen3ForCausalLM", "optional": true},
    {"id": "dit",   "model_id": "ACE-Step/acestep-v15-base", "backend": "ACE-Step v1.5"},
    {"id": "vae",   "model_id": "ACE-Step/Ace-Step1.5/vae", "backend": "Oobleck VAE"}
  ],
  "e2e_golden": "stereo_wav_48kHz",
  "pass_criteria": {"latent_pcc": 0.99, "wav_pcc_or_snr": "TBD"}
}
```

CLI shape: `auto-up --stack acestep-az-full` (runs towers in order, merges overlays).

#### 2. `family_backends.py` — register missing towers

| Tower | Backend entry needed | Reuse in tt-metal |
|-------|---------------------|-------------------|
| Text | Qwen3-Embedding (0.6B) | `tt_transformers` + Embed backend already drafted |
| LM | `acestep-5Hz-lm-*` | `tt_transformers` `Qwen3ForCausalLM` — wire 3 variants |
| DiT | **exists** | Current `auto-up` path |
| VAE | Oobleck decoder (encode optional host) | `models/tt_dit/models/audio_vae/vae_oobleck.py` |

#### 3. Capture drivers (new)

| Driver | Captures |
|--------|----------|
| `acestep_text_encode.py` | `text_hidden_states`, `lyric_hidden_states` from live prompt/lyrics |
| `acestep_lm_planner.py` | `audio_codes` → `lm_hints_25Hz` (replaces Call B when enabled) |
| `acestep_vae.py` | ref WAV encode + latents→PCM decode |
| `acestep.py` | **exists** — DiT Calls A/B/D/C |

#### 4. E2E plan upgrade

Extend or replace `e2e_plan.json` so the golden is **WAV**, not latents:

| Change | From → To |
|--------|-----------|
| `task_head.golden_reference` | `target_latents` → `demo_acestep_az` output WAV |
| `calls` | Add **T** (text), **LM** (optional, skips B), **VAE** (decode) |
| `input_construction` | Live `--prompt`, `--lyrics`, `--reference` (not `_captured/` embeddings) |
| `pass_criteria` | Latent PCC **and** waveform quality vs HF golden |

#### 5. Correctness layer

Today TTS backends get a generic skeleton marker with no audio comparator. A **TTS/WAV comparator** (PCC, SNR, or perceptual gate) is needed in `scripts/tt_hw_planner/correctness/`.

#### 6. Pipeline glue (partially done)

| Piece | Status |
|-------|--------|
| `text_encode.py` | Host wired (Phase 2A) |
| `pipeline_acestep.py` | Live inputs wired |
| `audio_decode.py` / TT VAE | Scaffold; `OOBLECK_DECODER_PORT_COMPLETE=False` |
| `lm_planner.py` | Not built yet (Phase 7) |
| `demo_acestep_az.py` | Reference entrypoint — should become the planner e2e smoke target |

#### 7. CLI / bringup loop changes

- `bringup_plan.py`: `build_stack_bringup_plan()` — multi-root discovery
- `bringup_loop.py`: namespaced `_captured/<tower>/` per checkpoint
- `cli.py`: `--stack`, tower-sequenced graduation, stack-level promote

### Effort estimate

Assuming DiT graduation (Phase 0) stays green:

| Work item | Complexity | Notes |
|-----------|------------|-------|
| Stack manifest + multi-model CLI | **M** (~1–2 wk) | Core plumbing |
| Text encode on TT (Qwen3-Embedding) | **S–M** | Reuse `tt_transformers` |
| TT VAE decode (Oobleck PCC ≥ 0.99) | **M** | Phase 4 work |
| Full WAV e2e + TTS comparator | **M** | New golden contract |
| LM planner ×3 variants on TT | **M** | Reuse Qwen3; ACE-Step heads |
| Trace + perf on full stack | **L** | Phase 8 |

**Total for planner-owned full stack:** roughly **6–10 engineer-weeks** on top of what Phases 3–5 are doing manually.

### Practical recommendation

Don't try to make one `auto-up ACE-Step/acestep-v15-base` call swallow everything. Instead:

1. **Keep** current `auto-up` for DiT (latents gate) — it's working.
2. **Add** `auto-up --stack acestep-az-full` that runs **4 sequential bring-ups** (text, dit, vae, lm) with a **stack e2e** at the end.
3. **Point the stack e2e** at `demo_acestep_az.py` with WAV golden.

That mirrors how ACE-Step actually ships: one umbrella bundle (`Ace-Step1.5`), four checkpoints, one demo entrypoint — while reusing the Qwen3 work already in `tt_transformers` instead of greenfield ports.

### Summary

| Question | Answer |
|----------|--------|
| Can one `auto-up acestep-v15-base` do the full demo? | **No** — single-model, latents-only contract |
| What's missing for full stack? | Stack manifest, 3 new capture drivers, WAV e2e, TTS comparator, LM/VAE/text backends |
| Reuse vs greenfield? | Qwen3 already in `tt_transformers`; Phase 7/2A are wiring, not new arch ports |
| Recommended path? | Phased manual bring-up first (**4 → 3 → 2C → 5**); defer `auto-up --stack` until e2e green |

---

## Q5: Efficient way to get all 3 towers end-to-end on TTNN?

**Question:** In short, what is the efficient way to have all 3 end-to-end on TTNN? (Qwen is already in tt-metal.)

**Answer:**

**Do not** greenfield-port Qwen or wait on one mega `auto-up`. **Do** wire ACE-Step into existing `tt_transformers` and finish the gaps in this order:

| Priority | Phase | Work |
|----------|-------|------|
| 1 | **4** | TT Oobleck **decode** (biggest missing TT piece) |
| 2 | **3** | CFG @ 30 steps on DiT (already on TT) |
| 3 | **2C** | TT **Qwen3-Embedding-0.6B** via `tt_transformers` |
| 4 | **5** | `demo_acestep_az.py` — TT text + TT DiT + TT VAE |
| 5 | **7** | Optional LM (`acestep-5Hz-lm-1.7B`, same Qwen3 stack) |
| 6 | **6/8** | Trace + perf (last) |

**Hot path on TT:**

```
TT Qwen3-Embedding + host ref VAE encode (once) → TT DiT → TT Oobleck decode → WAV
```

**Host forever:** ref WAV encode only. **Not on critical path:** TT VAE encoder, `tt_hw_planner --stack` orchestration.

See `docs/superpowers/plans/2026-07-03-acestep-az-phases.md` and `docs/acestep-az-phases-summary.md`.

---

## Related paths

| Resource | Path |
|----------|------|
| This Q&A doc | `docs/acestep-session-qa-2026-07-03.md` |
| Official model list | [awesome-ace-step](https://github.com/ace-step/awesome-ace-step) |
| Planner e2e scope (latents-only) | `models/demos/hf_eager/acestep_v15_base/e2e_plan.json` |
| A→Z phase plan | `docs/superpowers/plans/2026-07-03-acestep-az-phases.md` |
| Progress log | `docs/acestep-az-progress.md` |
| Reference demo (full stack target) | `models/tt_dit/pipelines/acestep/demo_acestep_az.py` |
| Planner backends registry | `scripts/tt_hw_planner/family_backends.py` |
| ACE-Step capture driver (DiT only) | `scripts/tt_hw_planner/capture_drivers/acestep.py` |
| Host VAE bridge | `models/demos/hf_eager/acestep_v15_base/tt/vae_host.py` |
| TT VAE scaffold | `models/tt_dit/models/audio_vae/vae_oobleck.py` |
| Qwen3 in tt-metal | `models/tt_transformers/` |
