# ACE-Step v1.5 — Full A→Z TTNN Plan (text + reference audio → music)

**Goal:** User provides **caption/lyrics + reference WAV** → pipeline returns **listenable stereo music**, running **trace + 2-CQ** on all hot TT paths, with **end-to-end wall time competitive with A100 (< 2 s reference)**.

**Updated:** 2026-07-03 (post diagram + trace/host-glue clarification)

**Reference hardware:** Nvidia A100 full stack (Qwen embed + VAE encode + DiT×30 + CFG + VAE decode) **< 2 s**.

**Target hardware:** p150 Blackhole 1×1 mesh, serialized via `flock /tmp/tt_ace_device.lock`.

---

## Where we are today

| Layer | Status | Notes |
|-------|--------|-------|
| **Calls A, B, D, C** (condition, tokenizer, detokenizer, DiT) | ✅ TTNN + trace + 2CQ | PCC 0.998 latents @ gate `infer_steps=4`, captured inputs |
| **Host glue** (context assembly, ODE Euler, noise) | ✅ Working | Negligible time; stays on host by design |
| **Live text encode** (Qwen3-Embedding-0.6B) | ❌ Not wired | Uses `_captured/` tensors; `prompts` ignored |
| **Live ref-audio encode** (Oobleck encoder) | ❌ Not wired | Uses captured `refer_audio` |
| **CFG + APG/ADG** | ❌ Not wired | v0 runs single forward/step; prod doubles DiT |
| **VAE decode (host)** | ✅ Phase A done | `vae_host.py`, listenable WAV |
| **VAE decode (TT)** | 🟡 Phase B scaffold | Structure runs; PCC pending; flag `False` |
| **Full music e2e perf test** | 🟡 Phase C skeleton | `test_e2e_music_perf_traced_acestep.py` |
| **Production step count** | ⚠️ Mismatch | Gate/tests use **4 steps**; production uses **~30** |

**P150 baseline (latents only, 4 steps, trace+2CQ):** ~0.19 s total (~0.035 s/denoise step).
Rough extrapolation to 30 steps (no CFG): ~1.0–1.2 s denoise + ~0.05 s prefill → **within reach of 2 s** once VAE + live preflight are on TT.

---

## Architecture principle (trace + 2CQ e2e)

Full trace e2e does **not** require every box on device. It requires:

1. **Heavy subgraphs traced on TT:** `TracedConditionEncoder2CQ`, `TracedAudioPath2CQ`, `TracedDecoder2CQ`
2. **Host runs the scheduler loop** (ODE Euler, batching for CFG) — same as Flux/SD pipelines
3. **Blue blocks = glue**, not missing TT compute — see updated diagram `docs/acestep_v15_e2e_dataflow_ttnn_status.mmd`

---

## Milestone map (sequential gates, parallelizable file work)

```
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5 ──► M6
 │      │      │      │      │      │      │
 │      │      │      │      │      │      └─ A100-class e2e perf + demo CLI
 │      │      │      │      │      └─ TT Oobleck decode (Phase B)
 │      │      │      │      └─ Live ref-audio → refer_audio (VAE encode)
 │      │      │      └─ Live text → embeddings (Qwen3 TT)
 │      │      └─ Host-VAE full music e2e (Phase A2 + C host path)
 │      └─ Fix trace-capture hangs / stabilize CI
 └─ Current: latents e2e green
```

---

## M0 — Stabilize traced path (blocker cleanup)

**Problem:** `test_e2e_generate_audio_traced.py` / trace capture intermittently hangs (>900 s).

**Tasks:**
- [ ] Reproduce hang with `pytest ... -s -v --timeout=900` under flock; capture Tracy if needed
- [ ] Verify trace region size, 2CQ device params (`trace_region_size=50M`, `num_command_queues=2`)
- [ ] Ensure single device owner (kill stale pytest/flock holders)
- [ ] Document tmux recipe for long runs (Cursor-spawned jobs die on IDE exit)

**Exit:** Traced e2e pytest completes reliably in < 5 min.

---

## M1 — Listenable music e2e (host VAE path)

**Purpose:** Validate **TT latents → WAV** before TT VAE port.

**Already done:**
- `models/demos/hf_eager/acestep_v15_base/tt/vae_host.py`
- `test_e2e_generate_audio_with_vae_host.py` (golden latents PASS)
- `demo_generate_audio_wav.py`
- `test_e2e_music_perf_traced_acestep.py` skeleton

**Tasks:**
- [ ] Run **A2:** `test_e2e_tt_latents_host_vae` (device + flock)
- [ ] Run **Phase C host VAE:** `ACESTEP_USE_TT_VAE=0 ACESTEP_SAVE_WAV=1` perf test
- [ ] User gate: listen to `/tmp/acestep_phase_a_tt.wav` and `/tmp/acestep_phase_c.wav`

**Exit:** Printed `e2e_music_s` with finite waveform; user confirms audio is musical.

---

## M2 — Live text conditioning (Qwen3-Embedding-0.6B on TT)

**Today:** `build_inputs(use_captured=True)` loads frozen `text_hidden_states` / `lyric_hidden_states` from `_captured/`. Pipeline logs: *prompts ignored*.

**Target API:**
```python
pipeline(
    prompts=["upbeat electronic dance track"],
    lyrics=["..."],
    reference_audio=None,  # M3
    ...
)
```

**Tasks:**
- [ ] Add `models/tt_dit/pipelines/acestep/text_encode.py` (or reuse `models/demos/qwen3_embedding/`)
- [ ] Wire ACE-Step processor/tokenizer (same chat template as HF `AceStepConditionGenerationModel`)
- [ ] TT forward via `tt_transformers` for **Qwen3-Embedding-0.6B** (umbrella ckpt `Ace-Step1.5`)
- [ ] Optional: trace text encoder if prefill is measurable (>50 ms); else host embed is acceptable if total e2e < 2 s
- [ ] PCC test: TT embeddings vs torch on fixed prompt set
- [ ] Replace captured path in `build_inputs` when `use_captured=False`
- [ ] Update `pipeline_acestep.py` to pass live `prompts` / lyrics

**Exit:** Changing prompt text changes output latents (sanity); embedding PCC ≥ 0.99.

---

## M3 — Live reference audio (cover mode)

**Today:** `refer_audio_acoustic_hidden_states_packed` and `src_latents` come from captures; `is_covers=1` in gate config.

**Target:** User WAV → Oobleck **encoder** → `refer_audio` + `src_latents` for cover/timbre.

**Tasks:**
- [ ] Host-first: `vae_host.encode_reference(wav_path)` using torch Oobleck encoder (unblocks live UX fast)
- [ ] TT port: Oobleck **encoder** (or encode-only subgraph) — can follow decoder port patterns
- [ ] Wire `reference_audio: path | tensor` into pipeline; set `is_covers` from presence of ref
- [ ] Validate cover mode: output timbre follows reference (ear test + latent stats vs HF)
- [ ] Document max ref length (gate: 100 frames @ 25 Hz ≈ 4 s latent horizon; align with prod)

**Exit:** Same prompt, different reference WAV → audibly different output; HF parity on fixed clip.

---

## M4 — Production sampling (30 steps + CFG + APG)

**Today:** 4 infer steps, no CFG, no APG — fine for PCC gate, not for quality/speed comparison vs A100.

**Tasks:**
- [ ] Set production default `num_inference_steps=30` (configurable; keep gate=4 for fast CI via env `ACESTEP_E2E_INFER_STEPS`)
- [ ] Wire **CFG** using `models/tt_dit/pipelines/cfg.py` — **2× batched DiT forward/step** (cond + uncond)
- [ ] Port **APG/ADG** from HF `apg_guidance.py` — host vector math on velocities (keep on host)
- [ ] Extend `TracedDecoder2CQ` for CFG batch dim or dual trace replay
- [ ] PCC/quality check at 30 steps vs HF reference (lower bar than 4-step gate if stochastic)
- [ ] Profile denoise step time at 30 steps; target **< ~1.5 s** denoise on p150 to leave budget for preflight + VAE

**Exit:** 30-step TT output matches HF perceptually; perf table at production steps.

---

## M5 — TT Oobleck VAE decode (Phase B complete)

**Today:** Scaffold — `OOBLECK_DECODER_PORT_COMPLETE = False`.

**Tasks:**
- [ ] Fix **ConvTranspose1d** padding parity vs torch
- [ ] Fix **dilated ResUnit** center-crop boundaries
- [ ] Pass `test_vae_oobleck_decoder.py` PCC ≥ 0.99
- [ ] Set `OOBLECK_DECODER_PORT_COMPLETE = True`
- [ ] Run Phase C with `ACESTEP_USE_TT_VAE=1`
- [ ] Optional: trace VAE decode if runtime > 100 ms

**Exit:** Full stack TT latents + TT VAE → WAV; VAE decode time in perf table.

---

## M6 — Full A→Z demo + A100-class performance

**Deliverables:**
- [ ] CLI demo: `prompt + lyrics + reference_wav → out.wav` (single flock command)
- [ ] Perf test reports:

  ```
  End-to-end music (text encode + ref encode + DiT + VAE)
  Reference Nvidia A100: < 2.0 s
    preflight | latent_gen | vae_decode | total_e2e
  ```

- [ ] All hot paths: `traced=True`, `use_2cq=True` on p150
- [ ] Regression caps in pytest (~20% slack over recorded baseline)

**Performance budget (illustrative @ 30 steps, 1×1 p150):**

| Stage | Budget | Basis |
|-------|--------|-------|
| Text + ref encode | 0.15–0.30 s | Qwen TT + VAE encode (or amortized trace) |
| Prefill (A+B+D) | 0.05 s | Measured ~0.05 s @ 4-step gate |
| Denoise ×30 (+CFG) | 0.90–1.40 s | ~0.035 s/step × 30 × ~1.3 CFG overhead |
| VAE decode | 0.10–0.25 s | Host today; TT target similar or better |
| **Total** | **< 2.0 s** | A100 reference |

If over budget: prioritize **2CQ overlap on decoder**, **CFG batching**, **reduce D2H per step** (keep `xt` device-resident), **trace VAE**.

**Exit:** User-facing demo works; mean `total_e2e` ≤ 2.0 s on p150 (or documented gap with optimization backlog).

---

## Test matrix (manual gates)

| Gate | Command | Device |
|------|---------|--------|
| Latents PCC | `pytest .../test_e2e_generate_audio_traced.py -s -v` | flock |
| Host VAE | `pytest .../test_e2e_generate_audio_with_vae_host.py -k tt_latents_host_vae -s -v` | flock |
| TT VAE component | `pytest .../test_vae_oobleck_decoder.py -s -v --timeout=900` | flock |
| Music e2e perf | `ACESTEP_USE_TT_VAE=0 ACESTEP_SAVE_WAV=1 pytest .../test_e2e_music_perf_traced_acestep.py -k 1x1 -s -v` | flock |
| Full TT VAE perf | `ACESTEP_USE_TT_VAE=1 ...` (after M5) | flock |
| Live prompt e2e | TBD `test_e2e_live_prompt_acestep.py` (add in M2) | flock |

**Env:**
```bash
cd /local/ttuser/dvartanians/ace/tt-metal
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
flock -n /tmp/tt_ace_device.lock echo FREE || echo BUSY
```

---

## Parallel work (safe while device busy)

| Track | Can proceed without device |
|-------|---------------------------|
| M2 text encode | Processor + unit tests with CPU golden |
| M4 CFG/APG | Host-side math + HF parity tests |
| M5 Oobleck layers | Padding/crop fixes, structure tests |
| Diagrams/docs | ✅ Updated TTNN status diagram |
| M3 host encode | Torch encoder wrapper |

**Never parallelize** flock-guarded pytest on the same p150.

---

## Out of scope (follow-ups)

- All three **`acestep-5Hz-lm-{0.6B,1.7B,4B}`** planners (see **Phase 7** and [awesome-ace-step](https://github.com/ace-step/awesome-ace-step))
- VAE encode on TT before host encode unblocks UX
- Tiled decode for >48 s audio (`decode_overlap`)
- Multi-GPU / 2×2 mesh scaling
- Hard CI gate at A100 parity (informational reference only)

---

## Key files

| Area | Path |
|------|------|
| TT pipeline entry | `models/tt_dit/pipelines/acestep/pipeline_acestep.py` |
| Traced hot path | `models/demos/hf_eager/acestep_v15_base/tt/pipeline.py` |
| 2CQ overlap | `models/demos/hf_eager/acestep_v15_base/tt/traced_pipeline.py` |
| Input builder | `models/demos/hf_eager/acestep_v15_base/tt/common.py` |
| Host VAE | `models/demos/hf_eager/acestep_v15_base/tt/vae_host.py` |
| TT VAE scaffold | `models/tt_dit/models/audio_vae/vae_oobleck.py` |
| Music perf | `models/tt_dit/tests/models/acestep/test_e2e_music_perf_traced_acestep.py` |
| Diagram | `docs/acestep_v15_e2e_dataflow_ttnn_status.mmd` |
