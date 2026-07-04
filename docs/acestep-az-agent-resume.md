# ACE-Step A→Z — Agent resume guide

Use this document to **deploy parallel agents** and **resume bring-up** from the current checkpoint without re-discovering context.

**Branch:** `dvartanians/feature/tt-hw-planner`
**Last known good commit:** `8c8052a3b8` (2026-07-03)
**Hardware:** p150 Blackhole, 1×1 mesh
**Goal:** Text prompt + reference WAV → listenable music WAV on the **full TT hot path** (TT text + TT DiT + TT VAE; host ref encode only for A→Z).

**Related docs:**
- Full phase spec: `docs/superpowers/plans/2026-07-03-acestep-az-phases.md`
- Condensed overview: `docs/acestep-az-phases-summary.md`
- Agent file matrix: `docs/acestep-az-agent-deploy.md`
- Session Q&A (auto-up scope, architecture): `docs/acestep-session-qa-2026-07-03.md`
- Progress log (may lag): `docs/acestep-az-progress.md`

---

## 1. Resume checkpoint — what is done

| Area | Status | Evidence |
|------|--------|----------|
| Phase 1 G0 | ✅ Done | TT latents → host VAE WAV; G0 tests pass |
| Phase 2 live inputs | ✅ Done | Host `text_encode.py`, ref audio in `vae_host.py`, `pipeline_acestep.py` |
| DiT Calls A/B/D/C on TT | ✅ Done | Graduated stubs; PCC @ 4 steps with captured inputs |
| Long-sequence DiT parity | ✅ Done | Sliding-window attention (`tt/mask_utils.py`); Phase 3.4 tests @ T=750 |
| Cover conditioning | ✅ Fixed | See §3 — critical for listenable cover output |
| Phase 3 prep | 🔄 Partial | `apg_guidance.py`, `denoise.py` wired; **30-step + CFG quality gate pending** |
| Phase 4 TT VAE code | 🔄 Partial | `OOBLECK_DECODER_PORT_COMPLETE=True`; **device PCC gate pending** |
| Phase 5 CLI skeleton | 🔄 Partial | `demo_acestep_az.py` produces listenable WAV at 8 steps / no CFG |
| Phase 2C TT text | ⏳ Not started | Still host `Qwen3-Embedding-0.6B` |
| Phase 7 LM planner | ⏳ Not started | No `lm_planner.py` |
| Phase 6/8 trace + perf | ⏳ Deferred | Trace capture hangs — do not block functional work |

**Interim demo command (known good for parity / listen test):**

```bash
export REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
PY="$REPO/python_env/bin/python"

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

flock /tmp/tt_ace_device.lock $PY -m models.tt_dit.pipelines.acestep.demo_acestep_az \
  --prompt "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm" \
  --lyrics "$LYRICS" \
  --reference /tmp/ref_kaazoom_25s.wav \
  --output /tmp/az_nocfg_8s_shift3.wav \
  --audio-duration 30 --infer-steps 8 --guidance-scale 1.0 --shift 3.0 --seed 42
```

**A/B reference WAVs (lab machine):**

| File | Role |
|------|------|
| `/tmp/diffusers_cover_30s.wav` | Quality bar (turbo diffusers) |
| `/tmp/hf_nocfg_8s_shift3_covers0.wav` | HF CPU reference |
| `/tmp/az_nocfg_8s_shift3.wav` | TT after cover + long-seq fixes |

---

## 2. Remaining work — execution order

**Do not wait on trace or LM for functional milestones.**

```
Priority:  4 (verify) → 3 (CFG@30) → 2C (TT text) → 5 (signoff) → 7 (LM) → 6/8 (trace/perf)
```

| Phase | Agent window | Deliverable | Manual gate |
|-------|--------------|-------------|-------------|
| **4** | `phase4-vae` + `device` | TT Oobleck decode PCC ≥ 0.99 | Listen: TT VAE WAV ≈ host on same latents |
| **3** | `phase3-sampler` + `device` | 30 ODE steps + CFG + APG | Quality vs HF @ 30 steps, same prompt+ref |
| **2C** | `phase2c-tt-text` + `device` | TT `Qwen3-Embedding-0.6B` via `tt_transformers` | PCC ≥ 0.99 vs host; pipeline flag on |
| **5** | integration + `device` | Full TT demo signoff | TT text + TT DiT + TT VAE @ 30 steps → listenable WAV |
| **7** | `phase7-lm` | `acestep-5Hz-lm-{0.6B,1.7B,4B}` on TT | 1.7B required; quality vs tokenizer path |
| **6** | `m0-trace` | DiT trace + 2-CQ stable | Traced e2e completes in < 15 min |
| **8** | `m0-trace` | Full stack trace; < 2 s e2e | After Phase 7 |

**Out of scope for now:** TT Oobleck encoder (host ref encode is enough), `tt_hw_planner auto-up --stack`, turbo DiT variant (`acestep-v15-xl-turbo`) unless explicitly requested.

---

## 3. Critical context agents must not regress

### Cover conditioning (HF + TT)

These fixes are required for listenable cover output. Re-breaking any of them causes ringing / noise:

| Setting | Correct value | Wrong (breaks audio) |
|---------|---------------|----------------------|
| Reference WAV | **Timbre only** | Same WAV as both src + timbre |
| `src_latents` | **Learned `silence_latent`** from diffusers checkpoint | Encoded from reference |
| Text instruction | **Cover instruction** (`COVER_DIT_INSTRUCTION`) | text2music mask-fill |
| `audio_duration` | **`--audio-duration`** (default 30 s) | 2 s or ref length |
| `is_covers` | **`0`** for reference-only cover | `1` (tokenizes silence → ringing) |

**Key files:** `models/demos/hf_eager/acestep_v15_base/tt/vae_host.py`, `tt/pipeline.py`, `models/tt_dit/pipelines/acestep/pipeline_acestep.py`

### Long-sequence DiT attention

HF DiT + timbre encoder use **bidirectional sliding-window attention** (`window=128`, alternating layers). TT stubs must use the same masks, not full attention.

- **Invisible at T=50** (seq ≤ 128); **breaks at 30 s** (DiT patchified seq=375, timbre refer seq=750).
- **Fix location:** `models/demos/hf_eager/acestep_v15_base/tt/mask_utils.py`, `_stubs/ace_step_di_t_model.py`, `_stubs/ace_step_timbre_encoder.py`
- **Gate tests:** `models/tt_dit/tests/models/acestep/test_phase34_long_seq.py` (PCC ~0.997 @ T=750)

### Device serialization

**Only one job** may hold the device at a time:

```bash
flock -n /tmp/tt_ace_device.lock echo FREE || echo BUSY
```

After a hung trace test: `tt-smi -r 0` before retrying device tests.

### Default flags

- **`traced=False`** for all Phases 2–5 and 7 functional work.
- **`ACESTEP_USE_TT_VAE=1`** is the default in `audio_decode.py` once Phase 4 gate passes.
- CFG is disabled when `traced=True` (Phase 6 work).

---

## 4. tmux session bootstrap

Run once when starting a new agent session:

```bash
export REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole

# Create session (safe if already exists: attach instead)
tmux new-session -d -s acestep-az -n phase4-vae 2>/dev/null || true
tmux new-window -t acestep-az -n phase3-sampler 2>/dev/null || true
tmux new-window -t acestep-az -n phase2c-tt-text 2>/dev/null || true
tmux new-window -t acestep-az -n phase5-demo 2>/dev/null || true
tmux new-window -t acestep-az -n phase7-lm 2>/dev/null || true
tmux new-window -t acestep-az -n device 2>/dev/null || true
tmux new-window -t acestep-az -n m0-trace 2>/dev/null || true

# Check device lock
flock -n /tmp/tt_ace_device.lock echo "device FREE" || echo "device BUSY — wait"
```

Tail agent logs: `bash docs/acestep-az-window-watch.sh {4|3|2c|5|7|device|m0}`

---

## 5. Agent deployment matrix

**Rule:** Parallel agents must **not edit the same files**. Integration agent runs **after** parallel agents merge.

### Parallel-safe agents (CPU / file work)

| Window | Phase | Scope | Exclusive files | Do NOT touch |
|--------|-------|-------|-----------------|--------------|
| `phase4-vae` | 4 verify | Run device gate; fix decoder PCC if failing | `models/tt_dit/models/audio_vae/`, `test_vae_oobleck_decoder.py` | `pipeline_acestep.py` until gate green |
| `phase3-sampler` | 3 | CFG loop quality; 30-step HF parity test | `denoise.py`, `apg_guidance.py`, `test_e2e_live_inputs_acestep.py` (prod profile) | traced decoder |
| `phase2c-tt-text` | 2C | TT Qwen3-Embedding | new `text_encode_tt.py` or TT branch; extend `test_text_encode_acestep.py` | `lm_planner.py` |
| `phase7-lm` | 7A/7B | 5Hz LM planner | new `lm_planner.py`, `test_e2e_lm_planner_acestep.py` | DiT stubs, VAE |
| `phase5-demo` | 5 polish | CLI/docs only after 2C+3+4 | `demo_acestep_az.py`, README snippets | pipeline core until integration |

### Serial agents

| Window | When | Scope |
|--------|------|-------|
| `device` | After code lands | **One pytest/demo at a time** under `flock` |
| Integration | After 2C + 3 + 4 code | Wire `pipeline_acestep.py`: TT text, TT VAE, CFG defaults |
| `m0-trace` | After Phase 5 signoff | Phase 6 then 8; `docs/acestep-m0-trace-run.sh` |

---

## 6. Copy-paste agent prompts

Paste these into Cursor agents (one agent per window). Each agent should log to its `/tmp/acestep_agent_*.log`.

### Agent 4 — TT VAE device gate (start here)

```
Repo: /local/ttuser/dvartanians/ace/tt-metal
Branch: dvartanians/feature/tt-hw-planner

Task: Complete Phase 4 device gate for TT Oobleck decoder.

Context: OOBLECK_DECODER_PORT_COMPLETE=True in vae_oobleck.py but vae_oobleck.py
checklist still says PCC pending. Run test_vae_oobleck_decoder.py on device under flock.

If PCC < 0.99, fix oobleck_layers.py / vae_oobleck.py only. Then run full pipeline with
ACESTEP_USE_TT_VAE=1 and compare WAV to host decode on same latents.

Gate: test_vae_oobleck_decoder.py PASS; listen test TT VAE ≈ host VAE.
Log: /tmp/acestep_agent_4.log
Do not touch DiT stubs or cover conditioning in vae_host.py.
Read docs/acestep-az-agent-resume.md §3 before editing.
```

### Agent 3 — Production sampler (CFG @ 30 steps)

```
Repo: /local/ttuser/dvartanians/ace/tt-metal

Task: Complete Phase 3 — production sampler quality gate.

Context: apg_guidance.py and denoise.py exist. CFG runs sequential 2× DiT when
guidance_scale > 1 and traced=False. demo_acestep_az.py exposes --guidance-scale,
--infer-steps, --shift.

Work:
1. Add/extend prod profile in test_e2e_live_inputs_acestep.py (-k prod) at 30 steps + CFG.
2. Compare output quality vs HF reference on fixed prompt+ref (PCC or listen).
3. Keep ACESTEP_E2E_INFER_STEPS=4 for fast CI; document env override.

Gate: 30-step + CFG output acceptable vs HF; no regression on test_phase34_long_seq.py.
Log: /tmp/acestep_agent_3.log
Do not enable traced CFG (Phase 6).
```

### Agent 2C — TT Qwen3-Embedding

```
Repo: /local/ttuser/dvartanians/ace/tt-metal

Task: Phase 2C — wire Qwen3-Embedding-0.6B on TT via models/tt_transformers/.

Context: Host golden is text_encode.py (Phase 2A). Reuse tt_transformers Qwen3
patterns — integration only, not a new arch port.

Work:
1. Add text_encode_tt.py (or TT branch in text_encode.py).
2. Map ACE-Step processor output → TT forward → text_hidden_states, lyric_hidden_states.
3. Extend test_text_encode_acestep.py with -k tt device PCC vs host.
4. Add pipeline flag ACESTEP_USE_TT_TEXT_ENCODE=1 / use_tt_text_encode=True.

Gate: TT embeddings PCC ≥ 0.99 vs host on fixed prompts.
Log: /tmp/acestep_agent_2c.log
Exclusive files: text_encode*.py, test_text_encode_acestep.py
```

### Agent 5 — Integration (run after 2C + 3 + 4)

```
Repo: /local/ttuser/dvartanians/ace/tt-metal

Task: Phase 5 integration — full TT A→Z signoff.

Wire pipeline_acestep.py:
- ACESTEP_USE_TT_TEXT_ENCODE=1 (from 2C)
- TT VAE default (from 4)
- 30 infer steps + CFG (from 3)
- Host ref encode stays on host (vae_host.py)

Run demo_acestep_az.py under flock with production settings:
  --infer-steps 30 --guidance-scale 7.0 --audio-duration 30 --use-tt-vae

Gate: One command, your prompt + ref → listenable music; no _captured/ required.
Log: /tmp/acestep_agent_device.log
Read docs/acestep-az-agent-resume.md §3 (cover conditioning) — do not regress.
```

### Agent 7 — 5Hz LM planner (after Phase 5 interim demo)

```
Repo: /local/ttuser/dvartanians/ace/tt-metal

Task: Phase 7 — acestep-5Hz-lm-{0.6B,1.7B,4B} via tt_transformers Qwen3.

Phase 7A first (host LM golden), then 7B (TT LM). Default variant: 1.7B.
NOT the same as Phase 2A Qwen3-Embedding — this is the audio-code planner.

Work:
1. Create lm_planner.py — load LM, chat template, generate audio_codes.
2. Map audio_codes → quantizer → detokenizer → lm_hints_25Hz.
3. Add --use-lm-planner --lm-model {0.6B,1.7B,4B} to demo_acestep_az.py.
4. When use_lm_planner=True, skip Call B tokenizer.

Reference: awesome-ace-step LM table; modeling_acestep_v15_base.py prepare_condition.
Log: /tmp/acestep_agent_7.log
```

### Agent M0 — Trace + perf (deferred)

```
Repo: /local/ttuser/dvartanians/ace/tt-metal

Task: Phase 6 — fix trace capture hang; enable traced e2e on DiT stack only.

Known issue: hangs at "capturing trace..." in test_e2e_generate_audio_traced.py.
Use docs/acestep-m0-trace-run.sh. traced=False work must not be blocked.

Do NOT run in parallel with functional device tests — separate tmux window m0-trace.
Log: /tmp/acestep_agent_m0.log
```

---

## 7. Device gate commands (serial — `device` window)

Shared setup:

```bash
export REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
PY="$REPO/python_env/bin/python"
```

### Sanity — long-seq parity (run after any DiT stub edit)

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_phase34_long_seq.py \
  -s -v --timeout=3600
```

### Phase 4 — TT VAE

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_vae_oobleck_decoder.py \
  -s -v --timeout=900

export ACESTEP_USE_TT_VAE=1 ACESTEP_SAVE_WAV=1
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_live_inputs_acestep.py \
  -s -v --timeout=3600
```

### Phase 3 — production sampler

```bash
export ACESTEP_USE_TT_VAE=0 ACESTEP_SAVE_WAV=1
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_live_inputs_acestep.py \
  -k "prod" -s -v --timeout=7200
```

### Phase 2C — TT text encode

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_text_encode_acestep.py \
  -k tt -s -v --timeout=900
```

### Phase 5 — full demo signoff

```bash
bash docs/acestep-az-phase5-run.sh
# → /tmp/az_phase5_signoff.wav
```

Or inline:

```bash
export ACESTEP_USE_TT_VAE=1
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
flock /tmp/tt_ace_device.lock $PY -m models.tt_dit.pipelines.acestep.demo_acestep_az \
  --prompt "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm" \
  --lyrics "$LYRICS" \
  --reference /tmp/ref_kaazoom_25s.wav \
  --output /tmp/az_phase5_signoff.wav \
  --infer-steps 30 --guidance-scale 7.0 --shift 3.0 \
  --audio-duration 30 --seed 42 \
  --use-tt-vae --use-tt-text-encode --no-traced
```

### Phase 7 — full production demo (TT LM planner)

```bash
bash docs/acestep-az-phase7-run.sh
# → /tmp/az_lm_tt.wav
```

Device gates (prefill + generation smoke):

```bash
flock /tmp/tt_ace_device.lock $PY -m pytest \
  models/tt_dit/tests/models/acestep/test_lm_planner_tt_acestep.py \
  -s -v --timeout=3600
```

### Phase 6 — trace (deferred)

```bash
bash docs/acestep-m0-trace-run.sh
```

---

## 8. Key file map

| Component | Path |
|-----------|------|
| A→Z demo CLI | `models/tt_dit/pipelines/acestep/demo_acestep_az.py` |
| TT pipeline (Calls A–D) | `models/tt_dit/pipelines/acestep/pipeline_acestep.py` |
| ODE + CFG + APG | `models/tt_dit/pipelines/acestep/denoise.py`, `apg_guidance.py` |
| Host text encode | `models/tt_dit/pipelines/acestep/text_encode.py` |
| TT text encode (TODO) | `text_encode_tt.py` (to be added) |
| VAE decode routing | `models/tt_dit/pipelines/acestep/audio_decode.py` |
| TT Oobleck decoder | `models/tt_dit/models/audio_vae/vae_oobleck.py`, `oobleck_layers.py` |
| Host ref encode + cover prep | `models/demos/hf_eager/acestep_v15_base/tt/vae_host.py` |
| Eager TT pipeline | `models/demos/hf_eager/acestep_v15_base/tt/pipeline.py` |
| Sliding-window masks | `models/demos/hf_eager/acestep_v15_base/tt/mask_utils.py` |
| DiT / timbre stubs | `models/demos/hf_eager/acestep_v15_base/_stubs/` |
| Long-seq gate tests | `models/tt_dit/tests/models/acestep/test_phase34_long_seq.py` |
| Live input e2e | `models/tt_dit/tests/models/acestep/test_e2e_live_inputs_acestep.py` |
| Qwen3 reuse | `models/tt_transformers/` |
| LM planner (TODO) | `models/tt_dit/pipelines/acestep/lm_planner.py` |

---

## 9. Agent orchestration checklist

Use this when deploying a fresh agent batch:

```
[ ] git checkout dvartanians/feature/tt-hw-planner && git pull
[ ] Read docs/acestep-az-agent-resume.md (this file)
[ ] flock -n /tmp/tt_ace_device.lock → FREE (or kill stale pytest, tt-smi -r 0)
[ ] tmux session acestep-az up (§4)
[ ] Deploy Agent 4 first (VAE gate) — blocks Phase 5 signoff
[ ] Deploy Agent 3 + 2C in parallel (non-overlapping files)
[ ] Hold Agent 5 (integration) until 3 + 4 (+ ideally 2C) code merged
[ ] Run test_phase34_long_seq.py after any DiT/mask edit
[ ] Run demo listen test after pipeline integration
[ ] Deploy Agent 7 only after Phase 5 interim demo green
[ ] Agent M0 (trace) only in m0-trace window — never parallel with device gates
[ ] Update docs/acestep-az-progress.md when a phase gate passes
```

---

## 10. Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Ringing / noise on cover | `is_covers=1` or wrong `src_latents` | §3 cover table |
| PCC ~0.57 @ 30 s audio | Full attention instead of sliding window | `mask_utils.py` + stub masks |
| `OOBLECK_DECODER_PORT_COMPLETE` skip | PCC gate not run | Agent 4 |
| Trace hang at capture | Phase 6 blocker | `m0-trace` only; keep `traced=False` |
| Device BUSY | Stale pytest | Kill process; `tt-smi -r 0` |
| Pre-commit push fail | File > 500 KB | Compress PNG or skip (see session commit notes) |

---

## 11. Finish lines

| Milestone | Criteria |
|-----------|----------|
| **Phase 5 signoff** | TT text + TT DiT + TT VAE; 30 steps + CFG; listenable WAV from `demo_acestep_az.py` |
| **Phase 7 signoff** | `--use-lm-planner --lm-model 1.7B` on TT; quality vs tokenizer path |
| **Phase 8 signoff** | Full stack traced e2e; `total_e2e` recorded; target < 2 s vs A100 |

**Quality stretch (separate track):** Port `acestep-v15-xl-turbo` for diffusers-matching quality — not required for base Phase 5 signoff.
