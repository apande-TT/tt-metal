# ACE-Step v1.5 Full E2E Music + Perf — 3-Phase Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver three manually testable phases that culminate in a pytest producing listenable music and end-to-end wall-clock timing (latent gen + VAE decode), with A100 <2s noted as informational reference.

**Architecture:** DiT pipeline (`AceStepPipelineTT` / `AceStepPipeline`) emits `target_latents` `[B,T,64]`. ACE-Step VAE is **`AutoencoderOobleck`** (~169M params) in `ACE-Step/Ace-Step1.5/vae/` — separate from `acestep-v15-base`. Decode: transpose to `[B,64,T]`, `vae.decode(...).sample` → stereo PCM `[B,2,samples]`.

**Tech stack:** tt-metal p150 Blackhole, hf_eager graduated stubs, tt_dit pipeline, diffusers `AutoencoderOobleck`, pytest, `flock /tmp/tt_ace_device.lock`.

**Device constraint:** ONE on-device process at a time. Parallel agent work must be file-level (scaffold, host VAE, docs) until lock is free.

---

## Phase overview (manual gates)

| Phase | Deliverable | Manual gate | Device needed? |
|-------|-------------|-------------|----------------|
| **A** | TT latents → **host** Oobleck → `.wav` + timing | User listens to WAV; checks timing print | Yes (latents) |
| **B** | **TT** Oobleck decoder, PCC vs torch | Component pytest PASS (PCC ≥ 0.99) | Yes |
| **C** | Full TT stack + traced perf table incl. VAE | E2e perf pytest prints music e2e time | Yes |

Phases are **sequential for manual validation** (A before B before C). Implementation work parallelizes where noted.

---

## Shared env (all phases)

```bash
cd /local/ttuser/dvartanians/ace/tt-metal
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
flock -n /tmp/tt_ace_device.lock echo FREE || echo BUSY
```

VAE weights (already cached):

```
~/.cache/huggingface/hub/models--ACE-Step--Ace-Step1.5/snapshots/*/vae/
```

---

## Phase A — Host VAE bridge (listenable music fast)

**Purpose:** Unblock audible validation without waiting for TT VAE port.

### Files to create/modify

| File | Responsibility |
|------|----------------|
| `models/demos/hf_eager/acestep_v15_base/tt/vae_host.py` | Load Oobleck, `latents_to_waveform(target_latents)` |
| `models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_with_vae_host.py` | Gate A pytest |
| `models/demos/hf_eager/acestep_v15_base/demo/demo_generate_audio_wav.py` | Optional CLI demo writing `.wav` |

### Task A1: `vae_host.py`

- `load_oobleck_vae()` → `AutoencoderOobleck.from_pretrained(vae_path, torch_dtype=float32)`
- `latents_btc_to_waveform(latents: Tensor [B,T,C])` → transpose `[B,C,T]`, decode, return `[B,2,samples]`
- `save_wav(path, waveform, sample_rate=48000)` via `torchaudio` or `scipy.io.wavfile`
- Env override: `ACESTEP_VAE_PATH` for checkpoint dir

### Task A2: Pytest `test_e2e_generate_audio_with_vae_host.py`

Two subtests (user can run separately):

1. **`test_vae_host_decode_golden_latents`** — NO device. Load HF golden latents from e2e run or `hf_generate_reference`, decode on host, assert finite waveform, save `/tmp/acestep_phase_a_golden.wav`
2. **`test_e2e_tt_latents_host_vae`** — device. Run `AceStepPipelineTT.generate` (or traced path), time latent gen, host VAE decode, print:

```
PHASE_A latent_gen_s=...
PHASE_A vae_decode_s=...
PHASE_A e2e_music_s=...
PHASE_A output_wav=/tmp/acestep_phase_a_tt.wav
```

Assert: latent PCC ≥ 0.99 (reuse e2e gate), waveform peak < 10 (sanity), file written.

### Manual test — Phase A

```bash
# A1: VAE-only (no device)
./python_env/bin/python -m pytest \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_with_vae_host.py \
  -k golden_latents -s -v

# A2: TT latents + host VAE (device)
flock /tmp/tt_ace_device.lock ./python_env/bin/python -m pytest \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_with_vae_host.py \
  -k tt_latents_host_vae -s -v

# Listen
aplay /tmp/acestep_phase_a_tt.wav   # or ffplay / scp locally
```

**Phase A exit criteria:** WAV sounds like music (not noise/static); timing lines printed.

---

## Phase B — TT Oobleck decoder port

**Purpose:** Move VAE decode onto Tenstorrent; validate per-component PCC before full integration.

### Files to create

| File | Responsibility |
|------|----------------|
| `models/tt_dit/models/audio_vae/vae_oobleck.py` | TT `OobleckDecoder` module |
| `models/tt_dit/models/audio_vae/oobleck_layers.py` | Snake1d, ResUnit, DecoderBlock (reuse `audio_ops.Snake`, Conv1dViaConv3d) |
| `models/tt_dit/tests/models/acestep/test_vae_oobleck_decoder.py` | PCC vs torch diffusers decoder |

### Task B1: Layer inventory (reference)

Torch decoder (~169M params): 32× Conv1d, 5× ConvTranspose1d, 36× Snake1d, 15× ResUnit.
Reuse patterns from `models/tt_dit/layers/audio_ops.py` (LTX vocoder already has Snake + conv1d).

### Task B2: Bottom-up port order

1. Snake1d → wrap existing `Snake` / `SnakeBeta`
2. OobleckResidualUnit
3. OobleckDecoderBlock (upsample + res units)
4. Full `OobleckDecoder.forward`

### Task B3: Component test

- Input: random `[1,64,T]` bfloat16 on device
- Reference: torch decoder on CPU fp32
- Assert PCC ≥ 0.99 on output waveform (or intermediate if staged)
- `--timeout=900`, single 1x1 mesh

### Manual test — Phase B

```bash
flock /tmp/tt_ace_device.lock ./python_env/bin/python -m pytest \
  models/tt_dit/tests/models/acestep/test_vae_oobleck_decoder.py -s -v --timeout=900
```

**Phase B exit criteria:** Component test PASS; PCC printed ≥ 0.99.

---

## Phase C — Full TT e2e perf (music timing)

**Purpose:** Single pytest: traced TT latents + TT VAE → timing table matching user goal.

### Files to create/modify

| File | Responsibility |
|------|----------------|
| `models/tt_dit/tests/models/acestep/test_e2e_music_perf_traced_acestep.py` | Full stack perf |
| `models/tt_dit/pipelines/acestep/pipeline_acestep.py` | Optional: `decode_waveform=True` flag calling TT VAE |
| `models/tt_dit/pipelines/acestep/audio_decode.py` | Thin wrapper: TT Oobleck + host normalize |

### Task C1: Extend pipeline

After `target_latents`, if `decode_waveform=True`:
- Transpose → TT Oobleck decode → optional peak normalize (match DiffSynth `-1 dB`)
- Return `{target_latents, waveform, timings}`

### Task C2: Perf pytest

Extend `test_e2e_perf_traced_acestep.py` pattern:

```
End-to-end music generation time (latent + VAE)
Reference (Nvidia A100 full stack): < 2.0s (informational)
  latent_gen | vae_decode | total_e2e
```

Warmup + 4 measured runs; assert waveform finite.

### Manual test — Phase C

```bash
flock /tmp/tt_ace_device.lock ./python_env/bin/python -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_music_perf_traced_acestep.py \
  -k 1x1 -s -v --timeout=3600
```

**Phase C exit criteria:** Table prints `total_e2e` mean; WAV optional save via env `ACESTEP_SAVE_WAV=1`.

---

## Parallel agent dispatch map

| Agent | Phase | Can run parallel with | Needs device |
|-------|-------|----------------------|--------------|
| Agent 1 | A — implement `vae_host.py` + pytest | B scaffold, C skeleton | Only for A2 test run |
| Agent 2 | B — scaffold `vae_oobleck.py` + layers + test stub | A, C | No (write only); yes to run B test |
| Agent 3 | C — skeleton perf test + pipeline hook spec | A, B | No (write only) |

**Do NOT run A2, B, C device tests concurrently** — serialize under `flock`.

---

## Dependencies

- Phase B manual test: independent of A (can use random latents), but A gives confidence latents are good
- Phase C manual test: requires B component PASS
- Current resume queue (Task A traced, Task D perf) may hold device — wait for `FREE` before Phase A2

---

## Out of scope (this plan)

- TT port of VAE **encoder** (only decoder needed for generation)
- `decode_overlap` tiled decode for >48s audio (follow-up)
- Committing `.pre_e2e` backups
- Gating P150 timing against A100 2s (informational only)
