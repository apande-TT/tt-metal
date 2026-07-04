# ACE-Step A→Z — Agent deployment matrix

**Rule:** Only one job may hold `flock /tmp/tt_ace_device.lock` at a time.

## Parallel-safe (run now)

| Window | Phase | Agent scope | Files (exclusive) | Log |
|--------|-------|-------------|-------------------|-----|
| `phase2a-text` | 2A | Host text encode (done) | `text_encode.py` | `/tmp/acestep_agent_2a.log` |
| `phase2c-tt-text` | 2C | TT Qwen3-Embedding | `text_encode_tt.py` or TT branch; `tt_transformers` wiring | `/tmp/acestep_agent_2c.log` |
| `phase2b-ref` | 2B | Ref audio encode | `vae_host.py`, ref-audio tests | `/tmp/acestep_agent_2b.log` |
| `phase4-vae` | 4 | TT Oobleck decoder | `vae_oobleck.py`, decoder tests | `/tmp/acestep_agent_4.log` |
| `phase3-sampler` | 3 prep | CFG/APG host math | new `apg_guidance.py`, `cfg` wiring stubs | `/tmp/acestep_agent_3.log` |
| `phase5-demo` | 5 prep | CLI skeleton | new `demo_acestep_az.py` only | `/tmp/acestep_agent_5.log` |
| `phase7-lm` | 7A/7B | 5Hz LM planner | new `lm_planner.py`, LM tests; wire via `tt_transformers` Qwen3 | `/tmp/acestep_agent_7.log` |

## Serial / idle (do not parallelize)

| Window | Phase | Why idle |
|--------|-------|----------|
| `phase1` | 1 G0 | **Done** — soft PASS; monitoring only |
| `device` | 1–5, 7 tests | **One device job at a time** — waits for code from parallel agents |
| `m0-trace` | 6, 8 | **Deferred** — trace hangs; after Phase 5 full-TT demo / Phase 7 LM |

## Recommended execution order (remaining)

**4 → 3 → 2C → 5 → 7** — file work on 3/4/2C can overlap; device tests serial.

## After parallel agents finish

One **integration agent** (sequential) wires `pipeline_acestep.py`:
- TT text encode from Phase 2C (`ACESTEP_USE_TT_TEXT_ENCODE=1`)
- reference_audio from 2B (host ref encode — stays host)
- TT VAE from Phase 4
- optional `--use-lm-planner` from Phase 7
- Then device window runs integration pytest / `demo_acestep_az.py`.

## Phase 2C notes

- Wire **`Qwen3-Embedding-0.6B`** through existing **`models/tt_transformers/`** — integration, not new arch.
- Host `text_encode.py` (Phase 2A) remains golden reference for PCC.

## Phase 7 notes

- **Official LM variants** ([awesome-ace-step](https://github.com/ace-step/awesome-ace-step#language-models-planner)): `acestep-5Hz-lm-0.6B`, **`acestep-5Hz-lm-1.7B`** (default), `acestep-5Hz-lm-4B`. All Qwen3 base; same integration, different checkpoints.
- **Not** the same as Phase 2A: 2A = `Qwen3-Embedding-0.6B` (text conditioning); Phase 7 = 5Hz audio-code planners.
- **Qwen3 already implemented** in `models/tt_transformers/` (`Qwen3ForCausalLM`, generator). Phase 7B = load ACE-Step LM weights per variant + audio-code output.
- CLI/env: `--lm-model {0.6B,1.7B,4B}` / `ACESTEP_LM_PLANNER_MODEL=acestep-5Hz-lm-1.7B`.
- When `use_lm_planner=True`, skip Call B tokenizer; feed LM `audio_codes` → detokenizer → `lm_hints_25Hz`.

## Commands

```bash
# See agent logs in any terminal
tail -f /tmp/acestep_agent_2a.log

# Switch tmux tab
az go 2a

# Start log tail in a window (already done by deploy script)
bash docs/acestep-az-window-watch.sh 2a
```
