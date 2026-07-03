# ACE-Step A→Z — Agent deployment matrix

**Rule:** Only one job may hold `flock /tmp/tt_ace_device.lock` at a time.

## Parallel-safe (run now)

| Window | Phase | Agent scope | Files (exclusive) | Log |
|--------|-------|-------------|-------------------|-----|
| `phase2a-text` | 2A | Live text encode | `text_encode.py`, `common.py` (build_inputs) | `/tmp/acestep_agent_2a.log` |
| `phase2b-ref` | 2B | Ref audio encode | `vae_host.py`, ref-audio tests | `/tmp/acestep_agent_2b.log` |
| `phase4-vae` | 4 | TT Oobleck decoder | `vae_oobleck.py`, decoder tests | `/tmp/acestep_agent_4.log` |
| `phase3-sampler` | 3 prep | CFG/APG host math | new `apg_guidance.py`, `cfg` wiring stubs | `/tmp/acestep_agent_3.log` |
| `phase5-demo` | 5 prep | CLI skeleton | new `demo_acestep_az.py` only | `/tmp/acestep_agent_5.log` |

## Serial / idle (do not parallelize)

| Window | Phase | Why idle |
|--------|-------|----------|
| `phase1` | 1 G0 | **Done** — soft PASS; monitoring only |
| `device` | 1–5 tests | **One device job at a time** — waits for code from 2A/2B/4 |
| `m0-trace` | 6 | **Deferred** — trace hangs; last per plan |

## After parallel agents finish

One **integration agent** (sequential) wires `pipeline_acestep.py`:
- prompts from 2A
- reference_audio from 2B
- Then device window runs integration pytest.

## Commands

```bash
# See agent logs in any terminal
tail -f /tmp/acestep_agent_2a.log

# Switch tmux tab
az go 2a

# Start log tail in a window (already done by deploy script)
bash docs/acestep-az-window-watch.sh 2a
```
