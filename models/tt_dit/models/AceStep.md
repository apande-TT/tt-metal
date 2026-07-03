# ACE-Step v1.5

## Introduction

[ACE-Step](https://huggingface.co/ACE-Step/acestep-v15-base) is a flow-matching diffusion transformer (DiT) for text-to-audio / music generation. Conditioning includes text, lyrics, and optional reference-audio timbre.

The tt_dit pipeline (`models/tt_dit/pipelines/acestep/pipeline_acestep.py`) delegates to the graduated hf_eager TTNN stubs and emits profiler sections for perf reporting.

## Performance

Performance is measured as seconds per audio-latent generation on a fixed e2e gate configuration (`infer_steps=4`, batch 1, captured Source-B inputs). Stages reported:

| Stage | Description |
|-------|-------------|
| Condition Encoder | Text + lyric + timbre conditioning |
| Audio Tokenizer | FSQ codec encode path |
| Detokenizer | FSQ decode → LM hints |
| Denoising | Flow-matching ODE loop (per-step + aggregate) |
| Total | End-to-end wall time |

### P150 (Blackhole, 1×1 mesh)

Measured 2026-07-03 on `sjc-snva-tp100` (tt_dit `AceStepPipeline`, trace + 2-CQ, JIT warm):

| System | Mesh | Infer Steps | Total (s) | Denoise (s) | Steps/s | Gen/s |
|--------|------|-------------|-----------|-------------|---------|-------|
| P150   | 1×1  | 4           | 0.189     | 0.138       | 28.9    | 5.30  |

Prior eager baseline (no trace): total 0.407 s, denoise/step 0.085 s, 2.46 gen/s.

Stage means (seconds, trace + 2-CQ): encoder 0.024, tokenizer 0.012, detokenizer 0.010, denoise/step 0.035.

Regression targets in the perf test allow ~20% slack above these baselines.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- HuggingFace access to `ACE-Step/acestep-v15-base`

## How to Run

```bash
# Install tt-metal (see INSTALLING.md in the repo root)

# Single P150 Blackhole device (1×1 mesh)
export ARCH_NAME=blackhole

# Optional: serialize on-device runs when sharing a lab machine
flock /tmp/tt_ace_device.lock ./python_env/bin/python -m pytest \
    models/tt_dit/tests/models/acestep/test_performance_acestep.py -s

# Filter to the 1×1 mesh configuration
pytest models/tt_dit/tests/models/acestep/test_performance_acestep.py -k 1x1 -s
```

### Expected console output

The test prints a summary table with mean / std / min / max for each stage:

```
================================================================================
ACE-STEP v1.5 PIPELINE PERFORMANCE RESULTS
================================================================================
Model: ACE-Step/acestep-v15-base
Inference Steps: 4
Backend: tt_dit AceStepPipeline
Mesh Shape: (1, 1)
--------------------------------------------------------------------------------
Condition Encoder         | Mean:   0.1234s | Std:   0.0012s | Min:   0.1220s | Max:   0.1250s
Audio Tokenizer           | Mean:   0.0456s | Std:   0.0008s | Min:   0.0445s | Max:   0.0465s
Detokenizer               | Mean:   0.0123s | Std:   0.0003s | Min:   0.0120s | Max:   0.0128s
Denoising (per step)      | Mean:   0.2345s | Std:   0.0045s | Min:   0.2300s | Max:   0.2400s
Total Pipeline            | Mean:   1.0500s | Std:   0.0100s | Min:   1.0400s | Max:   1.0650s
Run (wall clock)          | Mean:   1.0500s | Std:   0.0100s | Min:   1.0400s | Max:   1.0650s
--------------------------------------------------------------------------------
Average total denoising time: 0.9380s
Denoising throughput: 4.26 steps/second
Overall throughput: 0.9524 generations/second

Time breakdown:
  Encoder:     11.8%
  Tokenizer:   4.3%
  Detokenizer: 1.2%
  Denoising:   89.3%
================================================================================
```

In CI (`CI=true`), the test also writes a partial benchmark pickle under `generated/benchmark_data/`.
