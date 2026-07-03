# ACE-Step v1.5

## Introduction

[ACE-Step](https://huggingface.co/ACE-Step/acestep-v15-base) is a flow-matching diffusion transformer (DiT) for text-to-audio / music generation. Conditioning includes text, lyrics, and optional reference-audio timbre.

The tt_dit pipeline (`models/tt_dit/pipelines/acestep/pipeline_acestep.py`) is under active bring-up. Until it is wired end-to-end, the performance test exercises the graduated hf_eager TTNN pipeline (`models/demos/hf_eager/acestep_v15_base/tt/pipeline.py`) with the same stage timing hooks used by other tt_dit perf tests.

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

| System | Mesh | Infer Steps | Current Performance | Target Performance |
|--------|------|-------------|---------------------|--------------------|
| P150   | 1×1  | 4           | TBD                 | TBD                |

> Baselines will be filled after the first on-device perf run on P150.

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
Backend: hf_eager AceStepPipelineTT
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
