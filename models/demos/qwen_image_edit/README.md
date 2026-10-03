# Qwen-Image-Edit

## Platforms
T3K (8 x Wormhole, four n300 boards), opened as a 2x4 mesh with `FABRIC_1D`.

## Introduction
[Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) edits an image from a text
instruction. A Qwen2.5-VL encoder (32-block vision tower + 28-layer LM) reads the image and the
instruction. A 60-block MMDiT transformer denoises the image latents over a FlowMatch Euler schedule,
conditioned on that encoding and on the VAE latents of the input image. A causal 3D-conv VAE decodes
the result. This demo runs the whole chain in TTNN: vision tower, LM, VAE encoder, every denoising
step and the VAE decoder. 32 independent samples run per call.

## Prerequisites
- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) repository for source code
- Installed TT-Metalium / TT-NN: see [INSTALLING.md](../../../INSTALLING.md)
- Hugging Face access to `Qwen/Qwen-Image-Edit` (the weights, processor, scheduler and the HF
  reference pipeline are read from the hub cache)

## How to Run
Demo: the published example, 32 seeds, PNGs written to `demo/output/`:
```bash
python -m models.demos.qwen_image_edit.demo.demo_image_edit
```

Fewer samples, your own image and instruction:
```bash
python -m models.demos.qwen_image_edit.demo.demo_image_edit --batch 1 --image photo.png --prompt "Turn the sky purple"
```

Demo with a per-sample PCC against the HF float32 golden (the golden is built on CPU if it is not cached):
```bash
python -m models.demos.qwen_image_edit.demo.demo_image_edit --batch 4 --compare-golden
```

E2E test (Gates 1-3, prints `e2e PCC=`); the batch comes from `$TT_PERF_BATCH` (default 32):
```bash
TT_PERF_BATCH=32 pytest -p no:timeout models/demos/qwen_image_edit/tests/e2e/test_e2e_image_edit.py -s
```

Build the HF golden ahead of the test (float32 on CPU, cached in chunks of 4 seeds):
```bash
python -m models.demos.qwen_image_edit.reference.golden --batch 32
```

Trace contract (each stage captured and replayed, forward with zero host ops, depth knobs):
```bash
pytest -p no:timeout models/demos/qwen_image_edit/tests/test_pipeline_contract.py -s
```

Per-stage trace replay:
```bash
TT_PERF_BATCH=32 pytest -p no:timeout models/demos/qwen_image_edit/tests/e2e/test_image_edit_perf.py -s
```

## Details
- Entry point: `run_image_edit(pipe, enc)` in `tt/pipeline.py`. The demo and the e2e test both call
  it, so they run the same wiring. `build_pipeline(device, model=None, layers=None,
  vision_encode_layers=None, text_encode_layers=None, denoise_layers=None)` builds and returns the
  resident pipeline object, which carries `PIPELINE_STAGES = [vision_encode, text_encode, vae_encode,
  denoise, vae_decode]` and their `<stage>_trace_*` hooks.
- Weights and config come from the Hugging Face `Qwen/Qwen-Image-Edit` checkpoint (diffusers
  `QwenImageEditPipeline`, loaded in float32). The pipeline object keeps it as `pipe.hf`.
- The graduated TTNN modules come from three bring-ups:
  `models/demos/qwen_image_edit_text_encoder/_stubs` (7),
  `models/tt_dit/pipelines/qwen_image_edit_transformer/_stubs` (7) and
  `models/tt_dit/pipelines/qwen_image_edit_vae/_stubs` (11). All 25 run in the forward; the e2e test
  counts them.
- Mesh split: the text encoder is TP=4 over the columns, with its LM row-staged (layers 0-13 on row 0,
  14-27 on row 1). The transformer is TP=8 over all 8 chips. The VAE splits W over the columns and the
  batch over the rows, at 16 images per program.
- Goldens: the HF `QwenImageEditPipeline` itself, float32 on CPU, run on the same encoded inputs and
  per-sample generators (`reference/golden.py`). It records the prompt embeddings, the image latents
  and every step's latents to localise a failure.
- Denoising: the 50 steps follow the scheduler's own schedule. Step 0 runs eagerly; steps 1..49
  replay one captured device trace over persistent buffers. Every step is synchronised and logged as
  it completes.

### Inputs
`tt/inputs.py` holds the model's published example in one block, with each value's source:
`yarn-art-pikachu.png` and the prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome',
yarn art style, detailed, vibrant colors", with 50 steps (QwenImageEditPipeline docstring example 1).
Every sample uses that content, and sample `i` uses seed `0 + i` (seed 0 comes from the model README's
example), so sample 0 reproduces the published run. The example passes no negative prompt, so true
CFG is off on both sides. HF sizes the condition image to a 1024x1024 area. Here both sides use a
256x256 area (`DEFAULT_AREA`), because a float32 CPU golden for 32 samples at 1024x1024 is out of
reach. Change the content with the demo's `--image`, `--prompt`, `--seed`, `--steps` and `--area`,
and the batch with `--batch` or `$TT_PERF_BATCH`.

### Trace replay
Not yet reported by a trace-replay run of this pipeline (`tests/e2e/test_image_edit_perf.py`).

## Results
See [RUN_REPORT.md](RUN_REPORT.md) for the measured gate results for this demo.
