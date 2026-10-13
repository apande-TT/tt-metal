# Devstral-Small-2-24B-Instruct-2512

## Platforms
QB2 (4x Blackhole p300c), run as one 1x4 mesh: tensor-parallel TP=4 x DP=1, `FABRIC_1D`.
With only 2 chips reachable, `mesh.open_mesh` falls back to a 1x2 mesh (TP=2) and the resident KV cache
holds 2048 positions per row instead of 4096 (`tt.pipeline.kv_capacity_for`), so the per-chip KV budget is
unchanged. One chip cannot hold the 24B bf16 weights.

## Introduction
[mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)
is a 24B agentic-coding model (`Mistral3ForConditionalGeneration`): a 40-layer Ministral3 text decoder (GQA 32/8
heads, yarn RoPE with llama-4 query scaling, vocab 131072) behind a 24-layer Pixtral vision encoder and a
patch-merging projector. This demo runs both of its task heads end to end in TTNN — text generation
(text -> text) and image + text -> text — prefilling a batch of 32 rows and decoding token by token with a
resident KV cache and on-device sampling.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- The HF checkpoint in the local HF cache (`huggingface-cli download mistralai/Devstral-Small-2-24B-Instruct-2512`)

## How to Run
Text generation demo:
```bash
python -m models.demos.devstral_small_2_24b_instruct_2512.demo.demo_text_generation
python -m models.demos.devstral_small_2_24b_instruct_2512.demo.demo_text_generation --prompt "Write a bash one-liner that counts lines in all .py files."
```

Image + text demo:
```bash
python -m models.demos.devstral_small_2_24b_instruct_2512.demo.demo_image_text_to_text
python -m models.demos.devstral_small_2_24b_instruct_2512.demo.demo_image_text_to_text --image path/to/screenshot.png --prompt "What does this code do?"
```

A different batch (<= 32 rows, one sampling seed per row):
```bash
python -m models.demos.devstral_small_2_24b_instruct_2512.demo.demo_text_generation --batch 8
```

E2E tests (correctness gate, PCC vs the HF golden, all 32 rows):
```bash
TT_PERF_BATCH=32 ./python_env/bin/python -m pytest -p no:timeout models/demos/devstral_small_2_24b_instruct_2512/tests/e2e/test_e2e_image_text_to_text.py models/demos/devstral_small_2_24b_instruct_2512/tests/e2e/test_e2e_text_generation.py -s
```

Only the declared correctness gate:
```bash
TT_PERF_BATCH=32 ./python_env/bin/python -m pytest -p no:timeout models/demos/devstral_small_2_24b_instruct_2512/tests/e2e/test_e2e_image_text_to_text.py::test_e2e_image_text_to_text -s
```

Trace replay (per-stage latency; each stage set up from its `<stage>_trace_inputs()`, captured and replayed):
```bash
TT_PERF_BATCH=32 TT_TRACE_REPLAY_ITERS=8 ./python_env/bin/python -m pytest -p no:timeout models/demos/devstral_small_2_24b_instruct_2512/tests/e2e/test_text_generation_perf.py -s
```

Trace-capture and fully-on-device self-checks:
```bash
TT_PERF_BATCH=32 ./python_env/bin/python -c "from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P; print(P.trace_capture_selftest())"
TT_PERF_BATCH=32 ./python_env/bin/python -c "from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P; print(P.host_op_selftest())"
```

Per-component PCC of the graduated stubs (TP=4). These read the bring-up's captured HF tensors
(`_captured/<component>/*.pt`), which are not committed: run them where the bring-up capture ran.
```bash
./python_env/bin/python -m pytest -p no:timeout models/demos/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_model_sharded.py -s
```

## Details
- Entry point: `build_pipeline(device, model=None, layers=None, **kwargs)` in `tt/pipeline.py` returns the
  resident pipeline; `run_text_generation` and `run_image_text_to_text` in the same file are the chained
  forward that both the demos and the e2e tests call. `PIPELINE_STAGES = ["encode", "prefill", "decode"]`.
- The forward is composed only of the graduated bring-up stubs in `_stubs/` (vision tower, projector, text model,
  decoder layers, attention, MLPs, norms, rotary embeddings, token embedding, lm_head). Each stub shards its own
  weights over the 4 chips and runs its own all_reduce / all_gather; embeddings, norms and rotary tables are
  replicated.
- Weights and config are loaded from the HF hub checkpoint via `transformers` (`AutoModelForImageTextToText`)
  by `reference/checkpoint.py::load_reference_model`. On a host without a GPU, transformers would dequantize the
  checkpoint's FP8 text-decoder linears to bf16 (~48 GB of host RAM); the loader keeps them as FP8 codes + their
  per-tensor scale (~22 GB, file-backed) and rebuilds the bf16 weight on access with transformers' own dequant
  arithmetic, so the weights the TT build uploads and the CPU reference computes with are bit-identical to
  transformers' dequantized load (checked on load). Everything is uploaded to the mesh once.
- Goldens are built against the same HF reference running `model.generate()` on CPU in bf16
  (`tests/e2e/hf_reference.py`), cached under `~/.cache/devstral_e2e_golden` (override with
  `DEVSTRAL_GOLDEN_CACHE`).
- Sampling follows the checkpoint's `generation_config` (temperature 0.15, EOS 2, pad 11) through the Gumbel-max
  form over the top-32 logits, with a per-row seeded noise table resident on device; the HF golden applies the
  identical rule, and both sides decode until every row emits EOS (the KV capacity of 4096 positions is only a
  safety cap, asserted not to bind).
- Memory (QB2 registered figures, 28.6 GB usable per chip with a CCL axis): text weights 44.4 GB / 4, the
  replicated token table 1.34 GB and a 40 x 2 x [32, 2, 4096, 128] bf16 KV cache (5.4 GB) per chip.
- Numerics: both residual streams (text decoder and vision encoder) are carried in fp32, with the attention /
  MLP output projections and their all_reduce in fp32; matmul inputs and weights are bf16. RMSNorms apply the
  checkpoint weight as a separate multiply after the canonical `models/common/rmsnorm.py` norm, and prefill
  SDPA uses the widest key chunk the sequence allows (prompts are right-padded to 512 positions), because the
  fused gamma and narrow key chunks each bias the output level, and the image-token rows of the text decoder
  amplify any level bias over 40 layers.

### Inputs
The inputs are TOOL-AUTHORED (no published example was found for this checkpoint) and live in
`demo/inputs.py`: one text prompt, and one 224x224 image of a two-line Python function with a question about it.
Every batch row gets the same content; rows differ only in their sampling seed (row index + `--seed-offset`).
Change them with `--prompt` / `--image` on the demos, or edit `demo/inputs.py` for the tests. The per-stage trace
hooks (`<stage>_trace_inputs` in `tt/pipeline.py`) encode these same inputs: the image request for `encode`, the
text request for `prefill` / `decode`.

### Trace replay
From `tests/e2e/test_text_generation_perf.py` (`TT_PERF_BATCH=32`, `TT_TRACE_REPLAY_ITERS=8`; printed
`TRACE_REPLAY_PATH=trace+1cq ... batch=32 warmup=3 iters=8`), every stage replayed from a captured trace:

| Stage | `TRACE_STAGE_MS` | path | `TRACE_STAGE_ITEMS` |
|---|---|---|---|
| encode | 249.3803 | `trace+1cq` | 8192 |
| prefill | 6064.1015 | `trace+1cq` | 16384 |
| decode | 55.7033 | `trace+1cq` | 32 |

`TRACE_PER_TOKEN_MS=55.7033`, `TRACE_HEADLINE_UNIT=token` (`TRACE_STAGE_REPEATS[decode]=127`).

## Results
See [RUN_REPORT.md](RUN_REPORT.md) for the measured gate results for this demo.
