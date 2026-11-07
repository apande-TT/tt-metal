# GLM-5.3-Flash

## Platforms
Blackhole Galaxy (GalaxyBH, 32x Blackhole), one 1x32 mesh: tensor parallel TP=32, data parallel DP=1.

## Introduction
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a natively multimodal
image-text-to-text model: a 24-block vision tower feeds a 45-layer hybrid language model. The 45 layers are
34 Kimi delta-attention (linear) layers and 11 DeepSeek sparse-attention layers, with manifold-constrained
hyper-connections, 3 dense MLP layers and 42 MoE layers (288 routed experts plus 1 shared expert, top-8).
Here it runs end to end on device: image + prompt in, generated text out.

## Prerequisites
- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) repository for source code.
- Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- The `zai-org/GLM-5.3-Flash` checkpoint in the Hugging Face cache (FP8, ~306 GB on disk).

## How to Run
Demo (the published example: cat image + "Describe the image."):
```bash
python -m models.demos.glm_5_3_flash.demo.demo_image_text_to_text
```

Another image / prompt:
```bash
python -m models.demos.glm_5_3_flash.demo.demo_image_text_to_text --image <url-or-path> --prompt "What is in this picture?"
```

32 samples per call, each sampled with its own seed (generation_config temperature / top_p):
```bash
python -m models.demos.glm_5_3_flash.demo.demo_image_text_to_text --batch 32 --sample
```

End-to-end correctness gate (batch from `$TT_PERF_BATCH`):
```bash
TT_PERF_BATCH=32 ./python_env/bin/python -m pytest -p no:timeout models/demos/glm_5_3_flash/tests/e2e/test_e2e_image_text_to_text.py -s
```

Trace-capture and host-op self-tests:
```bash
./python_env/bin/python -m pytest -p no:timeout models/demos/glm_5_3_flash/tests/e2e/test_e2e_trace.py -s
```

Per-component PCC tests from bring-up:
```bash
./python_env/bin/python -m pytest models/demos/glm_5_3_flash/tests/pcc -s
```

## Details
- Entry point: `run_image_text_to_text(pipe, encoded, ...)` in `tt/pipeline.py`. The resident pipeline
  object comes from `build_pipeline(device, model=None, layers=None, **kwargs)` in the same file. The demo
  and `tests/e2e` both call these.
- The chain is built entirely from the graduated ports in `_stubs/`:
  - vision: `vision_model` = `vision_patch_embed`, `vision_rotary_embedding`, then 24 x `vision_block`
    (each `r_m_s_norm`, `vision_attention`, `vision_m_l_p`), then `vision_patch_merger`.
  - text: `text_model` = 45 x `text_decoder_layer` + `text_hyper_head` + `text_r_m_s_norm`. Each layer
    is built from:
    - `text_hyper_connection` with its `text_unweighted_r_m_s_norm`
    - `text_r_m_s_norm`
    - attention: `text_linear_attention` (with `text_forget_gate` and `text_r_m_s_norm_gated`) or
      `text_attention` (with `text_indexer`)
    - MLP: `text_m_l_p` or `text_mo_e` (`text_topk_router`, `text_experts`, and `text_m_l_p` as the shared
      expert)
- TP=32 placement:
  - heads are sharded across chips (linear attention, sparse attention, indexer, vision attention);
  - MoE is expert-parallel, 9 experts per chip;
  - MLPs are column/row parallel;
  - collectives are `all_gather` + an on-device sum;
  - embeddings, norms and lm_head are replicated.
- Decode keeps the KDA recurrent and conv state and the sparse-attention KV cache resident on device. The
  capacity is `index_topk` (2048) positions; below that, the indexer's selection is exactly causal.
- Weights and config come from the Hugging Face checkpoint `zai-org/GLM-5.3-Flash`. The bf16 model (~640 GB)
  is larger than host RAM, so `tt/checkpoint.py` streams one decoder layer at a time from the FP8 shards. It
  applies HF's own key renames and FP8 dequantisation, and is bit-identical to `from_pretrained`.
- Goldens (`reference/hf_reference.py`) run HF's own `Glm5Next*` module classes in bf16, with the sdpa and grouped_mm
  implementations that `from_pretrained` selects, streamed layer by layer.

### Inputs
- The published example of `Glm5NextForConditionalGeneration` (cat image + "Describe the image."),
  scaled to an input size of 1024 (`DEFAULT_AREA`: 1024 x 1024 pixels in total, aspect ratio kept) and
  encoded by the HF processor's chat template. That gives a 1382-token prompt. `TT_PERF_AREA` (the
  optimize/emit-e2e `--area` flag) overrides the size; `TT_PERF_AREA=0` runs the image as published
  (893 tokens). Verified at batch 32 on the Galaxy at both sizes.
- A copy is kept under `_captured/e2e_example/` for the trace stages.
- Change the input with `--image`, `--prompt`, `--max-image-tokens` and `--reasoning-effort`.
- With `--batch N --sample`, every row carries the same input and row `b` samples with seed
  `b + --seed-offset`.

### Trace replay
Not yet recorded: trace-replay output is copied here from the trace-replay run.

## Results
See [RUN_REPORT.md](RUN_REPORT.md) for the measured gate results for this demo.
