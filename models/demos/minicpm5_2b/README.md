# MiniCPM5-2B

## Platforms
Blackhole QB2 (p300c), run on ONE chip of the box (`TT_VISIBLE_DEVICES=0,1` selects board 0; the
two boards of this host have no ethernet link, so opening all four chips fails at fabric init).
Memory budget used: the registered 32 GB DRAM per chip, 29.2 GB usable at TP=1 (weights ~4.5 GB bf16,
KV cache ~0.2 GB and sampling noise ~1.1 GB at batch 32).

## Introduction
[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) is a 2B-parameter decoder-only
language model (`LlamaForCausalLM`: 42 layers, hidden 2048, 16 query / 2 KV heads, vocab 130560) with
a thinking-mode chat template. This demo runs text generation entirely on device -- token embedding,
RoPE, the 42-layer decoder stack with a resident KV cache, the final norm, the LM head and the
sampler (top-k 50 / top-p 0.95, seeded) -- for 32 independent samples per call.

## Prerequisites
- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) repository for source code.
- Installed TT-Metalium / TT-NN: see [INSTALLING.md](../../../../INSTALLING.md).
- Access to the HuggingFace checkpoint `openbmb/MiniCPM5-2B` (downloaded on first run).

## How to Run
Demo (the model card's example prompt, 32 seeded samples):
```bash
python -m models.demos.minicpm5_2b.demo.demo_text_generation
```

Demo with your own prompt, batch and seed:
```bash
python -m models.demos.minicpm5_2b.demo.demo_text_generation \
    --prompt "Write a haiku about silicon." --batch 8 --base-seed 7 --show 8
```

End-to-end correctness gate (TT generation vs HF `generate()`; prints `e2e PCC=`):
```bash
TT_PERF_BATCH=32 pytest -p no:timeout \
    models/demos/minicpm5_2b/tests/e2e/test_e2e_minicpm5_2b.py::test_generate_matches_hf_generate -s
```

Trace-capture and fully-on-device self-tests:
```bash
TT_PERF_BATCH=32 pytest -p no:timeout \
    models/demos/minicpm5_2b/tests/e2e/test_e2e_minicpm5_2b.py::test_trace_capture_selftest \
    models/demos/minicpm5_2b/tests/e2e/test_e2e_minicpm5_2b.py::test_host_op_selftest -s
```

Per-component PCC tests of the graduated stubs:
```bash
pytest -p no:timeout models/demos/minicpm5_2b/tests/pcc -s
```

## Details
- Entry point: `build_pipeline(device, model=None, layers=None, ...)` and
  `MiniCPM5Pipeline.run_text_generation(...)` in [tt/pipeline.py](tt/pipeline.py) -- the one chained
  forward both the demo and the e2e tests call. `PIPELINE_STAGES = ["prefill", "decode"]`, each with
  the `<stage>_trace_setup / _trace_step / _trace_inputs / _trace_items` hooks; `layers` caps the
  depth of the single repeated stack (`model.layers`).
- The forward chains the graduated stubs in [_stubs/](_stubs): `token_embed` -> `rotary_embedding`
  -> `encoder_stack` (42 blocks: `decoder_layer` on even layers, `layer` on odd layers, each holding
  `r_m_s_norm`, `attention`, `r_m_s_norm`, and `mlp` (even) / `m_l_p` (odd)) -> `r_m_s_norm` (final)
  -> `decoder_head`, then the on-device sampler.
- Weights and config are loaded from the HuggingFace hub (`openbmb/MiniCPM5-2B`) via
  `transformers.AutoModelForCausalLM`; the bf16 checkpoint is uploaded as bf16.
- Goldens ([tt/reference.py](tt/reference.py)) come from the HF reference model run in fp32:
  `model.generate()` with the checkpoint's own TopK/TopP warpers, eos ids `[1, 130073]` and the
  example's `max_new_tokens=128`. The random draw is a Gumbel-max with the same seeded noise the TT
  sampler adds, so one seed gives one sample on both sides.

### Inputs
[tt/inputs.py](tt/inputs.py) holds the model card's example (README.md example 1) verbatim in one
block: the chat message "Who are you? Please briefly introduce yourself.", `enable_thinking=True`,
`max_new_tokens=128`. All 32 rows use that prompt and differ only in the sampling seed
`BASE_SEED + i` (`BASE_SEED = 1234`). Change the prompt / batch / seed with the demo's `--prompt`,
`--batch` and `--base-seed`; the e2e tests read the batch from `$TT_PERF_BATCH`.

### Trace replay
Output of the trace replay (`measure_adapter` over `build_pipeline`, batch 32, one Blackhole chip):

| stage | TRACE_STAGE_MS | path |
|---|---|---|
| prefill | 583.9820 | trace+1cq |
| decode | 368.6648 | trace+1cq |

`TRACE_PER_TOKEN_MS=368.6648`, `TRACE_HEADLINE_UNIT=token`.

## Results
See [RUN_REPORT.md](RUN_REPORT.md) for the measured gate results for this demo.
