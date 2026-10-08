# Kolibri-1

## Platforms
QB2: four Blackhole p300c chips as a 1x4 mesh (tensor parallel over the four chips, FABRIC_1D).

## Introduction
[Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) is a German/English mixture-of-experts reasoning
model: 50 decoder layers (sliding-window RoPE attention interleaved 4:1 with full-attention NoPE layers), 384 routed
SwiGLU experts per layer with top-6 sigmoid routing plus one shared expert, shipped as FP8 weights in 128x128 blocks.
This demo runs its text generation (chat prompt -> sampled answer) entirely on device for a batch of 32 users: prefill,
KV-cached decode and generation_config's sampling rule (top-k 128, top-p 0.97, temperature 1.0).

## Prerequisites
- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) repository for the source code.
- Installed TT-Metalium / TT-NN: see [INSTALLING.md](../../../INSTALLING.md).
- The checkpoint (~78 GB) and tokenizer download from Hugging Face on first use (`huggingface-cli login` if needed).

## How to Run
Text generation demo (the model card's example question, 32 users, seed b on user b):
```bash
python -m models.demos.kolibri_1.demo.demo_text_generation
```

Another prompt, fewer users:
```bash
python -m models.demos.kolibri_1.demo.demo_text_generation --message "Was ist die Hauptstadt von Frankreich?" --batch 8
```

End-to-end correctness test (prints `PERF_BATCH_STREAMS` and `e2e PCC`):
```bash
TT_PERF_BATCH=32 pytest -p no:timeout models/demos/kolibri_1/tests/e2e/test_e2e_text_generation.py -s
```

Precompute the independent reference generation the test compares against (fp32 on CPU; the test builds it on a
cache miss too):
```bash
python -m models.demos.kolibri_1.tests.e2e.golden
```

Trace-capture and host-op selftests (each opens the mesh itself; `KOLIBRI_SELFTEST_LAYERS` sets their depth):
```bash
python -c "from models.demos.kolibri_1.tt.pipeline import trace_capture_selftest as t; print(t())"
python -c "from models.demos.kolibri_1.tt.pipeline import host_op_selftest as h; print(h())"
```

## Details
- Entry point: `build_pipeline(device, model=None, layers=None, **kwargs)` in `tt/pipeline.py` returns the resident
  `KolibriPipeline`; `generate(prompt_ids, seeds)` runs prefill + decode until every user hits a stop token. The demo
  and the e2e test both call it. `layers` caps the decoder depth (a capped build keeps both layer kinds).
- The forward is composed from the graduated bring-up stubs in `_stubs/` (see `tt/model.py`): `token_embed`,
  `decoder_layer`, `attention` (with resident K,V caches), `f_p8_linear` (the attention output projection),
  `sparse_moe_block` (routed experts), `router`, `m_l_p` (the shared expert), `r_m_s_norm` (final norm) and
  `decoder_head` (fp32 logits).
- Weights and config come from the Hugging Face checkpoint `Aleph-Alpha/Kolibri-1`, read one decoder layer at a time
  through the reference model's own classes (`tests/pcc/_reference_loader.py`).
- Goldens are built against that reference: a torch port of the aleph-alpha-inference vLLM plugin (Kolibri-1 is not a
  transformers model type), run in fp32 with the FP8 weights dequantised per call (`tests/e2e/golden.py`).
- Generation stops on generation_config's `eos_token_id` (127906 `<|im_end|>`, 127901); the safety cap is the KV
  capacity, 2048 positions per user (`KV_CAPACITY` in `tt/pipeline.py`, sized from QB2's registered 28.6 GB usable per
  chip with a CCL axis in play).

### Inputs
The user message is the model card's own example ("Erkläre kurz, was ein Mixture-of-Experts-Modell ist.", from the
card's "Querying the server" section) rendered with the tokenizer's chat template and `enable_thinking=False` (the
card's documented way to answer without a reasoning trace). Every user gets the same prompt; user b samples with seed
b. Change the message with `--message`, the batch with `--batch` (or `TT_PERF_BATCH`), and the seeds with
`--seed-offset`; the defaults live in `tt/inputs.py`.

### Trace replay
Copied from the trace-replay output (`agent/trace_replay.py::measure_adapter` over `build_pipeline`, all 50 layers,
batch 32, 1x4 mesh):

| Stage | `TRACE_STAGE_MS` | path |
|---|---|---|
| prefill | `TRACE_STAGE_MS[prefill]=1665.5914` | `trace+1cq` |
| decode | `TRACE_STAGE_MS[decode]=364.8680` | `trace+1cq` |

`TRACE_PER_TOKEN_MS=364.8680`, `TRACE_HEADLINE_UNIT=token`.

## Results
See [RUN_REPORT.md](RUN_REPORT.md) for the measured gate results for this demo.
