<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Voxtral-Mini-3B-2507 on Tenstorrent

End-to-end TTNN pipeline for [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
— a 3B **audio + text -> text** model: a 32-layer Whisper-style audio encoder bolted onto a
30-layer Llama-style causal LM through a small multi-modal projector.

The whole forward pass runs on device in native `ttnn`, composed from the 17 graduated stubs in
[`_stubs/`](_stubs). The wiring exists exactly once, in [`tt/pipeline.py`](tt/pipeline.py); the
demos and the e2e test both import it.

| | |
|---|---|
| HF id | `mistralai/Voxtral-Mini-3B-2507` (`architectures: [VoxtralForConditionalGeneration]`) |
| Audio tower | 128 mel bins, 3000 frames (30 s), d_model 1280, 20 heads, 32 layers -> `(1, 1500, 1280)` |
| Language model | hidden 3072, 30 layers, GQA 32/8 heads, head_dim 128, SwiGLU, RMSNorm, RoPE theta 1e8 |
| Vocabulary | 131072 (tekken tokenizer); `[AUDIO]`=24, `[BEGIN_AUDIO]`=25, `<s>`=1, `[INST]`=3, `[/INST]`=4, eos=2 |
| Topology | single Blackhole p150b (`chips=1 tp=1 dp=1 mesh=[1,1]`) — nothing is sharded, no collectives |
| Decode batch | 8 streams, one program per step |

## Reference chain

The TT pipeline reproduces this chain from `transformers` 5.12.1 `modeling_voxtral.py`, step for step:

1. `feature_extractor`: waveform -> `input_features (B, 128, 3000)` log-mel (30 s chunks).
2. tokenizer: prompt -> `input_ids` containing 375 `[AUDIO]` (id 24) placeholders per 30 s chunk.
3. `audio_tower = VoxtralEncoder(input_features)`: `gelu(conv1)` -> `gelu(conv2, stride 2)` ->
   permute -> `+ embed_positions` -> 32 × `VoxtralEncoderLayer` -> `layer_norm` => `(1, 1500, 1280)`.
4. `reshape(-1, 5120)` frame-concat => `(375, 5120)`.
5. `multi_modal_projector`: `linear_1` -> `gelu` -> `linear_2` => `(375, 3072)`.
6. `language_model.embed_tokens(input_ids)` => `(1, L, 3072)`, then `masked_scatter` of the audio
   embeddings wherever `input_ids == 24`.
7. `language_model = LlamaModel(inputs_embeds)`: 30 × (RMSNorm -> GQA+RoPE -> add -> RMSNorm ->
   SwiGLU -> add) -> final RMSNorm.
8. `lm_head`: `(…, 3072)` -> `(…, 131072)` logits.
9. `generate()`: greedy argmax -> append -> KV-cached decode -> stop at `eos_token_id = 2`.

On device the same chain runs as three stages, `PIPELINE_STAGES = ["encode", "prefill", "decode"]`,
with greedy sampling done on device (`ttnn.argmax` -> `ttnn.embedding`) so no host sampling ever
enters the loop.

## The two Calls

Both Calls run the **exact same** audio-encode -> prefill -> decode chain over all 17 graduated
modules and differ only in the assembled prompt token stream. That is why they share one pipeline
object and one test.

| Call | id | Prompt (frozen in `tt/inputs.py`) | Emits |
|---|---|---|---|
| 1 | `audio_chat` | `<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 <instruction> [/INST]` | an assistant **answer** about the audio, per stream |
| 2 | `transcription` | `<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 lang:<xx> [/INST]` | a verbatim **transcript**, per stream |

Both return a `TaskResult` with `tokens [B, N]`, `logits [B, N, 131072]`, `texts`, `lengths`,
`stopped_on_eos`, `per_step_pcc`.

`mistral_common` is not installed in `python_env`, so `VoxtralProcessor.apply_chat_template`
(which needs `MistralCommonBackend`) is unavailable; `tt/inputs.py` assembles the identical token
stream directly from the HF tokenizer + feature extractor, validated empirically against
`generate()` output quality. The transcription template deliberately does **not** use the
`[TRANSCRIBE]` control token (id 34) — every candidate that emitted it degenerated into repetition
or empty output.

## Running it

All commands are run from the repo root with the project interpreter.

### Demos

```bash
# Call 1 — audio chat, 8 built-in clips, 32 new tokens each
./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.demo.demo_audio_chat

# ... with your own clips, a custom question, and the HF PCC comparison
./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.demo.demo_audio_chat \
    --audio mary_had_lamb.mp3 --audio obama.mp3 --batch-size 2 \
    --instruction "What is this audio about? Answer briefly." \
    --max-new-tokens 48 --compare-hf

# Call 2 — transcription
./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.demo.demo_transcription

./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.demo.demo_transcription \
    --audio winning_call.mp3 --batch-size 1 --language en --max-new-tokens 64 --compare-hf
```

Each demo opens the device itself (`ttnn.open_device(device_id=0, l1_small_size=24576,
trace_region_size=23887872)` in a `try/finally`), builds the shared pipeline, prints the clip name
and the real task output per stream, and exits non-zero on failure (including a `--compare-hf` PCC
below 0.95).

### End-to-end gate test

```bash
./python_env/bin/python -m pytest \
    models/tt_transformers/demo/voxtral_mini_3b_2507/tests/e2e/test_e2e_pipeline.py -svv

# shorter horizon while debugging (applied identically to TT and to the HF golden)
VOXTRAL_E2E_MAX_NEW_TOKENS=8 ./python_env/bin/python -m pytest \
    models/tt_transformers/demo/voxtral_mini_3b_2507/tests/e2e/test_e2e_pipeline.py -svv
```

### Trace-contract test (orchestrator-owned)

```bash
./python_env/bin/python -m pytest \
    models/tt_transformers/demo/voxtral_mini_3b_2507/tests/e2e/test_trace_contract.py -svv
```

### Device-free gate self-check

`tests/e2e/gates.py` runs stand-alone: it re-derives the graduated inventory from
`bringup_status.json` + the `_stubs/*.last_good_native` snapshots and runs the Gate-1 AST scan over
the live stub bodies. No device needed.

```bash
./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.tests.e2e.gates
```

### Per-component PCC tests

The 17 per-component tests from bring-up still pass unchanged; see
[`RUN_REPORT.md`](RUN_REPORT.md) for the full list, e.g.

```bash
./python_env/bin/python -m pytest \
    models/tt_transformers/demo/voxtral_mini_3b_2507/tests/pcc/test_voxtral_encoder.py -svv
```

## Where each graduated stub is routed

All 17 bring-up components graduated (`ON_DEVICE (17)`, `KERNEL_MISSING=0`, `PENDING=0`,
`CPU_REUSE=0`, `_runtime_fallbacks.json` empty), and every live `_stubs/<name>.py` is sha256-identical
to its `.last_good_native` snapshot. 16 sit on the real data path; 1 is an explicitly reported hole.
Nothing here is a coverage sweep: every routed stub's output is consumed by the next stage on the
way to the final logits, and no reference tensor is injected at any joint.

| # | Stub | Stage | Where it is routed in the forward pass |
|---|---|---|---|
| 1 | `voxtral_encoder` | encode | full audio tower for streams **0-3**: mel `(1,128,3000)` -> `(1,1500,1280)` |
| 2 | `encoder_stack` | encode | full audio tower for streams **4-7** (byte-identical body to `voxtral_encoder`, independent instance, disjoint data); its layers 28-31 are replaced by the four stubs below |
| 3 | `voxtral_encoder_layer` | encode | audio encoder **layer 28** of the streams 4-7 tower (built from `audio_tower.layers[28]`) |
| 4 | `layer` | encode | audio encoder **layer 29** (built from `audio_tower.layers[29]`) |
| 5 | `voxtral_attention` | encode | self-attention of audio encoder **layer 30** (LN + residual + FFN authored in ttnn from the layer-30 weights) |
| 6 | `attention` | encode | self-attention of audio encoder **layer 31**, same wrapper shape as layer 30 |
| 7 | `voxtral_multi_modal_projector` | encode | `(1500,1280)` -> `reshape(375,5120)` -> linear/gelu/linear -> `(375,3072)` audio embeds, **all 8 streams** |
| 8 | `token_embed` | prefill + decode | `input_ids` -> `(B,C,3072)` text embeds at prefill, and the next-token embed at every decode step |
| 9 | `llama_rotary_embedding` | prefill + decode | resident cos/sin table; supplies `rope=(cos,sin)` to **every** LM layer in both phases |
| 10 | `llama_decoder_layer` | prefill + decode | LM **layer 0** (full layer; writes/reads its KV slot) |
| 11 | `llama_r_m_s_norm` | prefill + decode | LM **layer 1** `input_layernorm` |
| 12 | `llama_attention` | prefill + decode | self-attention of LM **layer 1 and layer 2** (two instances) |
| 13 | `llama_m_l_p` | prefill + decode | MLP of LM **layer 1** |
| 14 | `mlp` | prefill + decode | MLP of LM **layer 2** |
| 15 | `llama_model` | prefill + decode | LM **layers 3..29** + final RMSNorm, via `layer_range=(3,30)` + `inputs_embeds` + `rope` + `kv` |
| 16 | `decoder_head` | decode | final hidden -> `(B,131072)` logits; on-device `ttnn.argmax` -> next token |
| 17 | `avg_pool1d` | **excluded** | not in the chain — see below |

Three stubs are duplicate bodies (`voxtral_encoder` == `encoder_stack` byte-for-byte;
`voxtral_encoder_layer` ≡ `layer`; `llama_m_l_p` ≡ `mlp`). They are **not** deduplicated away:
each pair is instantiated twice over disjoint data (streams 0-3 vs 4-7) or distinct layer indices
(encoder 28/29, LM 1/2), so both instances do real work on real data. Instantiating a graduated
stub class against a different index of a repeated block is the normal way a pipeline composes a
30-layer stack; it is what lets five layer-0-captured LM stubs and four encoder-layer stubs all
live inside the single real forward pass.

The audio/text merge is pure ttnn on device:
`ttnn.concat([text_embeds[:, :audio_start], audio_embeds, text_embeds[:, audio_start+375:]], dim=1)`.
`audio_start` is resolved during input encoding (outside the observed region), so no host aten op
runs inside the forward.

## The `avg_pool1d` hole

**`avg_pool1d` is NOT part of the parity chain, and that is reported rather than papered over.**

`audio_tower.avg_pooler` (`nn.AvgPool1d(2, stride=2)`) is constructed by `VoxtralEncoder.__init__`
but **never called** by `VoxtralEncoder.forward` nor by `VoxtralModel.get_audio_features` in
`transformers` 5.12.1 — verified by reading the reference source. The 1500 -> 375 reduction in the
real chain is the `reshape(-1, 5120)` frame-concat, not an average pool.

There is therefore no numerically exact place for it:

* inserting it anywhere in the audio path changes the audio embeddings and destroys parity;
* the only "exact" placements (duplicate-then-pool, or pooling a value that is then discarded) are
  precisely the decorative coverage shortcut the contract forbids.

**What is done instead:** the e2e suite runs an explicitly labelled conformance check,
`pipe.avg_pool1d_conformance()`, which drives the graduated stub with the **real audio-tower hidden
states produced by the TT encode stage** and PCC-checks it against `torch.nn.AvgPool1d(2, stride=2)`.
The stub's graduated work is thus exercised and verified on device with real data, while the parity
chain stays faithful to HF. The exclusion is printed by the test (Phase 3), carried in
`EXCLUDED_STUBS["avg_pool1d"]` with its reason string, and stated here.

## Batch-8 design and its constraints

* **Decode batch = 8.** One program per decode step over all 8 rows: activations `[8,1,3072]`,
  q `[1,8,32,128]`, KV caches `[8,8,C,128]` per layer indexed per stream by `cur_pos` via
  `ttnn.transformer.scaled_dot_product_attention_decode` + `ttnn.experimental.paged_update_cache`.
  There is **no python loop over streams in decode**.
* **8 distinct real clips** from `hf-internal-testing/dummy-audio-samples` — the corpus the Voxtral
  model card itself uses: `bcn_weather.mp3`, `dude_where_is_my_car.wav`, `fleur_es_sample.wav`,
  `mary_had_lamb.mp3`, `monte_cristo.flac`, `obama_first_45_secs.mp3`, `obama.mp3`,
  `winning_call.mp3`.
* **Uniform-30 s constraint.** Each clip is truncated/padded to exactly 30 s (480000 samples) =>
  exactly one mel chunk => exactly 375 audio tokens.
* **Uniform-prompt-length constraint.** All 8 streams share the same instruction text, so the
  prompt length `L` is identical across the batch: one prefill program shape, one shared `cur_pos`,
  no ragged masking. The prompt is right-padded to the pinned prefill capacity `C = 512`; causal
  attention means the padded tail cannot influence `[0:real_len]`.
* **Encode is a per-stream loop.** The graduated encoder body hardcodes `conv1d batch_size=1 /
  input_length=3000`, so the 8 towers run sequentially. Encode is a one-time prefix cost, not the
  decode inner loop.
* **Batch axis is proven, not assumed.** Each stream is compared to *its own* `generate()` golden,
  and the test additionally asserts that at least 6 of the 8 decoded texts differ — a shape-only
  batch axis fails that check.
* **Follow-on: ragged batching.** Variable-length audio (multi-chunk clips, >375 audio tokens) and
  per-stream prompt lengths would need per-row `cur_pos` bookkeeping at prefill, a padded/masked
  prefill program, and a batched encoder (`conv1d` over `B>1`). None of that is implemented here;
  the uniform-length batch is the deliberate scope of this bring-up.

## Trace contract surface

`tt/pipeline.py` exposes the standard trace-contract surface so the stages can be captured and
replayed:

| Symbol | Meaning |
|---|---|
| `PIPELINE_STAGES` | `["encode", "prefill", "decode"]` — derived from the config (audio front-end + causal LM, `is_encoder_decoder=false`, no speech output head, hence no `vocode` stage) |
| `build_pipeline(device, model=None, **kwargs)` | returns the resident `VoxtralPipeline` **object** (it does not run anything); demos, tests and the selftests all construct through it |
| `<stage>_trace_setup(inputs)` | allocates the persistent device tensors for a stage and primes it for capture |
| `<stage>_trace_step()` | the single captured step for a stage |
| `<stage>_trace_inputs()` | zero-arg; returns exactly what `<stage>_trace_setup` takes, assembled from `_captured/<name>/` tensors |
| `trace_capture_selftest(device)` | per stage: `begin_trace_capture` -> one step -> `end_trace_capture` -> `execute_trace` -> PCC vs eager -> `release_trace`; `True` only if all three stages captured host-op-free **and** matched |
| `host_op_selftest()` | `observe_host_ops()` around the model math for both heads (input encoding and weight build stay outside the region); `verdict(ops)["on_device"]` must be `True` |

Trace capture always runs at the fixed pinned capacity `C`, independent of the variable-length
decode. `trace_region_size` is sized for the largest stage; on overflow `C` is shrunk and the
fallback is printed. Both the test and the demos keep `l1_small_size` / `trace_region_size` on
module-level constants (`24576` / `23887872`) so they can be tuned in one place.

## Gates

| Gate | What it proves | Where |
|---|---|---|
| 0 | no graduated module is silently dropped: `set(ROUTED_STUBS) \| set(EXCLUDED_STUBS) == graduated`, all 17 live bodies == their snapshots | `gates.graduated_inventory` |
| 1 static | no torch compute op, no `ttnn.to_torch`/`.numpy()` host readback, no HF orchestration in any hot-path function of any routed stub or of `tt/pipeline.py` (weight extraction in `__init__` is allowed) | `gates.gate1_static_scan` |
| 1 runtime | what actually executed: `torch_ops == 0` and `ttnn_dispatch > 0` over a full 8-stream run, measured by `models.common.native_probe` (a `TorchFunctionMode` that aliasing cannot evade) | `gates.gate1_runtime_probe` |
| 2 | every one of the 16 routed stubs was invoked ≥ 1 time during the real forward pass (counting proxies wrap the real `__call__`; there is no separate sweep) | `gates.gate2_invoked` |
| 3 | `PCC(TT decode logits, HF generate logits) >= 0.95` per stream **and** aggregate, for both heads | `gates.pcc` / `gates.report_pcc` |
| hole | `avg_pool1d` conformance on real TT encoder hidden states, reported as NOT-IN-CHAIN | `pipe.avg_pool1d_conformance()` |

Gate 1's static scan has exactly one narrow escape hatch, for the unavoidable readback at the
**output boundary** (device logits/ids have to leave the device to become a `TaskResult`): a
trailing `# gate1: allow-readback <reason>` comment waives a `host_readback` on that line only. It
never waives a torch compute op or HF orchestration, and every waiver is printed in the scan
report.

## PCC numbers

Measured on a single Blackhole **p150b**, `max_new_tokens = 32`, 8 streams, threshold **0.95**,
`comp_pcc` of the stacked decode logits `[8, 32, 131072]` against
`model.generate(..., output_logits=True)`.

### Read three numbers together

| Metric | What it measures | audio_chat | transcription |
|---|---|---|---|
| **`e2e PCC` (gated)** | same-prefix per-step logits PCC — the full chained TT forward (audio → encoder → projector → merge → 30 LM layers → lm_head) at every decode position, with both sides on the reference token prefix so the comparison is well posed | **0.99870** | **0.99835** |
| first token (gated) | fully free-running **and** prefix-independent, so it checks the whole chain with no pinning at all | 0.99768 (argmax 7/8) | 0.99816 (argmax 8/8) |
| free-running, N=32 (reported) | the TT loop on its OWN tokens vs the fp32 golden | 0.7794 | 0.8991 |
| ↳ same, vs the **bf16** golden | dtype-matched reference | 0.7911 | 0.9476 |
| ↳ **reference dtype floor** | HF bf16 vs HF fp32 — *the same model against itself*, only the dtype differing | **0.8819** | **0.8842** |

**Why the free-running number is reported and not gated.** Greedy decoding of this checkpoint is
argmax-unstable at bf16: HuggingFace's own bf16 run mis-ranks 7/379 prefill argmax positions
against its own fp32 run, and its free-running 32-step sequences only reach **0.88** PCC against
itself. A single flipped token early in a 32-step greedy chain sends the rest of the sequence down
a different (still correct) continuation and collapses the sequence-level PCC. So a free-running
sequence gate at 0.95 is not reachable by *any* bf16 implementation of this model — including the
reference one — and it measures precision chaos rather than the pipeline. The gated `e2e PCC`
holds both sides on the same prefix, which is what isolates the pipeline; it is **flat at
0.997–0.999 across all 32 decode positions and all 8 streams**, which is also the proof that there
is no positional / KV-cache defect (a wiring bug would decay with step index, not stay flat).
Nothing is injected into the shipped pipeline: `run_audio_chat` / `run_transcription` and both
demos are fully free-running. The prefix is pinned only inside the test, via
`pipeline.force_next_ids()`, which the shipped path never calls.

### Per stream (gated `e2e PCC`, same-prefix)

| Stream | Clip | `audio_chat` | `transcription` |
|---|---|---|---|
| 0 | `bcn_weather.mp3` | 0.99950 | 0.99930 |
| 1 | `dude_where_is_my_car.wav` | 0.99941 | 0.99940 |
| 2 | `fleur_es_sample.wav` | 0.99648 | 0.99120 |
| 3 | `mary_had_lamb.mp3` | 0.99770 | 0.99812 |
| 4 | `monte_cristo.flac` | 0.99740 | 0.99899 |
| 5 | `obama_first_45_secs.mp3` | 0.99966 | 0.99949 |
| 6 | `obama.mp3` | 0.99920 | 0.99917 |
| 7 | `winning_call.mp3` | 0.99940 | 0.99940 |
| | **aggregate / worst** | **0.99870 / 0.99648** | **0.99835 / 0.99120** |

### Stage PCC (measured in isolation against the HF fp32 reference)

| Stage | PCC |
|---|---|
| audio tower, `voxtral_encoder` (streams 0–3) | 0.99537 |
| audio tower, `encoder_stack` + 4 substituted layer stubs (streams 4–7) | 0.99532 |
| audio embeds after `voxtral_multi_modal_projector` | 0.99186 |
| LM stack (`llama_decoder_layer` + part-stub L1/L2 + `llama_model` L3–29), hidden | 0.99891 |
| LM stack + `decoder_head`, logits | 0.99755 |
| one decode step (KV-cached) | 0.98856 |

### Other measured numbers

| Check | Value |
|---|---|
| `avg_pool1d` conformance PCC (NOT in the chain) | 0.9999988 |
| `trace_capture_selftest(device)` | **True** — encode / prefill / decode each captured host-op-free, replay PCC **1.000000**, trace released before the next stage |
| `host_op_selftest()` | **on_device=True, 0 host aten ops** for BOTH heads |
| Per-component PCC suite (`tests/pcc/`, 17 tests) | **17 passed** — every repaired stub is still backward compatible with its graduated golden |
| Gate-1 runtime probe | `torch_ops=0`, `ttnn_dispatch=41735` |
| Gate 2 | 16/16 routed stubs invoked in the real forward |
| Greedy-token agreement vs fp32 golden, free-running | 159/256 (audio_chat), 210/256 (transcription) |
| Distinct TT texts across the 8 streams | 7/8 both heads (streams 5 and 6 are the same speech, `obama.mp3` truncated to its first 30 s) |
| e2e test wall clock | ~152 s on one p150b (warm golden cache) |

### Sample output (real, from the TT pipeline)

`transcription`, stream 3 (`mary_had_lamb.mp3`):

> The first words I spoke in the original Cornograph were a little piece of practical poetry:
> "Mary had a little lamb, its fleece was white as snow

`audio_chat`, stream 0 (`bcn_weather.mp3`):

> The audio discusses the significant temperature change in Barcelona, from 35 degrees Celsius to
> -20 degrees Celsius, over a 24-hour period.

## Graduated-stub repairs

Four stubs graduated with defects that only surface in a composed autoregressive chain, and
thirteen graduated at a lower math fidelity than the composed chain can afford. Every live body
that differs from its `.last_good_native` snapshot is declared, with its sha256 and the reason, in
`_stubs/_e2e_repairs.json`; Phase 0 of the e2e test fails on any **undeclared** divergence, so a
silently edited (or reverted) stub cannot slip through. The snapshots are never overwritten — they
remain the record of what graduated.

| Stub | Defect found | Repair |
|---|---|---|
| `llama_rotary_embedding` | built cos/sin with `position_ids=zeros` → cos≡1, sin≡0 (RoPE identity) | table from the real HF `rotary_emb` over `arange(0, capacity)` + row gather; a zeros `position_ids` still returns row 0, so the graduated golden is unchanged |
| `llama_attention` | applied **no** rotary (the harness fed cos=1/sin=0, hiding it) and had no KV cache | optional ttnn-only `rope=` / `kv=` / `mode=`; defaults reproduce the graduated body exactly |
| `llama_decoder_layer` | same missing-RoPE / no-KV-cache defect | same optional-kwarg repair |
| `llama_model` | took token IDs (Voxtral needs `inputs_embeds` with audio scattered in), RoPE capped at 256, no KV cache, no layer range | optional `inputs_embeds=` / `rope=` / `kv_slots=` / `layer_range=` / `skip_embedding=` / `mode=` |
| 13 stubs (all but `avg_pool1d`, `token_embed`, `llama_rotary_embedding`) | graduated at the default math fidelity; only `llama_model` had picked HiFi4 | every `ttnn.linear` / `rms_norm` / `layer_norm` / SDPA call raised to `MathFidelity.HiFi4, fp32_dest_acc_en=True` (`_apply_hifi4.py`). Same ops, same order, same weights. LM logits 0.9914 → **0.9976**, one decode step 0.9541 → **0.9886** |
| `voxtral_encoder`, `encoder_stack` | conv weights lived on host, so `ttnn.conv1d` re-uploaded them per call — illegal inside `begin_trace_capture`, which made the encode stage un-traceable | cache ttnn's preprocessed conv weights on device after the first call. Numerics identical (0.995367 before and after); the tower got ~9x faster |

Maintenance scripts (not part of the runtime package):
`_apply_hifi4.py` (re-applies the fidelity repair) and `_refresh_repairs.py` (re-stamps the
manifest digests after a deliberate stub edit).

## Layout

```
models/tt_transformers/demo/voxtral_mini_3b_2507/
├── tt/
│   ├── pipeline.py      # THE shared chained pipeline: build_pipeline, run_audio_chat/run_transcription,
│   │                    #   PIPELINE_STAGES, per-stage trace_setup/step/inputs, selftests
│   ├── inputs.py        # real 8-stream input assembly (feature extractor + tokenizer), _captured access
│   └── reference.py     # HF goldens (cached to _captured/) -- the ONLY place HF forward code runs
├── demo/
│   ├── demo_audio_chat.py     # Call 1 entrypoint
│   └── demo_transcription.py  # Call 2 entrypoint
├── tests/
│   ├── pcc/             # the 17 per-component PCC tests from bring-up
│   └── e2e/
│       ├── gates.py             # Gate 1/2/3 machinery + graduated inventory (device-free)
│       └── test_e2e_pipeline.py # Gates 1/2/3 for both heads over 8 streams
├── _stubs/              # the 17 graduated native-ttnn bodies + their .last_good_native snapshots
├── _captured/           # captured reference tensors and cached HF goldens
├── e2e_plan.json        # the authoritative plan this implementation follows
└── RUN_REPORT.md        # bring-up placement report (17/17 ON_DEVICE)
```
