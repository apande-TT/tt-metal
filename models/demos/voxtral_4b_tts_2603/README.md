# Voxtral-4B-TTS-2603 — end-to-end TTNN pipeline

A real, chained TTNN pipeline for `mistralai/Voxtral-4B-TTS-2603`, composed from the graduated
stubs the bring-up tool produced under
`models/tt_transformers/demo/voxtral_4b_tts_2603/` (Source B) and gated against the HuggingFace
reference (Source A).

**Batch = 32 independent samples per call.** Everything below — every demo, every gate test file,
the trace contract — drives 32 distinct prompts through one program per step. No test states a
batch its body does not run; each reads the number off the pipeline object.

**Sections.** `consolidated.safetensors` holds three repeated block stacks — `layers` (26, the text
backbone), `acoustic_transformer.layers` (3) and `audio_tokenizer.decoder_blocks` (8). The first two
are ported and gated; the third is not, and *Holes* says why. Each ported stack is held as a plain
list of same-typed blocks so the profiler's structural walk can find, size and cap it — a stack the
walk cannot see gets its depth inferred for the whole run.

---

## What this model actually is

The hub repo ships `params.json`, `consolidated.safetensors` and `tekken.json` — and **no
`config.json`**, so `AutoConfig` / `AutoModel` raise "Unrecognized model". `params.json` declares
`model_type=voxtral_tts`, for which transformers 5.12.1 has no class.

The golden is therefore the verified reconstruction Source B ships
(`tests/pcc/_reference_loader.py`): a `MistralForCausalLM` built from the native checkpoint —
26 layers, hidden 3072, head_dim 128, GQA 32/8, MLP 9216, plain RoPE θ=1e6, tied embeddings,
vocab 131072. The native→HF remap (wq/wk RoPE permute, w1/w2/w3 → gate/down/up) reproduces
Mistral's own published HF conversion of the declared base bit-identically (max|diff| == 0.0).

Input is built with the **real tokenizer**: `tekken.json` rebuilt on `tiktoken`
(`mistral_common` is not installed in `python_env`, and there is no `tokenizer_config.json` for
`AutoTokenizer`). 32 distinct English texts, each truncated to exactly 32 real tokens — no padding,
so every row is genuine content and the causal mask is the plain lower-triangular one.

> **Do not gate this model on coherent generated text.** The TTS finetune's tied text head is
> effectively untrained: next-token entropy 11.417 nats against 11.784 for a uniform draw over
> 131072 tokens, top-1 probability ~6e-4. Hidden-state norms are healthy through all 26 layers.
> The gate is PCC against HF, never readability.

---

## The three Calls

The bring-up tool graduated the *same* text stack at three granularities — leaf ops, the block, and
the whole stack. Rather than stack redundant copies in one path, the pipeline routes them across
task heads that share **no** graduated module, so all 10 are load-bearing and none is wasted. Call 3
then re-uses the leaf stubs over a *different* section of the checkpoint.

### Call 1 — `text_generation`
Generative head; reference is `model.generate()`. The composed leaf-stub stack:

```
ids [32, 32]                      (tekken tokenizer, 32 distinct prompts)
  -> token_embed                                              (graduated)
  -> rotary_embedding(position_ids) -> (cos, sin)             (graduated)
  -> block  0        decoder_layer stub                       (graduated)
  -> block  1        layer stub                               (graduated)
  -> block  2        r_m_s_norm -> attention -> +res -> r_m_s_norm -> mlp   -> +res
  -> block  3        r_m_s_norm -> attention -> +res -> r_m_s_norm -> m_l_p -> +res
  -> blocks 4..25    decoder_layer stub                       (graduated)
  -> r_m_s_norm on model.norm                                 (graduated)
  -> decoder_head(x[:, -1:, :]) -> logits [32, 1, 131072]     (graduated)
  -> ttnn.argmax                                              (sampling stays on device)
```

Blocks 2 and 3 sit in the middle of the residual stream: their output feeds blocks 3..25 and the
logits. Removing any leaf changes the final answer — this is not a coverage sweep.

### Call 2 — `hidden_states`
Non-generative feature extraction; reference is `MistralModel.forward().last_hidden_state`, which
is literally the tensor Source B captured for this component (`_captured/model/output.pt`).
Routes the graduated whole-stack port: `ids [32, 32]` → `model` stub → `[32, 32, 3072]`.

### Call 3 — `acoustic`
The checkpoint's **second** repeated stack, `acoustic_transformer.layers`, ported onto the same
graduated leaf stubs (`tt/acoustic.py`). Its three blocks are plain Mistral decoder layers at a
different RoPE θ — same native key names (`attention.wq`, `feed_forward.w1`, `attention_norm`, …),
so the same key map and the same RoPE permute apply — and `params.json →
multimodal.audio_model_args.acoustic_transformer_args` specifies every dimension:

```
lm_hidden [32, 32, 3072]        (the Call-1 text stack's hidden state, handed over ON DEVICE)
  -> llm_projection                                           (graduated decoder_head)
  -> blocks 0..2   r_m_s_norm -> attention -> +res -> r_m_s_norm -> mlp -> +res
  -> r_m_s_norm on acoustic_transformer.norm                  (graduated)
  -> semantic_codebook_output -> [32, 32, 8320]               (graduated decoder_head)
```

Reference is `tt/acoustic.py::AcousticReference`: real `transformers` `MistralDecoderLayer` /
`MistralRMSNorm` modules loaded with the checkpoint's own tensors — not a hand-rolled imitation.
The section's other three tensors (`input_projection`, `time_projection`,
`acoustic_codebook_output`) are the flow-matching sampler's surface and are **not driven** — see
*Holes*.

---

## Results — all measured on device (QB2, 1× Blackhole)

| Call | Gate 3 metric | **FINAL_PCC** | Target |
|---|---|---|---|
| 1 `text_generation` | min over 32 samples × 16 decode steps of PCC(TT next-token logits, HF next-token logits), teacher-forced on HF's own greedy prefix | **0.9988631485049634** | ≥ 0.99 |
| 2 `hidden_states` | min over 32 samples of PCC(TT `last_hidden_state`, HF `last_hidden_state`) | **0.9999818067958669** | ≥ 0.99 |
| 3 `acoustic` | min over 32 samples of PCC(TT, torch reference) for BOTH the acoustic hidden state and the semantic-codebook logits | **0.9999956204347794** | ≥ 0.99 |

Supporting numbers:

| | Call 1 | Call 2 |
|---|---|---|
| batch driven | 32 | 32 |
| prompt_len / horizon | 32 / 16 | 32 / n-a |
| prefill hidden PCC per sample | 0.999980 – 0.999997 | 0.999982 – 0.999997 |
| prefill logits PCC per sample | 0.999967 – 0.999999 | — |
| per-step logits PCC | 0.998863 – 0.999999 | — |
| token agreement (teacher-forced) | 487/512 = 0.951 | — |
| of the 25 disagreements, HF near-ties (p-ratio ≥ 0.5) | 25/25, worst ratio 0.885 | — |
| vs Source B's own captured golden | — | 0.9998978992397038 |
| batch-drop guard (pairwise distinct) | 32/32 hidden, 32/32 logits | 32/32 TT, 32/32 HF |

**Gate 1** — every routed block is an instance of the class from its own `_stubs/<name>.py` module
(so `decoder_layer` cannot pass as `layer`); every intermediate is a `ttnn.Tensor`; the stubs'
`native_probe` still reports `torch_ops = 0`.

**Gate 2** — invocation counts for ONE full forward at 26 layers, recomputed from the wiring
(`tt/generation.py::expected_invocation_counts`, the single source of truth the test reads):

```
token_embed 1   rotary_embedding 1   decoder_layer 23   layer 1   attention 2
r_m_s_norm  5   mlp 1   m_l_p 1   decoder_head 1        |  model 1  (Call 2)
```

All **10/10** graduated modules invoked. Load-bearing spot-check: halving block 2's feed-forward
moves the final logits (max|Δ| 2.58, PCC vs baseline 0.9177), so "invoked" means "in the data path".

> `decoder_layer` is **23**, not the 24 that `e2e_plan.json` v1 wrote: index 0 plus indices 4..25
> is 1 + 22. The plan's arithmetic slip was caught by recomputing the count from the wiring, and
> the plan is corrected.

### Behavioral proof (Call 1)
Free-running self-cascade (TT feeds its own on-device argmax back) matches HF's greedy ids on
444/512 positions. Exact self-cascade match is **not** a gate on this checkpoint — a near-uniform
distribution makes ties bf16-vs-fp32 brittle. Under teacher forcing every one of the 25
disagreements is a near-tie in HF's *own* distribution (probability ratio ≥ 0.5). Scoring by token
*rank* would be wrong here: a flat distribution puts true probability ties at rank ~7.

---

## Command 3 — trace contract (host-free, per stage)

`PIPELINE_STAGES = ["prefill", "decode"]`, derived from Source A's config: the reference is
`MistralForCausalLM` and `is_encoder_decoder = False`. No `encode` (decoder-only). No `vocode` —
see *Holes*.

Each stage exposes, on the pipeline object:
`<stage>_trace_setup(inputs)`, `<stage>_trace_step()`, `<stage>_trace_inputs()` (zero-arg),
`<stage>_trace_items()` (zero-arg), plus the AR pair `decode_prefill()` / `decode_step()`.

The sequence axis is the variable dim (bound = `max_position_embeddings` = 128000) and is pinned to
a fixed capacity **C = 64** (`VOXTRAL_TRACE_C`), which covers the 32-token prompt plus the 16-step
horizon. The padded ids, the RoPE cos/sin and the causal mask are taken **from the HF reference
itself** (`hf.model.rotary_emb`, `transformers.masking_utils.create_causal_mask`) and uploaded into
persistent device buffers outside the trace, so the step reads only those buffers.

- **prefill right-pads**: causality leaves rows `[0:real_len]` bit-identical and the pad keys are
  masked out, so the answer is `hidden[:, :real_len]`. The next-token row index is fixed by the
  setup, so its slice is static and traceable.
- **decode left-pads**: the newest token is pinned at row `C-1`, so the traced step stays
  index-invariant as the context grows.

`<stage>_trace_items()`: prefill returns **B×C = 2048** (the repeated blocks process every pinned
position of every sample — stating 1 would price the arithmetic ceiling 2048× too small and report
a compute-bound stage as memory-bound); decode returns **B = 32** (one token per sample).

Measured:

```
trace prefill.hidden : C=64  PCC=1.0
trace prefill.logits : C=64  PCC=1.0
trace decode.token   : C=64  PCC=1.0
trace_capture_selftest = True   (batch=32, C=64, trace_region_size=200 MB)

host_op_selftest[text_generation]: on_device=True  n_host_ops=0
host_op_selftest[hidden_states]  : on_device=True  n_host_ops=0
```

No capacity fallback was needed at C=64; if a capture overflows the region, `trace_capture_selftest`
halves C and **prints** the fallback rather than dropping it.

### `build_pipeline` — the single build surface

```python
build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs)
```

Constructs and **returns** the resident pipeline object; it never runs the model. Demo entrypoints,
the e2e tests and `trace_capture_selftest` all build through it. Demo kwargs (`text`, `prompt`,
`language`, …) are accepted and ignored — shapes come from the config, not a prompt.

`layers` caps the depth of **every** repeated stack (both Call 1's composed stack and Call 2's
`model`-stub stack); `None` means every layer and `0` is rejected. Embeddings, norms, rotary tables
and the LM head stay intact, so a capped build still exercises every distinct op — just fewer times.
Proven non-inert on device:

```
layers=None -> depth 26,  decoder_layer invoked 23x
layers=4    -> depth  4,  decoder_layer invoked  1x
layers=2    -> clamped up to 4 (printed), all four block kinds still present
layers=0    -> ValueError
```

Call 1's stack is clamped **up** to 4 because below that a graduated block kind would be
*structurally absent* rather than merely built fewer times. A capped build must stay a model, not a
fragment.

**`prefill_layers` / `decode_layers`** are the per-stack overrides named after the `PIPELINE_STAGES`
entries. This model has **one** repeated text stack shared by both stages, so when the two overrides
disagree the build takes the max and prints that it collapsed them — the single-number trap is made
visible instead of silent. Each override falls back to `layers`.

Stacks are discoverable: each is a plain Python list of **same-typed** elements. Call 1's four block
kinds all share the one concrete `VoxtralBlock` wrapper so the list is homogeneous. The HF reference
stays reachable at `pipeline.reference_model` — ground truth for how many sections the model has and
how deep each is.

---

## Layout

```
models/demos/voxtral_4b_tts_2603/
  tt/common.py        shared setup: reference loader, tekken tokenizer, 32-prompt batch builder,
                      decode-horizon resolver, invocation counter, PCC
  tt/generation.py    Call 1: the composed stack + run_text_generation + the HF golden helper
  tt/hidden_states.py Call 2: the model-stub wrapper + run_hidden_states + the HF golden helper
  tt/acoustic.py      Call 3: the acoustic_transformer section + its torch reference + the
                      declared list of what is NOT driven
  tt/pipeline.py      THE shared pipeline: PIPELINE_STAGES, build_pipeline, the trace hooks,
                      trace_capture_selftest, host_op_selftest
  device_session.py   the device opener for the STANDALONE selftest entrypoints; outside tt/ so
                      the pipeline package keeps its no-self-open guarantee
  demo/demo_text_generation.py
  demo/demo_hidden_states.py
  demo/demo_acoustic.py
  tests/e2e/test_e2e_text_generation.py
  tests/e2e/test_e2e_hidden_states.py
  tests/e2e/test_e2e_acoustic.py
  tests/e2e/test_trace_and_host_ops.py
  tests/e2e/test_text_generation_perf.py   trace-replay perf (no PCC); the node that proves the
                      pipeline captures and replays at trace+1cq
  e2e_plan.json       the planning document these were built from
```

**The demo and the test share one pipeline.** The chained forward lives once in `tt/`; both the
demo entrypoint and the e2e test import and call the same function, so a passing test guarantees a
working demo.

## Running

```bash
cd /home/ttuser/tt-metal
export PYTHONPATH=/home/ttuser/tt-metal

# demos
./python_env/bin/python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_generation
./python_env/bin/python -m models.demos.voxtral_4b_tts_2603.demo.demo_hidden_states
./python_env/bin/python -m models.demos.voxtral_4b_tts_2603.demo.demo_acoustic
#   --batch --seq-len --horizon --layers --device-id   (only demo_text_generation has --horizon)

# gates
./python_env/bin/python -m pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_text_generation.py -s
./python_env/bin/python -m pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_hidden_states.py  -s
./python_env/bin/python -m pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_acoustic.py       -s
./python_env/bin/python -m pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_trace_and_host_ops.py -s

# trace-replay perf (separate from the gates; prints TRACE_PER_TOKEN_MS / TRACE_REPLAY_PATH)
./python_env/bin/python -m pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_text_generation_perf.py -s
```

Env knobs: `VOXTRAL_E2E_HORIZON` (decode horizon), `VOXTRAL_TRACE_C` (pinned capacity),
`TT_PERF_LAYERS` (depth cap, read by `build_pipeline` when no explicit `layers` is passed).

---

## Decode horizon

Resolved by `common.resolve_decode_horizon()` and applied identically to the TT loop and to
`model.generate(max_new_tokens=H)`, with an early break once every row emits `eos_token_id`:

1. **Stop token** — `eos_token_id = 2` from `generation_config`. This checkpoint's untrained text
   head never emits it in practice, so the safety cap governs.
2. **Config length** — `generation_config.max_new_tokens` is unset, and `max_length` is the
   transformers library default 20, which is ≤ the real prompt length. Not usable.
3. **Fallback** — neither signal exists, so **H = 16**, chosen for lack of any model signal and
   marked as such in the code. Overridable with `VOXTRAL_E2E_HORIZON`.

Trace capture is unaffected: it always runs at the fixed capacity C, so variable-length decode never
makes the traced shapes dynamic.

---

## Hardware budget

Registered figures for this box, used as-is:

| | |
|---|---|
| Box | QB2, 4× Blackhole |
| Chips opened by this run | **1** |
| DRAM per chip | 32 GB |
| Aggregate DRAM for this run | 32 GB (1 × 32 GB) — not the box total of 128 GB |
| Usable per chip | **29.2 GB** (the TP=1 figure; 28.6 GB is the TP>1/CCL-axis figure and does not apply — this pipeline is TP=1, no mesh, no collectives) |

Device footprint is a **budget estimate from parameter counts**, not a measurement from this run:
~7.7 GB for Call 1's stack (26 × 233 MB of bf16 weights + embedding + LM head) and ~6.9 GB for
Call 2's, so ~14.6 GB with both heads resident, plus a 200 MB trace region. Host RAM: the fp32
reference model is ~13.7 GB of the box's 249 GB.

**No per-stage batch ceiling is recorded.** Batch 32 ran at full 26-layer depth on both heads, with
both heads resident simultaneously, at C=64 — there was no point at which a stage could not hold 32,
so there is no ceiling to write down and none to re-test.

---

## Holes — recorded, not faked

1. **`encoder_stack` is not graduated and is not routed.** `harness_skipped.json` lists it;
   `skip_diagnosis.json` gives verdict `manual`; it has no `.last_good_native` snapshot. It is a
   generic role the plan auto-derived from the sibling VLM (Mistral-Small-3.1's vision encoder).
   Voxtral-4B-TTS is decoder-only, so none of the candidate submodule paths resolve. Routing it
   would be fabricating work.
2. **No waveform, and no `vocode` stage.** Two things are missing between Call 3's semantic codes
   and audio, and they are missing for different reasons:
   - **`audio_tokenizer.decoder_blocks` (8 blocks) is not ported.** It is a weight-normed causal-conv
     + sliding-window-attention vocoder with layer scale and QK norm (`conv_weight_norm`,
     `attn_sliding_window_size 16`, `layer_scale_init 0.01`, `decoder_convs_strides_str "1,2,2,2"`).
     `transformers` 5.12.1 ships `voxtral` and `voxtral_realtime` and neither is this architecture,
     so there is no reference to gate a port against. A port with nothing to compare it to is not a
     port, so it is not attempted, and its stack is the one declared section the structural walk
     cannot see.
   - **The flow-matching sampler is not driven.** `acoustic_transformer` also carries
     `input_projection` (36→3072, the noisy acoustic latents), `time_projection` (the diffusion
     timestep) and `acoustic_codebook_output` (3072→36). How those three conditioning terms combine
     is not stated by the checkpoint and has no reference implementation here, so Call 3 drives only
     the part whose role the names and shapes make unambiguous and declares the rest
     (`tt/acoustic.py::dump_section_report`, asserted by
     `tests/e2e/test_e2e_acoustic.py::test_the_unported_surface_is_declared_not_forgotten`).
3. **No incremental KV cache.** The graduated blocks are full-attention with no cache surface, so
   the decode stage recomputes a resident left-padded context at fixed C each step. That is
   host-free and traceable — `decode_prefill()` seeds the resident context and `decode_step()`
   re-reads it — but it is not an incremental cache, and there is no cross-attention KV to seed
   (the model is decoder-only).

## Source-B stub edits

All edits were **additive** so the B=1 path stays byte-identical, and each was proved by re-running
the component PCC test the bring-up tool generated:

| stub | edit | component PCC after | `native_probe` |
|---|---|---|---|
| `token_embed` | batch bound from `ids.shape[0]` / `input_ids.shape[0]` | 1.0 | 2 / 0 |
| `rotary_embedding` | leading bound from `position_ids.shape[0]`, `cos.shape[0]` | 0.9999623597927504 | 10 / 0 |
| `attention` | `_split_heads` / tail reshape bound from `proj.shape[0]`, `context.shape[0]` | 0.9999699583972707 | 31 / 0 |
| `decoder_layer` | same, in its private `TtAttention` | 0.9999960116508202 | 43 / 0 |
| `layer` | same | 0.9999960116508202 | 43 / 0 |
| `model` | `_leading_batch()` helper; `_split_heads`, tail reshape and the post-embedding reshape | 0.9998978986756867 (**bit-identical** to the pre-edit file re-run) | 1176 / 0 |

`ttnn_dispatch` counts are unchanged from what the bring-up recorded pre-edit, so the edits added
zero device ops. A hardcoded leading `1` here does not fail — it *silently drops samples 2..32* —
which is why every bound is now read off the tensor.

## Notes for the next person

- `ttnn.softmax` without an explicit `compute_kernel_config` runs a low-fidelity kernel. Over this
  26-layer stack that single op is the difference between 0.9858 (fail) and 0.9996. Every stub
  passes HiFi4 + `fp32_dest_acc`; do not remove it.
- Activations run **float32 against bfloat16 weights**. Weight dtype is the wrong lever — matmul
  truncates weights to bf16 regardless; op fidelity and activation dtype are the right ones.
- `ttnn.argmax` on a TILE tensor collapses onto one core; convert to `ROW_MAJOR` first.
- `ttnn.add` broadcasts a `(1,1,S,S)` mask against `[32,32,S,S]` scores, so no per-sample mask
  expansion is needed.
- `ttnn.Shape` does not support slice `__getitem__` — use `list(t.shape)[:-2]`.
- Measuring sample distinctness on leading values alone reports a false "1/32 distinct": every
  prompt starts with the same BOS token, so position 0 is identical by construction. Measure over
  the whole sample.
- `output_hidden_states[N_layers]` is **post**-final-norm, not layer N-1's output. Comparing the
  last layer against it shows a fake collapse.
