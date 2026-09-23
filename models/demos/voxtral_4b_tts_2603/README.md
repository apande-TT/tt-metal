# Voxtral-4B-TTS-2603 — end-to-end TTNN pipeline

A real, chained TTNN pipeline for `mistralai/Voxtral-4B-TTS-2603`, composed from the 31 graduated
stubs the bring-up tool produced under `models/tt_transformers/demo/voxtral_4b_tts_2603/`
(**Source B**) and gated against the HuggingFace reference (**Source A**).

**Batch = 32 independent samples per call.** Every demo, every gate test and the trace contract
drive 32 distinct prompts through one program per step. No test states a batch its body does not
run; each reads the number off the pipeline object.

**Two Calls, not three.** This package supersedes `plan_version 2`
(`text_generation` / `hidden_states` / `acoustic`), which was written against a 10-component
bring-up whose stub names no longer exist. See *The two Calls*.

---

## What this model actually is

The hub repo ships `params.json`, `consolidated.safetensors` and `tekken.json` — and **no
`config.json`**, so `AutoConfig` / `AutoModel` raise "Unrecognized model type". `params.json`
declares `model_type=voxtral_tts`, for which transformers ships no class.

The reference is therefore Source B's own
`tests/pcc/_reference_loader.py::load_reference_model()`, which consumes all 386 checkpoint
tensors (4002.35 M params incl. buffers) and returns a `VoxtralTTSReferenceModel`:

| part | what it is | depth |
|---|---|---|
| `.model` / `.lm_head` | a `MistralForCausalLM` text backbone, dim 3072, `rope_theta` 1e6, tied embeddings | 26 layers |
| `.acoustic_transformer` | flow-matching audio transformer + semantic/acoustic codebook heads | 3 layers |
| `.audio_tokenizer` | the codec that turns audio codes into a 24 kHz waveform | 8 decoder blocks |

`model.generate()` is **not** the TTS task: it drives the tied TEXT head, which cannot emit audio
codes. The golden therefore runs the reference's own submodules through the chain the checkpoint's
architecture dictates (`tt/golden.py`), and that helper is proven to BE the reference's arithmetic
by `test_golden_is_the_references_own_arithmetic`.

---

## The two Calls

Both are one authored chain in `tt/pipeline.py`. The demos and the e2e tests call the *same*
function, so a green test cannot coexist with a broken demo.

### Call 1 — `text_to_speech` (the model's advertised task)

`VoxtralTTSPipeline.run_text_to_speech`: tokenized text → a 24 kHz waveform.

```
prefill (26 layers, seeds the KV cache)
  └─ per frame:  acoustic sampler (7 Euler steps, CFG)  →  37 codes
                 └─ audio-token embedding  →  one decode step against the resident KV cache
  └─ vocode: the whole code sequence  →  waveform
```

Routes `token_embed`, `mistral_rotary_embedding`, `mistral_r_m_s_norm`, the four interchangeable
block kinds (`layer`, `mistral_decoder_layer`, `attention`+`mlp`, `mistral_attention`+`mistral_m_l_p`),
the acoustic section and the codec.

### Call 2 — `text_continuation`

`VoxtralTTSPipeline.run_text_continuation`: greedy causal-LM continuation, the only home of
`encoder_stack`, `mistral_model` (byte-identical aliases, split by batch row so both do a disjoint
share of the real work) and `decoder_head` (the LM head).

> **Coherence caveat.** This is a TTS checkpoint. The backbone emits AUDIO codebook tokens and its
> tied TEXT head is effectively untrained, so the continuation is near-uniform garbage even when the
> load is bit-correct. That does not weaken the gate — the gate compares TT against HF on the SAME
> input, and garbage-that-matches is a valid parity result.

---

## Results — measured on device (1× Blackhole p300c)

### What passes

| check | result |
|---|---|
| Gate 1 — every routed stub still native ttnn | **pass** (`test_gates.py`) |
| Gate 1 companion — Source B's 31 component PCC tests, golden caches cleared | **31/31 pass** |
| Gate 2 — all 31 graduated modules invoked inside the real forward path | **pass**, both Calls |
| Per-stage trace capture + replay, all 4 stages | **PCC 1.000000** each |
| Fully-on-device (host-aten-op observer), both Calls | **0 host ops** |
| Call 1 prefill hidden state vs HF, min over 32 samples | **0.999955** |
| Call 1 acoustic velocity field vs HF | **PCC 1.000000** (err RMS 5.9e-4 against \|v\| RMS 1.086) |
| Call 1 frame-0 semantic code, all 32 samples | **exact** |
| Call 2 step-0 logits vs HF, min over 32 samples | **0.999922** |

### Gate 3 — the final-output PCC, and why Call 1 does not reach 0.99

| Call | Gate 3 metric | measured | target |
|---|---|---|---|
| 1 `text_to_speech` | min over 32 of PCC(TT waveform, HF waveform) | **0.1249** | ≥ 0.99 |
| 2 `text_continuation` | min over 32 of per-step logit PCC + exact token equality | **0.5037** (token agreement 0.851) | ≥ 0.99 |

**This is a discretization result, not a wiring one, and it is at the hardware floor.** The chain
passes through two quantizations per frame, and both feed back into the next frame:

* The acoustic sampler rounds `x ∈ [-1, 1]` onto **21 levels**, so code edges are **0.1 apart** —
  and **20.6%** of the reference's own `x_final` values land within **0.01** of an edge.
* Frame 0's semantic codes are exact and its acoustic codes agree **94.2%**. Re-running the same
  frame on the *reference's* hidden state — i.e. with a mathematically perfect input — still only
  reaches **98.5%**. That residual is the documented FPU floor: ~4.9e-4 per float32-activation ×
  bfloat16-weight matmul (unchanged by a float32 weight, so not a knob) and ~5.7e-4 for the
  bfloat16 Q/K/V cast SDPA forces.
* One flipped code changes that frame's audio *and* the embedding fed to the next decode step, so
  the sequences separate and every later frame is compared against a different context.

Reaching bit-identical codes across 32 × 13 × 37 = 15 392 values would need roughly **1000×** lower
error than that floor. The plan's prescribed mitigation is the fidelity ladder, and it was climbed
in full (see *Source-B stub edits*): prefill hidden went **0.9971 → 0.999955**, semantic code
agreement **0.44 → 0.70**, acoustic **0.31 → 0.66**. The remaining gap is not reachable by
numerics on this hardware.

**Not done, deliberately:** the gate was not weakened to make this green. No `xfail`, no `skip`, no
lowered threshold, and the codes are never teacher-forced from the reference — the plan forbids it
and it would make the comparison meaningless.

---

## Command 3 — the trace contract (host-free, per stage)

`PIPELINE_STAGES = ["prefill", "decode", "acoustic", "vocode"]`, each exposing the same
model-agnostic seam: `<stage>_trace_inputs()` (zero-arg) → `<stage>_trace_setup(inputs)` →
`<stage>_trace_step()` → `<stage>_trace_items()`.

```
[trace] prefill   items=2048     replay PCC=1.000000 ok
[trace] decode    items=32       replay PCC=1.000000 ok
[trace] acoustic  items=1344     replay PCC=1.000000 ok
[trace] vocode    items=8192     replay PCC=1.000000 ok
```

Stage traces do **not** co-reside: each is released before the next is captured.

Two module-level entry points in `tt/pipeline.py` — the bring-up observers import the module in a
fresh process and call them **by name with no arguments**, so a method on the class would not count:

* `trace_capture_selftest(device=None)` → capture, replay and PCC-check one step of each stage.
* `host_op_selftest()` → run every Call under a `TorchDispatchMode` and assert **zero** host aten
  ops fired inside the model math.

`tt/` never opens a device. The opener lives in `device_session.py` at the package root, which is
the same carve-out a `__main__` selftest gets; the test fixture is the sole opener otherwise.

---

## Layout

```
demo/
  demo.py                       dispatcher
  demo_text_to_speech.py        Call 1 — writes 32 real .wav files
  demo_text_continuation.py     Call 2 — prints 32 continuations
tt/
  pipeline.py                   THE chained pipeline + build_pipeline + PIPELINE_STAGES + selftests
  text_stack.py                 embed → rope → 26 blocks → norm, with the resident per-layer KV cache
  acoustic_stage.py             the flow-matching sampler + discretization
  vocode_stage.py               the codec / waveform decoder
  continuation.py               Call 2's whole-stack bodies + LM head
  common.py                     tokenizer, 32-prompt input builder, reference bridge, InvocationCounter
  golden.py                     HF reference chains — imported by tests and `--compare` ONLY
tests/e2e/
  test_e2e_text_to_speech.py    Call 1 end to end vs the golden
  test_e2e_text_continuation.py Call 2 end to end vs the golden
  test_gates.py                 Gate 1 + Gate 2 + anti-shortcut scan + layers-knob proof
  test_component_pcc.py         re-runs Source B's 31 component tests with the golden caches CLEARED
  test_trace_and_host_ops.py    the trace contract + the fully-on-device check
device_session.py               the ONLY device opener outside the test fixture
```

**One-pipeline rule.** The chained forward pass lives only in `tt/`. The demos and the tests import
and call the same functions; a second copy in `demo/` would drift.

---

## Running

```bash
# demos  (--compare also runs the HF golden and prints the parity numbers)
python models/demos/voxtral_4b_tts_2603/demo/demo_text_to_speech.py --out-dir /tmp/wav
python models/demos/voxtral_4b_tts_2603/demo/demo_text_continuation.py --horizon 4
#   shared flags: --text (repeatable) --batch --layers --device-id --compare

# gates — the whole suite is ~7 min (426 s measured), 23 passed / 3 failed;
# the 3 are the Gate 3 PCC assertions above, and nothing else.
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/ -svv

# one file at a time
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_gates.py -svv               # ~20 s
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_component_pcc.py -svv       # ~2.5 min
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_trace_and_host_ops.py -svv  # ~73 s
pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_text_to_speech.py -svv
```

> The suite used to overrun the gate's 2700 s budget and read as a device hang. Two causes, both
> fixed: un-traceable `ttnn.zeros` wedged the board mid-capture (see *Source-B stub edits*), and
> four tests each rebuilt an identical 4 B pipeline. `tests/e2e/test_trace_and_host_ops.py` now
> builds **once** in a module-scoped fixture and hands that pipeline to both selftests — building a
> second copy beside it does not fit in DRAM (4.19 GB of 4.25 GB).

---

## Decode horizon

**Stop-token rule (priority 1).** The acoustic transformer's semantic head predicts
`AudioSpecialTokens.end_audio` (id 1, read off the reference — never a literal). Decoding stops
once **every** batch row has emitted it. The stop test is accumulated ON DEVICE (`eq` →
`logical_or` → `sum`); exactly one reduced scalar crosses to the host per frame as a loop-control
decision, via `.item()` on a 0-d reduction — which is `aten::_local_scalar_dense`, the one readback
the host-op observer treats as benign.

**Safety cap**, derived rather than magic: `ceil(frame_rate 12.5 × VOXTRAL_GATE_SECONDS)`, default
1.0 s → **13 frames** for the gate; the demo cap is the pipeline's real ceiling, 256 frames
(= the codec's prebuilt ALiBi mask, 2048 / 8 upsampling = 20.5 s).

Call 2 has **no** usable model signal — this checkpoint carries neither
`generation_config.max_new_tokens` nor a usable `max_length`, and the tied text head never emits
eos — so its horizon is 4, chosen for lack of one and applied identically to the golden. The eos id
is still read from the config and still breaks the loop if it ever fires.

Both sides run the identical stop rule and cap, so the two sequences are always compared over the
same model-grounded length. Trace capture is unaffected: it runs at a FIXED capacity.

---

## Source-B stub edits

All edits below are numeric or trace-safety fixes. **Every one of Source B's 31 component PCC tests
still passes** with the golden caches cleared (`test_component_pcc.py`), and every edited closure
still contains zero torch compute calls.

| stub(s) | edit | why |
|---|---|---|
| `mistral_r_m_s_norm`, `layer`, `mistral_decoder_layer` | stock `ttnn.rms_norm` → the same math spelled out in four float32 ops | the stock op sits at **9.65e-4** relative error vs **6.6e-8** spelled out; a norm error is relative, so it rescales the whole branch after it, and the stack runs 52 of them |
| `layer`, `mistral_decoder_layer`, `attention`, `mistral_attention`, `mlp`, `mistral_m_l_p`, `time_embedding`, `flow_matching_audio_transformer` | every `ttnn.linear`/`ttnn.matmul` now passes `compute_kernel_config` (HiFi4 + `fp32_dest_acc_en` + `packer_l1_acc`) | these defaulted to `fp32_dest_acc_en=False`, which rounds the matmul accumulator to bfloat16 at every step even though the activations are float32. 14 call sites across 6 files; the sibling `mistral_model.py` already passed it, which is why two ports of the same layer disagreed |
| `mistral_rotary_embedding` | the CONTIGUOUS rope tables are float32 (the `ttnn.embedding` GATHER table must stay bfloat16) | `_from_torch` defaults to bfloat16, so the fast path silently returned bfloat16 rope |
| `attention`, `mistral_attention`, `layer`, `mistral_decoder_layer`, `mlp`, `mistral_m_l_p` | the KV cache's zero tail is a memoised persistent buffer | see below |
| `causal_conv1d`, `causal_conv_transpose1d`, `parametrized_conv1d`, `parametrized_conv_transpose1d`, `voxtral_t_t_s_audio_tokenizer` | the conv padding / delay zero rows are memoised persistent buffers | see below |

**Why the zero buffers matter.** `ttnn.zeros` builds its tensor on the host and enqueues a *write*
to land it on device, and a captured trace replays kernels — it cannot replay a write. Every one of
these aborted capture with `TT_FATAL: Writes are not supported during trace capture`, and the failed
capture then **wedged the board** (`Read 0xffffffff over PCIe`), which is what the e2e gate had been
reporting as "tests/e2e exceeded 2700s with no verdict (likely device/fabric hang)". The hang was a
symptom of un-traceable code, not a flaky board.

---

## Holes — recorded, not faked

* **Voice conditioning.** The model card advertises 20 preset voices, but the voice prompts are
  AUDIO CODES this repo does not ship, and the checkpoint ships no codec *encoder* to derive them
  from a waveform. The prompt is built as `[bos] + tekken_encode(text) + [BEGIN_AUDIO]` with no
  voice prefix; `--voice` is accepted for interface parity and prints that it is unconditioned.
  **Effect on the gate: none** — TT and HF are fed the identical constructed prompt.
* **Codec encoder / waveform → codes.** The OSS checkpoint ships no `input_proj.*` or
  `encoder_blocks.*` tensors; `encode_waveforms` raises in the reference itself. There is no
  audio-input task head.
* **The serving prompt template and interleaved text/audio segmentation.** `params.json` carries
  `codebook_pattern: parallel` and `interleave_*_tokens_per_segment: 8192`, but the loop that
  applies them lives in `vllm-omni`, which is not one of the two permitted sources. With both limits
  at 8192 a short utterance is a single audio segment, which is what this pipeline implements.
* **`--batch` below 32 does not run the TTS decode step.** `ttnn.experimental.nlp_concat_heads_decode`
  pads its batch axis up to 32, so the `reshape(merged, [1, 1, batch, n_heads*head_dim])` that
  follows it in `layer.py::_decode_attn` fails `new_volume == old_volume` for any smaller batch
  (`--batch 4` raises `TT_FATAL: Invalid arguments to reshape`). Everything gated here runs at the
  model's batch of 32, where it is exercised end to end; a smaller batch needs the concat's padded
  width read off the returned tensor rather than assumed. Pre-existing, unrelated to the Gate 3
  numbers above.
* **`flow_matching_audio_transformer` reads as ungraduated** to the bring-up status reader even
  though its component PCC test passes — its live body no longer matches the shape that reader
  expects. The trace gate consequently permits eager execution; trace is nonetheless proven to work
  for all four stages above.

---

## Notes for the next person

* **A real prompt is not a tile multiple.** `[bos] + text + [BEGIN_AUDIO]` is 33 tokens. The KV
  cache appends on the sequence axis and needs a tile-aligned prefill, so the zero tail is added to
  the *ids*, in ROW_MAJOR, before the embedding — never as a concat across a tile boundary at row 33.
  The tail is invisible to the answer on both sides: prefill attention is `is_causal=True`, and
  decode reads `[0, position]` off `cur_pos` starting at the real length. Before this, Call 1 raised
  outright and Call 2 silently emitted token 0 for every row.
* **Compare two ports of the same module** when one is inaccurate. `mistral_model.py` (whole-stack)
  and `layer.py` (per-layer) implement identical arithmetic; the 5× accuracy gap between them was
  entirely `compute_kernel_config` and `rms_norm`, and diffing them found it in minutes.
* **Audit, don't eyeball.** A file can define a correct `_COMPUTE` constant and still have bare
  matmuls. An AST sweep for `ttnn.linear`/`ttnn.matmul` calls lacking `compute_kernel_config`, and a
  second for `ttnn.zeros`/`from_torch` inside a `build()` closure, found every offender.
* **The 300 s guard in `pytest.ini`** is sized for unit tests and is a flake here; every file in
  `tests/e2e/` sets `pytestmark = pytest.mark.timeout(3600)`. A bound is still enforced, so a
  genuine hang still fails.
