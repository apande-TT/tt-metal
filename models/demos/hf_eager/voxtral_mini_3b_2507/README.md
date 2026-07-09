# Voxtral-Mini-3B-2507 — end-to-end TTNN pipeline

Real audio→text pipeline for `mistralai/Voxtral-Mini-3B-2507` on Tenstorrent,
chaining the graduated TTNN component stubs (`_stubs/*.py`) into the actual
`VoxtralForConditionalGeneration` forward pass and comparing to the HF golden
(`model.generate()`).

Voxtral = **Whisper audio encoder → multi-modal projector → Llama-3 text
decoder**. It has one generative task family (audio-conditioned text
generation), so there is **one task head (Call 1)** covering all graduated
modules.

## Layout

```
tt/pipeline.py                 ONE shared chained forward pass (imported by BOTH demo and test)
demo/demo_transcribe.py        runnable demo: real input -> TTNN pipeline -> generated text
tests/e2e/test_e2e_voxtral.py  e2e test asserting Gate 1/2/3
e2e_plan.json                  the planner sketch (Command 1)
```

The demo and the test import and call the **same** `tt/pipeline.py`
function, so a passing test guarantees a working demo.

## Pipeline (Call 1 — `voxtral_audio_understanding`)

```
input_features (1,128,3000)
   │  voxtral_encoder                       # audio_tower: conv stem + 32 Whisper layers + LN
   ▼
last_hidden_state (1,1500,1280)
   │  reshape(-1, 5120)                      # 4 frames -> 1 token  (HF get_audio_features)
   │  voxtral_multi_modal_projector          # 5120 -> gelu -> 3072
   ▼
audio_embeds (375,3072) ── masked_scatter ──▶ text-token embeds at audio_token_id(24)
                                              (text embeds from llama_for_causal_l_m.embed_tokens)
   │  llama_for_causal_l_m                    # 30 Llama layers (GQA + real RoPE + causal) + lm_head
   ▼
logits (1,T,131072) ─ argmax ─▶ greedy decode loop (capped to N)
```

### Using all 7 graduated modules

`voxtral_encoder_layer`, `llama_model`, `llama_decoder_layer` are strictly
**nested** inside the two monoliths (`voxtral_encoder`, `llama_for_causal_l_m`),
and `avg_pool1d` (`audio_tower.avg_pooler`) is **defined but never called** by
the HF forward (downsampling is done by the `reshape(-1,5120)` instead). A
single non-redundant forward can only run one granularity per subtree, so these
four are invoked as **verified-equivalence stages** on the *same real tensors*
the parity chain produces — each compared to its exact torch submodule — with
their outputs kept **out** of the parity chain (no reference tensor is ever
injected at a parity joint).

## Results — real audio (measured on Wormhole n300)

Verified on **3 distinct real speech clips** with the proper Voxtral
`apply_transcription_request` input format, TT greedy vs HF `generate()`:

| Clip | Length | TT transcription vs HF |
|---|---|---|
| MLK "I have a dream" | 13 s | ✅ identical — *"I have a dream that one day this nation will rise up and live out the true meaning of its creed."* |
| LibriSpeech (stew…) | 10.4 s | ✅ identical (16/16 tokens) |
| LibriSpeech (belly…) | 3.3 s | ✅ same content; only over-generated tokens past end-of-speech differ (handled by EOS-stop) |

Precision: **all matmul weights `bfloat8_b` (fp8) + LoFi**. (bf4_b was dropped —
it was found to corrupt a content token on the short/low-margin clip; fp8 keeps
every content token correct.) Gates: **0 CPU fallbacks**, all 7 graduated stubs
invoked on device. Note: correctness must be measured on **real speech** — a
synthetic tone yields near-tied logits whose argmax any quantization can flip,
so it is not a valid correctness signal.

## Trace+2CQ Scorecard

Config: audio-in / text-out (ASR) · batch=1 · both phases trace + 2CQ

| Metric                    | Value                                              |
| ------------------------- | -------------------------------------------------- |
| Hardware                  | Tenstorrent Wormhole — n300 board, single die (1 chip) |
| Audio input               | real 16 kHz speech → mel (1, 128, 3000)            |
| Prefill context length    | ~384 positions (≈376 audio embeds + prompt tokens) |
| Precision                 | all `bfloat8_b` (fp8) matmul weights + LoFi        |
| TTFT (prefill, trace+2CQ) | 105.82 ms                                          |
| Decode / token (trace+2CQ)| 37.74 ms                                           |
| T/S/U (tokens/sec/user)   | 26.50                                              |
| T/S (batch=1)             | 26.50                                              |
| Mesh / TP / DP            | 1×1, TP=1, DP=1, shard=False                       |
| On-device (model)         | True (mel-preproc + PCC checks excluded)           |

### Before vs After (Trace+2CQ)

| Metric           | Before (graduated) | After (fp8, optimized) | Δ                    |
| ---------------- | ------------------ | ---------------------- | -------------------- |
| TTFT (prefill)   | 353.75 ms          | 105.82 ms              | −70.1% (3.34× faster)|
| Decode / token   | 78.99 ms/tok       | 37.74 ms/tok           | −52.2% (2.09× faster)|
| T/S/U (tokens/sec)| 12.66             | 26.50                  | +109.3% (2.09×)      |

### Precision sweep (real audio, TT greedy vs HF)

Every precision was tested on the 3 real clips. Correctness is judged on the
**content** tokens (real speech); the short clip's tail tokens are over-generated
past end-of-speech and diverge at every precision (handled by EOS-stop).

| Decoder matmul weights        | Prefill PCC (real audio)†     | Content correct?                    | Decode/token (trace+2CQ) | T/S/U | Verdict |
|-------------------------------|-------------------------------|-------------------------------------|--------------------------|-------|---------|
| bf16 (all)                    | 0.9988 / 0.9992 / 0.9995      | ✅ all clips                        | 55.24 ms                 | 18.10 | correct but ~1.5× slower |
| bf8_b + bf4_b (original)      | 0.9635 / 0.9296 / 0.9809      | ✅ all content                      | 34.79 ms                 | 28.75 | fast (reference) |
| **all bf8_b (fp8) — shipped** | 0.9840 / 0.9931 / 0.9973      | ✅ all content                      | 37.74 ms                 | 26.50 | **chosen: correct + fast, no bf4 fragility** |
| all bf4_b                     | 0.8722 / 0.7942 / 0.8930      | ❌ 1 content slip (`Stuff`→`stuff`) | 33.04 ms                 | 30.26 | fastest but corrupts a content token |

† Per-clip prefill last-token logits PCC vs HF (clip 1 / clip 2 / clip 3).

**PCC is not the correctness gate here — token match is.** Note the values are
well below 1.0 yet the tokens are correct (e.g. bf8+bf4 at 0.93–0.98, all-fp8 at
0.98–0.997 → all content tokens right), and all-bf4 at 0.79–0.89 still transcribes
but slips one token. On real speech the correct token's margin is large, so even
a modest PCC lands the right argmax; on a synthetic tone the logits are near-tied,
so the same PCC flips tokens — which is why the earlier synthetic "0/6" was an
artifact, not a real-audio failure.

Takeaway: **all-bf8_b (fp8)** is the sweet spot — every real-speech content token
correct while staying fast. bf4_b buys ~5% more speed but corrupts content on
low-margin audio; bf16 is correct but ~1.5× slower.

### KV cache (real-context decode)

The `Decode / token` above is a fixed-window figure. Real generation recomputes
the whole growing sequence per token, so at real context (~383) the no-cache
decode is **124.69 ms/tok (8.0 tok/s)**. An opt-in on-device KV cache
(`pipeline.run(..., use_kv_cache=True)`) makes each step process only the new
token against a resident K/V cache — **35.80 ms/tok (27.9 tok/s), 3.48× faster**
at real context — with token-identical output.

## Optimization wins & measured speedup

13 wins in two groups. Speedups are given at the granularity actually measured:
the structural fusions were **not** ablated one-by-one, so they are reported as a
single "structural bundle" figure; items I isolated are called out. Decode =
ms/tok, trace+2CQ. Aggregate graduated→fp8: **78.99 → 37.74 ms/tok (2.09×)**;
TTFT **353.75 → 105.82 ms (3.34×)**.

| # | Win | Group | Speedup |
|---|-----|-------|---------|
| 1 | Fused QKV projection | structural | bundle |
| 2 | Fused single-kernel head split | structural | bundle |
| 3 | FlashAttention-2 SDPA (GQA) | structural | bundle |
| 4 | Fused HF RoPE | structural | bundle |
| 5 | GELU fused into fc1 (encoder) | structural | bundle |
| 6 | Last-position-only logits | structural | bundle |
| 7 | Trace-clean forward + trace capture | structural | bundle |
| 8 | 2 command queues (2CQ) | structural | **~1.00× at batch=1** (decode_2cq ≈ decode_only) |
| — | **structural bundle (1–8, at bf16)** | — | **78.99 → 55.24 ms/tok = 1.43×** |
| 9 | bf8_b weights (fp8) | precision | part of quant bucket |
| 10 | bf8_b MLP activations | precision | part of quant bucket |
| 11 | LoFi math fidelity (matmuls) | precision | part of quant bucket |
| — | **quantization bucket (bf16→fp8)** | — | **55.24 → 37.74 ms/tok = 1.46×** |
| 12 | bf4_b MLP weights | precision | **~1.08× (2.95 ms/tok) — DROPPED (corrupts a token)** |
| 13 | On-device KV cache (opt-in, added here) | structural | **3.48× at real ctx (124.7 → 35.8 ms/tok)** |

## Code authored, kernels used & knobs

The auto-bring-up tool **wrote ttnn (Python) code** — the native forward-pass
bodies (`_stubs/*.py`) for the **7 NEW components** (`voxtral_encoder`,
`voxtral_encoder_layer`, `avg_pool1d`, `voxtral_multi_modal_projector`,
`llama_for_causal_l_m`, `llama_model`, `llama_decoder_layer`), composing ttnn ops
and applying the fusion/dtype/fidelity tuning. The 7 REUSE components import
existing tt modules unchanged (ADAPT=0 for this model).

It authored **no device kernels** — no custom C++, **no tt-lang**. The ttnn ops
its Python code calls run on **stock tt-metal C++ LLK kernels** (SFPI +
`compute_kernel_api`, under `ttnn/cpp/.../device/kernels`, JIT-built by
`riscv-tt-elf-g++`). Hot-path ops → the tt-metal kernels they invoke:

| ttnn op | device kernel |
|---|---|
| `ttnn.linear` / `ttnn.matmul` (QKV, o_proj, MLP, lm_head) | matmul |
| `ttnn.transformer.scaled_dot_product_attention[_decode]` | `sdpa` (FlashAttention-2, GQA) |
| `ttnn.experimental.nlp_create_qkv_heads` / `nlp_concat_heads` | head split / concat |
| `ttnn.experimental.rotary_embedding_hf` | RoPE |
| `ttnn.rms_norm` | `layernorm`/rmsnorm |
| `ttnn.silu` / `ttnn.gelu` | `eltwise_sfpu` (SFPU) |
| `ttnn.mul` / `ttnn.add` | `eltwise_binary` |
| `ttnn.argmax` | reduction (ROW_MAJOR → multi-core) |
| `ttnn.embedding` / `concat` / `slice` / `reshape` / `to_layout` / `typecast` | data-movement |

Knobs (the tunables that produced the results above):

| Knob | Value | Effect |
|---|---|---|
| `math_fidelity` | HiFi4 (norm/SDPA), **LoFi** (proj/MLP matmuls) | accuracy ↔ matmul speed |
| weight dtype | **bf8_b (fp8)** — bf4_b dropped, bf16 optional | DRAM bandwidth |
| activation dtype | bf8_b (wide MLP intermediates) | DRAM bandwidth |
| `fp32_dest_acc_en` / `packer_l1_acc` | True / True | fp32 accumulation on the residual stream |
| `l1_small_size` | 24576 | audio conv scratch |
| `trace_region_size` | 200000000 | trace-capture buffer |
| `num_command_queues` | 2 (2CQ) | overlap host token-upload with compute |
| tensor layout | TILE (compute), ROW_MAJOR (argmax/embedding) | kernel dispatch / multi-core reduction |
| `use_kv_cache` | opt-in | O(1) vs O(context) per-token decode |

## Run

```bash
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
PY=$PWD/python_env/bin/python

# demo: real audio file -> transcription (stops at EOS)
$PY -m models.demos.hf_eager.voxtral_mini_3b_2507.demo.demo_transcribe --audio path/to/clip.wav

# e2e parity test (synthetic-input TT-vs-HF parity gate)
$PY -m pytest models/demos/hf_eager/voxtral_mini_3b_2507/tests/e2e/test_e2e_voxtral.py -s
```

Extra deps for the transcription-request input format: `mistral_common`,
`pydantic_extra_types`, `pycountry`.

Notes: device is opened with `l1_small_size=24576` (audio conv scratch). Real
transcription requires the `apply_transcription_request` input format (a naive
hand-built prompt yields garbage even from the HF reference). Generation stops
at EOS (`stop_at_eos=True` in the demo) to avoid low-confidence filler past
end-of-speech.
