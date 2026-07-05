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

## Results (on device, N=6)

| Gate | Result |
|---|---|
| Gate 1 — routed stubs still native ttnn | ✅ all 7 `_stubs/*.py` byte-identical to `.last_good_native`, 0 CPU fallbacks |
| Gate 2 — all graduated modules invoked | ✅ `avg_pool1d, llama_decoder_layer, llama_for_causal_l_m, llama_model, voxtral_encoder, voxtral_encoder_layer, voxtral_multi_modal_projector` |
| Gate 3 — e2e PCC vs HF golden ≥ 0.95 | ✅ **prefill logits PCC = 0.9984**; TT greedy sequence = HF `generate()` **6/6 tokens** |

Verified-equivalence PCC (aux stubs vs torch): `voxtral_encoder_layer=0.99973`,
`avg_pool1d=1.0`, `llama_decoder_layer=0.99753`, `llama_model=0.99637`.
Per-step logits PCC: `[0.99841, 0.99651, 0.99879, 0.99890, 0.99939, 0.99939]`.

## Run

Environment (this `pr-46283` worktree shares the `.so`/env of `/home/ttuser/tt-metal`):

```bash
cd /home/ttuser/tt-metal
export TT_METAL_HOME=/home/ttuser/tt-metal
export PYTHONPATH=/home/ttuser/tt-metal-pr46283:/home/ttuser/tt-metal
PY=/home/ttuser/tt-metal/python_env/bin/python

# e2e test (Gate 1/2/3)
$PY -m pytest models/demos/hf_eager/voxtral_mini_3b_2507/tests/e2e/test_e2e_voxtral.py -s

# demo (audio -> text)
$PY -m models.demos.hf_eager.voxtral_mini_3b_2507.demo.demo_transcribe --max-new-tokens 16
```

Notes: device is opened with `l1_small_size=24576` (audio conv scratch). The
input is a deterministic synthetic 16 kHz waveform built through the real
`WhisperFeatureExtractor` + tokenizer; parity is measured against
`model.generate()` on the **identical** `(input_ids, input_features)`, so the
audio content itself is irrelevant to correctness. The TT decode loop has no KV
cache (recomputes the full sequence per step), so `N` (`--max-new-tokens`) is
kept small; both sides are capped to the same `N`.
