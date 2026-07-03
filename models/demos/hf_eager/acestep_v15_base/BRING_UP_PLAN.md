# Bring-up plan: `ACE-Step/acestep-v15-base`

Backend template: **hf_eager universal (TTS)** at `models/demos/hf_eager/demo.py` (canonical HF id: `None`).
New `model_type` = `acestep`; sibling `model_type` = `None`.

**Summary:** 6 REUSE · 13 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `None` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `ace_step_di_t_model` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `ace_step_di_t_layer` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **REUSE** | `ace_step_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `qwen3_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `ace_step_encoder_layer` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **REUSE** | `qwen3_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `ace_step_condition_encoder` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `ace_step_lyric_encoder` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `ace_step_timbre_encoder` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `ace_step_audio_tokenizer` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `audio_token_detokenizer` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `attention_pooler` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `timestep_embedding` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **REUSE** | `qwen3_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |
| **NEW** | `lambda` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `residual_f_s_q` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |
| **NEW** | `f_s_q` | `—` | `transformers/src/transformers/models/acestep/modeling_acestep.py` |

## Shared modules (always reusable, no copy needed)

| Purpose | tt-metal path |
|---|---|
| LayerNorm / RMSNorm | `models/common/rmsnorm.py` |
| LightweightModule base | `models/common/lightweightmodule.py` |
| Tensor helpers | `models/common/tensor_utils.py` |
| Generic utility funcs | `models/common/utility_functions.py` |

## Action by status

- **REUSE**: import / call the sibling's tt-module unchanged. Weight names match. The global PCC gate enforces this — if it fails, `force_adapt_all` demotes the REUSE component to NEW and the brain iterates per-component.
- **NEW**: write/adapt the TTNN port. A stub file is generated under `_stubs/` (torch fallback by default), then progressively rewritten to native ttnn through per-component PCC iteration. If a sibling tt-file with the same role exists, the agent reuses its layout and updates shape constants (hidden_size, num_heads, intermediate_size, eps); otherwise it writes from scratch against the HF reference.

## Per-component shape diff

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 2048 | — |
| intermediate_size | 6144 | — |
| max_position_embeddings | 32768 | — |
| num_attention_heads | 16 | — |
| num_hidden_layers | 24 | — |
| num_key_value_heads | 8 | — |
| patch_size | 2 | — |
| vocab_size | 64003 | — |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 2048 | — |
| intermediate_size | 6144 | — |
| max_position_embeddings | 32768 | — |
| num_attention_heads | 16 | — |
| num_hidden_layers | 24 | — |
| num_key_value_heads | 8 | — |
| patch_size | 2 | — |
| vocab_size | 64003 | — |

### `ace_step_di_t_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=475 sample_paths=['decoder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ace_step_di_t_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=24 leaves=456 sample_paths=['decoder.layers.0', 'decoder.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ace_step_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=64 leaves=384 sample_paths=['decoder.layers.0.self_attn', 'decoder.layers.0.cross_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen3_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=237 leaves=237 sample_paths=['decoder.layers.0.self_attn_norm', 'decoder.layers.0.self_attn.q_norm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ace_step_encoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=16 leaves=192 sample_paths=['encoder.lyric_encoder.layers.0', 'encoder.lyric_encoder.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen3_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=40 leaves=160 sample_paths=['decoder.layers.0.mlp', 'decoder.layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ace_step_condition_encoder` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=151 sample_paths=['encoder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ace_step_lyric_encoder` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=99 sample_paths=['encoder.lyric_encoder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ace_step_timbre_encoder` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=51 sample_paths=['encoder.timbre_encoder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ace_step_audio_tokenizer` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=32 sample_paths=['tokenizer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `audio_token_detokenizer` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=28 sample_paths=['detokenizer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `attention_pooler` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=27 sample_paths=['tokenizer.attention_pooler'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `timestep_embedding` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=10 sample_paths=['decoder.time_embed', 'decoder.time_embed_r'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen3_rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=5 leaves=5 sample_paths=['decoder.rotary_emb', 'encoder.lyric_encoder.rotary_emb'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `lambda` — NEW
_[supplemental module-tree pass] module-tree: occ=4 leaves=4 sample_paths=['decoder.proj_in.0', 'decoder.proj_in.2'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `residual_f_s_q` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=4 sample_paths=['tokenizer.quantizer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `f_s_q` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=2 sample_paths=['tokenizer.quantizer.layers.0'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
