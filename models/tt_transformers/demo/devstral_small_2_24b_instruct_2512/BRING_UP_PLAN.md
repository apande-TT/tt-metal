# Bring-up plan: `mistralai/Devstral-Small-2-24B-Instruct-2512`

Backend template: **Mistral-Small-3.1 (mistral3 VLM)** at `models/tt_transformers/demo/simple_text_demo.py` (canonical HF id: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`).
New `model_type` = `mistral3`; sibling `model_type` = `mistral3`.

**Summary:** 11 REUSE · 7 NEW component(s).

> **Notes:**
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): Mistral-Small-3.1 (mistral3 VLM) (score 100: exact model_type 'mistral3'); tt_transformers / simple_text_demo (multimodal) (score 40: category 'VLM' default (generic runner)); qwen25_vl (auto-upstream) (score 30: category 'VLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `Mistral-Small-3.1 (mistral3 VLM)` (selected) | 100 | exact model_type 'mistral3' |
| 2 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 3 | `qwen25_vl (auto-upstream)` | 30 | category 'VLM' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **ADAPT** | `token_embed` | `models/tt_transformers/tt/embedding.py` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **ADAPT** | `layer` | `models/tt_transformers/tt/multimodal/llama_layernorm.py` | `—` |
| **ADAPT** | `encoder_stack` | `models/tt_transformers/tt/multimodal/llama_vision_encoder.py` | `—` |
| **ADAPT** | `decoder_head` | `models/tt_transformers/tt/lm_head.py` | `—` |
| **NEW** | `ministral3_model` | `—` | `transformers/src/transformers/models/mistral3/modeling_mistral3.py` |
| **NEW** | `ministral3_decoder_layer` | `—` | `transformers/src/transformers/models/mistral3/modeling_mistral3.py` |
| **NEW** | `pixtral_vision_model` | `—` | `transformers/src/transformers/models/mistral3/modeling_mistral3.py` |
| **NEW** | `pixtral_attention_layer` | `—` | `transformers/src/transformers/models/mistral3/modeling_mistral3.py` |
| **NEW** | `pixtral_transformer` | `—` | `transformers/src/transformers/models/mistral3/modeling_mistral3.py` |
| **REUSE** | `ministral3_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `ministral3_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `pixtral_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `pixtral_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `ministral3_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `pixtral_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `mistral3_multi_modal_projector` | `—` | `transformers/src/transformers/models/mistral3/modeling_mistral3.py` |
| **REUSE** | `ministral3_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |
| **NEW** | `mistral3_patch_merger` | `—` | `transformers/src/transformers/models/mistral3/modeling_mistral3.py` |
| **REUSE** | `mistral3_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `pixtral_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |

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

### `token_embed` — ADAPT
_reuse_registry: embedding -> models/tt_transformers/tt/embedding.py::ScaledEmbedding (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | silu |
| hidden_size | 5120 | 5120 |
| intermediate_size | 32768 | 32768 |
| max_position_embeddings | 393216 | 131072 |
| num_attention_heads | 32 | 32 |
| num_hidden_layers | 40 | 40 |
| num_key_value_heads | 8 | 8 |
| vocab_size | 131072 | 131072 |

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | silu |
| hidden_size | 5120 | 5120 |
| intermediate_size | 32768 | 32768 |
| max_position_embeddings | 393216 | 131072 |
| num_attention_heads | 32 | 32 |
| num_hidden_layers | 40 | 40 |
| num_key_value_heads | 8 | 8 |
| vocab_size | 131072 | 131072 |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | silu |
| hidden_size | 5120 | 5120 |
| intermediate_size | 32768 | 32768 |
| max_position_embeddings | 393216 | 131072 |
| num_attention_heads | 32 | 32 |
| num_hidden_layers | 40 | 40 |
| num_key_value_heads | 8 | 8 |
| vocab_size | 131072 | 131072 |

### `layer` — ADAPT
_reuse_registry: llama_layernorm -> models/tt_transformers/tt/multimodal/llama_layernorm.py::TtLayerNorm (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | silu |
| hidden_size | 5120 | 5120 |
| intermediate_size | 32768 | 32768 |
| max_position_embeddings | 393216 | 131072 |
| num_attention_heads | 32 | 32 |
| num_hidden_layers | 40 | 40 |
| num_key_value_heads | 8 | 8 |
| vocab_size | 131072 | 131072 |

### `encoder_stack` — ADAPT
_reuse_registry: llama_vision_encoder -> models/tt_transformers/tt/multimodal/llama_vision_encoder.py::TtLlamaVisionEncoder (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | silu |
| hidden_size | 5120 | 5120 |
| intermediate_size | 32768 | 32768 |
| max_position_embeddings | 393216 | 131072 |
| num_attention_heads | 32 | 32 |
| num_hidden_layers | 40 | 40 |
| num_key_value_heads | 8 | 8 |
| vocab_size | 131072 | 131072 |

### `decoder_head` — ADAPT
_reuse_registry: lm_head -> models/tt_transformers/tt/lm_head.py::LMHead (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | silu |
| hidden_size | 5120 | 5120 |
| intermediate_size | 32768 | 32768 |
| max_position_embeddings | 393216 | 131072 |
| num_attention_heads | 32 | 32 |
| num_hidden_layers | 40 | 40 |
| num_key_value_heads | 8 | 8 |
| vocab_size | 131072 | 131072 |

### `ministral3_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=403 sample_paths=['language_model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ministral3_decoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=40 leaves=400 sample_paths=['language_model.layers.0', 'language_model.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pixtral_vision_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=243 sample_paths=['vision_tower'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pixtral_attention_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=24 leaves=240 sample_paths=['vision_tower.transformer.layers.0', 'vision_tower.transformer.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pixtral_transformer` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=240 sample_paths=['vision_tower.transformer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ministral3_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=40 leaves=160 sample_paths=['language_model.layers.0.self_attn', 'language_model.layers.1.self_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ministral3_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=40 leaves=160 sample_paths=['language_model.layers.0.mlp', 'language_model.layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pixtral_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=24 leaves=96 sample_paths=['vision_tower.transformer.layers.0.attention', 'vision_tower.transformer.layers.1.attention'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pixtral_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=24 leaves=96 sample_paths=['vision_tower.transformer.layers.0.feed_forward', 'vision_tower.transformer.layers.1.feed_forward'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ministral3_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=81 leaves=81 sample_paths=['language_model.layers.0.input_layernorm', 'language_model.layers.0.post_attention_layernorm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pixtral_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=49 leaves=49 sample_paths=['vision_tower.ln_pre', 'vision_tower.transformer.layers.0.attention_norm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral3_multi_modal_projector` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=5 sample_paths=['multi_modal_projector'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ministral3_rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['language_model.rotary_emb'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral3_patch_merger` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['multi_modal_projector.patch_merger'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral3_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=1 leaves=1 sample_paths=['multi_modal_projector.norm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `pixtral_rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['vision_tower.patch_positional_embedding'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
