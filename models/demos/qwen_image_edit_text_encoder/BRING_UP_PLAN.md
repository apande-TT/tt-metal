# Bring-up plan: `/tmp/tt_hw_planner_components/qwen_image_edit_text_encoder`

Backend template: **qwen25_vl (auto-upstream)** at `models/demos/qwen25_vl` (canonical HF id: `None`).
New `model_type` = `qwen2_5_vl`; sibling `model_type` = `None`.

**Summary:** 9 REUSE · 6 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `None` and re-run for a sharper diff.
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): qwen25_vl (auto-upstream) (score 100: exact model_type 'qwen2_5_vl'); tt_transformers / simple_text_demo (multimodal) (score 40: category 'VLM' default (generic runner)); Mistral-Small-3.1 (mistral3 VLM) (score 30: category 'VLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `qwen25_vl (auto-upstream)` (selected) | 100 | exact model_type 'qwen2_5_vl' |
| 2 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 3 | `Mistral-Small-3.1 (mistral3 VLM)` | 30 | category 'VLM' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **ADAPT** | `token_embed` | `models/tt_transformers/tt/embedding.py` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **ADAPT** | `layer` | `models/tt_transformers/tt/multimodal/llama_layernorm.py` | `—` |
| **ADAPT** | `encoder_stack` | `models/tt_transformers/tt/multimodal/llama_vision_encoder.py` | `—` |
| **ADAPT** | `decoder_head` | `models/tt_transformers/tt/lm_head.py` | `—` |
| **NEW** | `v_l_text_model` | `—` | `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` |
| **NEW** | `v_l_decoder_layer` | `—` | `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` |
| **NEW** | `vision_transformer_pretrained_model` | `—` | `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` |
| **NEW** | `v_l_vision_block` | `—` | `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` |
| **REUSE** | `v_l_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `v_l_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `v_l_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `v_l_vision_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `v_l_patch_merger` | `—` | `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` |
| **REUSE** | `v_l_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |
| **NEW** | `vision_patch_embed` | `—` | `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` |
| **REUSE** | `vision_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |

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
| hidden_act | silu | — |
| hidden_size | 3584 | — |
| intermediate_size | 18944 | — |
| max_position_embeddings | 128000 | — |
| num_attention_heads | 28 | — |
| num_hidden_layers | 28 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 152064 | — |

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 3584 | — |
| intermediate_size | 18944 | — |
| max_position_embeddings | 128000 | — |
| num_attention_heads | 28 | — |
| num_hidden_layers | 28 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 152064 | — |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 3584 | — |
| intermediate_size | 18944 | — |
| max_position_embeddings | 128000 | — |
| num_attention_heads | 28 | — |
| num_hidden_layers | 28 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 152064 | — |

### `layer` — ADAPT
_reuse_registry: llama_layernorm -> models/tt_transformers/tt/multimodal/llama_layernorm.py::TtLayerNorm (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 3584 | — |
| intermediate_size | 18944 | — |
| max_position_embeddings | 128000 | — |
| num_attention_heads | 28 | — |
| num_hidden_layers | 28 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 152064 | — |

### `encoder_stack` — ADAPT
_reuse_registry: llama_vision_encoder -> models/tt_transformers/tt/multimodal/llama_vision_encoder.py::TtLlamaVisionEncoder (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 3584 | — |
| intermediate_size | 18944 | — |
| max_position_embeddings | 128000 | — |
| num_attention_heads | 28 | — |
| num_hidden_layers | 28 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 152064 | — |

### `decoder_head` — ADAPT
_reuse_registry: lm_head -> models/tt_transformers/tt/lm_head.py::LMHead (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 3584 | — |
| intermediate_size | 18944 | — |
| max_position_embeddings | 128000 | — |
| num_attention_heads | 28 | — |
| num_hidden_layers | 28 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 152064 | — |

### `v_l_text_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=283 sample_paths=['language_model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_decoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=28 leaves=280 sample_paths=['language_model.layers.0', 'language_model.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `vision_transformer_pretrained_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=262 sample_paths=['visual'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_vision_block` — NEW
_[supplemental module-tree pass] module-tree: occ=32 leaves=256 sample_paths=['visual.blocks.0', 'visual.blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=32 leaves=128 sample_paths=['visual.blocks.0.mlp', 'visual.blocks.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=122 leaves=122 sample_paths=['visual.blocks.0.norm1', 'visual.blocks.0.norm2'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=28 leaves=112 sample_paths=['language_model.layers.0.mlp', 'language_model.layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=28 leaves=112 sample_paths=['language_model.layers.0.self_attn', 'language_model.layers.1.self_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_vision_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=32 leaves=64 sample_paths=['visual.blocks.0.attn', 'visual.blocks.1.attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_patch_merger` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=4 sample_paths=['visual.merger'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `v_l_rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['language_model.rotary_emb'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `vision_patch_embed` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['visual.patch_embed'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `vision_rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['visual.rotary_pos_emb'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
