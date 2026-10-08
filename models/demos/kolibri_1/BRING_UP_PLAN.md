# Bring-up plan: `Aleph-Alpha/Kolibri-1`

Backend template: **tt_transformers / simple_text_demo** at `models/tt_transformers/demo/simple_text_demo.py` (canonical HF id: `None`).
New `model_type` = `kolibri1`; sibling `model_type` = `None`.

**Summary:** 4 REUSE · 5 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `None` and re-run for a sharper diff.
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): tt_transformers / simple_text_demo (score 40: category 'LLM' default (generic runner)); NemotronH (nemotron_h hybrid Mamba2/MoE) (score 30: category 'LLM' default); falcon7b_common (auto-upstream) (score 30: category 'LLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo` (selected) | 40 | category 'LLM' default (generic runner) |
| 2 | `NemotronH (nemotron_h hybrid Mamba2/MoE)` | 30 | category 'LLM' default |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **ADAPT** | `token_embed` | `models/tt_transformers/tt/embedding.py` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **ADAPT** | `layer` | `models/tt_transformers/tt/multimodal/llama_layernorm.py` | `—` |
| **ADAPT** | `encoder_stack` | `models/tt_transformers/tt/multimodal/llama_vision_encoder.py` | `—` |
| **ADAPT** | `decoder_head` | `models/tt_transformers/tt/lm_head.py` | `—` |
| **NEW** | `model` | `—` | `transformers/src/transformers/models/kolibri1/modeling_kolibri1.py` |
| **NEW** | `decoder_layer` | `—` | `transformers/src/transformers/models/kolibri1/modeling_kolibri1.py` |
| **NEW** | `f_p8_linear` | `—` | `transformers/src/transformers/models/kolibri1/modeling_kolibri1.py` |
| **NEW** | `sparse_moe_block` | `—` | `transformers/src/transformers/models/kolibri1/modeling_kolibri1.py` |
| **REUSE** | `m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `router` | `—` | `transformers/src/transformers/models/kolibri1/modeling_kolibri1.py` |

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
| hidden_size | 2560 | — |
| max_position_embeddings | 262144 | — |
| num_attention_heads | 48 | — |
| num_hidden_layers | 50 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 128000 | — |

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 2560 | — |
| max_position_embeddings | 262144 | — |
| num_attention_heads | 48 | — |
| num_hidden_layers | 50 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 128000 | — |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 2560 | — |
| max_position_embeddings | 262144 | — |
| num_attention_heads | 48 | — |
| num_hidden_layers | 50 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 128000 | — |

### `layer` — ADAPT
_reuse_registry: llama_layernorm -> models/tt_transformers/tt/multimodal/llama_layernorm.py::TtLayerNorm (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 2560 | — |
| max_position_embeddings | 262144 | — |
| num_attention_heads | 48 | — |
| num_hidden_layers | 50 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 128000 | — |

### `encoder_stack` — ADAPT
_reuse_registry: llama_vision_encoder -> models/tt_transformers/tt/multimodal/llama_vision_encoder.py::TtLlamaVisionEncoder (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 2560 | — |
| max_position_embeddings | 262144 | — |
| num_attention_heads | 48 | — |
| num_hidden_layers | 50 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 128000 | — |

### `decoder_head` — ADAPT
_reuse_registry: lm_head -> models/tt_transformers/tt/lm_head.py::LMHead (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 2560 | — |
| max_position_embeddings | 262144 | — |
| num_attention_heads | 48 | — |
| num_hidden_layers | 50 | — |
| num_key_value_heads | 4 | — |
| vocab_size | 128000 | — |

### `model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=58302 sample_paths=['model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `decoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=50 leaves=58300 sample_paths=['model.layers.0', 'model.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `f_p8_linear` — NEW
_[supplemental module-tree pass] module-tree: occ=57950 leaves=57950 sample_paths=['model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `sparse_moe_block` — NEW
_[supplemental module-tree pass] module-tree: occ=50 leaves=57800 sample_paths=['model.layers.0.mlp', 'model.layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=19250 leaves=57750 sample_paths=['model.layers.0.mlp.experts.0', 'model.layers.0.mlp.experts.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=301 leaves=301 sample_paths=['model.layers.0.self_attn.q_norm', 'model.layers.0.self_attn.k_norm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `router` — NEW
_[supplemental module-tree pass] module-tree: occ=50 leaves=50 sample_paths=['model.layers.0.mlp.gate', 'model.layers.1.mlp.gate'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
