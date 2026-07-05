# Bring-up plan: `mistralai/Voxtral-Mini-3B-2507`

Backend template: **hf_eager universal (catch-all)** at `models/demos/hf_eager/demo.py` (canonical HF id: `None`).
New `model_type` = `voxtral`; sibling `model_type` = `None`.

**Summary:** 7 REUSE · 7 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `None` and re-run for a sharper diff.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `llama_for_causal_l_m` | `—` | `transformers/src/transformers/models/voxtral/modeling_voxtral.py` |
| **NEW** | `llama_model` | `—` | `transformers/src/transformers/models/voxtral/modeling_voxtral.py` |
| **NEW** | `llama_decoder_layer` | `—` | `transformers/src/transformers/models/voxtral/modeling_voxtral.py` |
| **NEW** | `voxtral_encoder` | `—` | `transformers/src/transformers/models/voxtral/modeling_voxtral.py` |
| **NEW** | `voxtral_encoder_layer` | `—` | `transformers/src/transformers/models/voxtral/modeling_voxtral.py` |
| **REUSE** | `voxtral_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `llama_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `llama_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `llama_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `voxtral_multi_modal_projector` | `—` | `transformers/src/transformers/models/voxtral/modeling_voxtral.py` |
| **NEW** | `avg_pool1d` | `—` | `transformers/src/transformers/models/voxtral/modeling_voxtral.py` |
| **REUSE** | `llama_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |

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
| hidden_size | 3072 | — |
| vocab_size | 131072 | — |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_size | 3072 | — |
| vocab_size | 131072 | — |

### `llama_for_causal_l_m` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=304 sample_paths=['language_model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `llama_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=303 sample_paths=['language_model.model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `llama_decoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=30 leaves=300 sample_paths=['language_model.model.layers.0', 'language_model.model.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `voxtral_encoder` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=293 sample_paths=['audio_tower'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `voxtral_encoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=32 leaves=288 sample_paths=['audio_tower.layers.0', 'audio_tower.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `voxtral_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=32 leaves=128 sample_paths=['audio_tower.layers.0.self_attn', 'audio_tower.layers.1.self_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `llama_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=30 leaves=120 sample_paths=['language_model.model.layers.0.self_attn', 'language_model.model.layers.1.self_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `llama_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=30 leaves=120 sample_paths=['language_model.model.layers.0.mlp', 'language_model.model.layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `llama_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=61 leaves=61 sample_paths=['language_model.model.layers.0.input_layernorm', 'language_model.model.layers.0.post_attention_layernorm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `voxtral_multi_modal_projector` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=3 sample_paths=['multi_modal_projector'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `avg_pool1d` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['audio_tower.avg_pooler'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `llama_rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['language_model.model.rotary_emb'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
