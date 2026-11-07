# Bring-up plan: `zai-org/GLM-5.3-Flash`

Backend template: **GLM-5 Next (glm5_next hybrid linear-attn/MoE VLM)** at `models/demos/glm5_next` (canonical HF id: `zai-org/GLM-5.3-Flash`).
New `model_type` = `glm5_next`; sibling `model_type` = `glm5_next`.

**Summary:** 9 REUSE · 13 NEW component(s).

> **Notes:**
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): GLM-5 Next (glm5_next hybrid linear-attn/MoE VLM) (score 100: exact model_type 'glm5_next'); tt_transformers / simple_text_demo (multimodal) (score 40: category 'VLM' default (generic runner)); Mistral-Small-3.1 (mistral3 VLM) (score 30: category 'VLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `GLM-5 Next (glm5_next hybrid linear-attn/MoE VLM)` (selected) | 100 | exact model_type 'glm5_next' |
| 2 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 3 | `Mistral-Small-3.1 (mistral3 VLM)` | 30 | category 'VLM' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `text_model` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `text_decoder_layer` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **REUSE** | `text_linear_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **ADAPT** | `text_mo_e` | `models/tt_transformers/tt/mixtral_moe.py` | `—` |
| **NEW** | `vision_model` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `vision_block` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **REUSE** | `text_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `text_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `text_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `vision_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `vision_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `text_hyper_connection` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **REUSE** | `text_unweighted_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **NEW** | `text_forget_gate` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `text_indexer` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `text_experts` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `text_topk_router` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `text_r_m_s_norm_gated` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `vision_patch_merger` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `text_hyper_head` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
| **NEW** | `vision_patch_embed` | `—` | `transformers/src/transformers/models/glm5_next/modeling_glm5_next.py` |
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

### `text_model` — NEW
_module-tree: occ=1 leaves=942 sample_paths=['language_model']_

| field | new model | sibling |
|---|---|---|

### `text_decoder_layer` — NEW
_module-tree: occ=45 leaves=939 sample_paths=['language_model.layers.0', 'language_model.layers.1']_

| field | new model | sibling |
|---|---|---|

### `text_linear_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=34 leaves=374 sample_paths=['language_model.layers.0.self_attn', 'language_model.layers.1.self_attn']_

| field | new model | sibling |
|---|---|---|

### `text_mo_e` — ADAPT
_reuse_registry: moe_routing_mixtral_style -> models/tt_transformers/tt/mixtral_moe.py::TtMoeLayer (ADAPT). derived from compatibility.py BUILDING_BLOCKS 'MoE routing (Mixtral-style)'. Generic MoE block currently hard-codes num_devices=8 and top-2. Adapting to other top-k or device counts needs a small refactor. Larger-scale MoE (DeepSeek/GPT-OSS) has standalone demos. | module-tree: occ=42 leaves=252 sample_paths=['language_model.layers.3.mlp', 'language_model.layers.4.mlp']_

| field | new model | sibling |
|---|---|---|

### `vision_model` — NEW
_module-tree: occ=1 leaves=251 sample_paths=['visual']_

| field | new model | sibling |
|---|---|---|

### `vision_block` — NEW
_module-tree: occ=24 leaves=240 sample_paths=['visual.blocks.0', 'visual.blocks.1']_

| field | new model | sibling |
|---|---|---|

### `text_m_l_p` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=45 leaves=180 sample_paths=['language_model.layers.0.mlp', 'language_model.layers.1.mlp']_

| field | new model | sibling |
|---|---|---|

### `text_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=11 leaves=121 sample_paths=['language_model.layers.3.self_attn', 'language_model.layers.7.self_attn']_

| field | new model | sibling |
|---|---|---|

### `text_r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=113 leaves=113 sample_paths=['language_model.layers.0.input_layernorm', 'language_model.layers.0.post_attention_layernorm']_

| field | new model | sibling |
|---|---|---|

### `r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=97 leaves=97 sample_paths=['visual.blocks.0.norm1', 'visual.blocks.0.norm2']_

| field | new model | sibling |
|---|---|---|

### `vision_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=24 leaves=96 sample_paths=['visual.blocks.0.attn', 'visual.blocks.1.attn']_

| field | new model | sibling |
|---|---|---|

### `vision_m_l_p` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=24 leaves=96 sample_paths=['visual.blocks.0.mlp', 'visual.blocks.1.mlp']_

| field | new model | sibling |
|---|---|---|

### `text_hyper_connection` — NEW
_module-tree: occ=90 leaves=90 sample_paths=['language_model.layers.0.attn_hc', 'language_model.layers.0.ffn_hc']_

| field | new model | sibling |
|---|---|---|

### `text_unweighted_r_m_s_norm` — REUSE
_reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=90 leaves=90 sample_paths=['language_model.layers.0.attn_hc.input_norm', 'language_model.layers.0.ffn_hc.input_norm']_

| field | new model | sibling |
|---|---|---|

### `text_forget_gate` — NEW
_module-tree: occ=34 leaves=68 sample_paths=['language_model.layers.0.self_attn.forget_gate', 'language_model.layers.1.self_attn.forget_gate']_

| field | new model | sibling |
|---|---|---|

### `text_indexer` — NEW
_module-tree: occ=11 leaves=44 sample_paths=['language_model.layers.3.self_attn.indexer', 'language_model.layers.7.self_attn.indexer']_

| field | new model | sibling |
|---|---|---|

### `text_experts` — NEW
_module-tree: occ=42 leaves=42 sample_paths=['language_model.layers.3.mlp.experts', 'language_model.layers.4.mlp.experts']_

| field | new model | sibling |
|---|---|---|

### `text_topk_router` — NEW
_module-tree: occ=42 leaves=42 sample_paths=['language_model.layers.3.mlp.gate', 'language_model.layers.4.mlp.gate']_

| field | new model | sibling |
|---|---|---|

### `text_r_m_s_norm_gated` — NEW
_module-tree: occ=34 leaves=34 sample_paths=['language_model.layers.0.self_attn.o_norm', 'language_model.layers.1.self_attn.o_norm']_

| field | new model | sibling |
|---|---|---|

### `vision_patch_merger` — NEW
_module-tree: occ=1 leaves=7 sample_paths=['visual.merger']_

| field | new model | sibling |
|---|---|---|

### `text_hyper_head` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['language_model.hc_head']_

| field | new model | sibling |
|---|---|---|

### `vision_patch_embed` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['visual.patch_embed']_

| field | new model | sibling |
|---|---|---|

### `vision_rotary_embedding` — REUSE
_reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['visual.rotary_pos_emb']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
