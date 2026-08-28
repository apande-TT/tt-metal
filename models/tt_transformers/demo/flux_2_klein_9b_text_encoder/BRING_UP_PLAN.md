# Bring-up plan: `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`

Backend template: **tt_transformers / simple_text_demo** at `models/tt_transformers/demo/simple_text_demo.py` (canonical HF id: `None`).
New `model_type` = `qwen3`; sibling `model_type` = `None`.

**Summary:** 5 REUSE · 0 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `None` and re-run for a sharper diff.
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): qwen3 (auto-upstream) (score 100: exact model_type 'qwen3'); tt_transformers / simple_text_demo (score 40: category 'LLM' default (generic runner)); falcon7b_common (auto-upstream) (score 30: category 'LLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `qwen3 (auto-upstream)` | 100 | exact model_type 'qwen3' |
| 2 | `tt_transformers / simple_text_demo` (selected) | 40 | category 'LLM' default (generic runner) |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **ADAPT** | `token_embed` | `models/tt_transformers/tt/embedding.py` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **ADAPT** | `layer` | `models/tt_transformers/tt/decoder.py` | `—` |
| **ADAPT** | `encoder_stack` | `models/tt_transformers/tt/multimodal/llama_vision_encoder.py` | `—` |
| **ADAPT** | `decoder_head` | `models/tt_transformers/tt/lm_head.py` | `—` |
| **ADAPT** | `decoder_layer` | `models/tt_transformers/tt/decoder.py` | `—` |
| **REUSE** | `r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **REUSE** | `rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |

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
| hidden_size | 4096 | — |
| intermediate_size | 12288 | — |
| max_position_embeddings | 40960 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 36 | — |
| num_key_value_heads | 8 | — |
| vocab_size | 151936 | — |

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 4096 | — |
| intermediate_size | 12288 | — |
| max_position_embeddings | 40960 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 36 | — |
| num_key_value_heads | 8 | — |
| vocab_size | 151936 | — |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 4096 | — |
| intermediate_size | 12288 | — |
| max_position_embeddings | 40960 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 36 | — |
| num_key_value_heads | 8 | — |
| vocab_size | 151936 | — |

### `layer` — ADAPT
_reuse_registry: decoder_layer -> models/tt_transformers/tt/decoder.py::TransformerBlock (ADAPT). Composite Attention+MLP+RMSNorm block; tt_transformers TransformerBlock is the template (presumed adaptable — runs canonical, LLM refines via iterate loop if per-component PCC < 0.99)._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 4096 | — |
| intermediate_size | 12288 | — |
| max_position_embeddings | 40960 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 36 | — |
| num_key_value_heads | 8 | — |
| vocab_size | 151936 | — |

### `encoder_stack` — ADAPT
_reuse_registry: llama_vision_encoder -> models/tt_transformers/tt/multimodal/llama_vision_encoder.py::TtLlamaVisionEncoder (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 4096 | — |
| intermediate_size | 12288 | — |
| max_position_embeddings | 40960 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 36 | — |
| num_key_value_heads | 8 | — |
| vocab_size | 151936 | — |

### `decoder_head` — ADAPT
_reuse_registry: lm_head -> models/tt_transformers/tt/lm_head.py::LMHead (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | silu | — |
| hidden_size | 4096 | — |
| intermediate_size | 12288 | — |
| max_position_embeddings | 40960 | — |
| num_attention_heads | 32 | — |
| num_hidden_layers | 36 | — |
| num_key_value_heads | 8 | — |
| vocab_size | 151936 | — |

### `decoder_layer` — ADAPT
_[supplemental module-tree pass] reuse_registry: decoder_layer -> models/tt_transformers/tt/decoder.py::TransformerBlock (ADAPT). Composite Attention+MLP+RMSNorm block; tt_transformers TransformerBlock is the template (presumed adaptable — runs canonical, LLM refines via iterate loop if per-component PCC < 0.99). | module-tree: occ=36 leaves=432 sample_paths=['layers.0', 'layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=145 leaves=145 sample_paths=['layers.0.self_attn.q_norm', 'layers.0.self_attn.k_norm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=36 leaves=144 sample_paths=['layers.0.mlp', 'layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['rotary_emb'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
