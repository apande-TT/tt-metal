# Bring-up plan: `/tmp/tt_hw_planner_components/qwen_image_edit_transformer`

Backend template: **tt_dit/qwenimage (auto-upstream)** at `models/tt_dit/pipelines/qwenimage` (canonical HF id: `None`).

**Summary:** 3 REUSE · 7 NEW component(s).

> **Notes:**
> - Sibling config could not be fetched; classification falls back to NEW for components without a clear file match. Set HF_TOKEN or pre-download `None` and re-run for a sharper diff.
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): hf_eager universal (Image / diffusion) (score 40: category 'Image' default (generic runner)); Stable Diffusion 1.4 (score 30: category 'Image' default); stable_diffusion_xl_base (auto-upstream) (score 30: category 'Image' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `hf_eager universal (Image / diffusion)` | 40 | category 'Image' default (generic runner) |
| 2 | `Stable Diffusion 1.4` | 30 | category 'Image' default |
| 3 | `stable_diffusion_xl_base (auto-upstream)` | 30 | category 'Image' default |

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **ADAPT** | `patch_embed` | `models/tt_transformers/tt/multimodal/llama_conv2d_patch.py` | `—` |
| **REUSE** | `self_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mlp` | `models/tt_transformers/tt/mlp.py` | `—` |
| **ADAPT** | `layer` | `models/tt_transformers/tt/multimodal/llama_layernorm.py` | `—` |
| **ADAPT** | `encoder_stack` | `models/tt_transformers/tt/multimodal/llama_vision_encoder.py` | `—` |
| **ADAPT** | `decoder_head` | `models/tt_transformers/tt/lm_head.py` | `—` |
| **NEW** | `qwen_image_transformer_block` | `—` | `—` |
| **REUSE** | `attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `feed_forward` | `—` | `—` |
| **NEW** | `qwen_timestep_proj_embeddings` | `—` | `—` |
| **NEW** | `ada_layer_norm_continuous` | `—` | `—` |
| **NEW** | `timestep_embedding` | `—` | `—` |
| **NEW** | `qwen_embed_rope` | `—` | `—` |
| **NEW** | `timesteps` | `—` | `—` |

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

### `patch_embed` — ADAPT
_reuse_registry: llama_conv2d_patch -> models/tt_transformers/tt/multimodal/llama_conv2d_patch.py::TtLlamaConv2dPatch (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| num_attention_heads | 24 | — |
| patch_size | 2 | — |

### `self_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|
| num_attention_heads | 24 | — |
| patch_size | 2 | — |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| num_attention_heads | 24 | — |
| patch_size | 2 | — |

### `layer` — ADAPT
_reuse_registry: llama_layernorm -> models/tt_transformers/tt/multimodal/llama_layernorm.py::TtLayerNorm (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| num_attention_heads | 24 | — |
| patch_size | 2 | — |

### `encoder_stack` — ADAPT
_reuse_registry: llama_vision_encoder -> models/tt_transformers/tt/multimodal/llama_vision_encoder.py::TtLlamaVisionEncoder (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| num_attention_heads | 24 | — |
| patch_size | 2 | — |

### `decoder_head` — ADAPT
_reuse_registry: lm_head -> models/tt_transformers/tt/lm_head.py::LMHead (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| num_attention_heads | 24 | — |
| patch_size | 2 | — |

### `qwen_image_transformer_block` — NEW
_[supplemental module-tree pass] module-tree: occ=60 leaves=1620 sample_paths=['transformer_blocks.0', 'transformer_blocks.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=60 leaves=780 sample_paths=['transformer_blocks.0.attn', 'transformer_blocks.1.attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `feed_forward` — NEW
_[supplemental module-tree pass] module-tree: occ=120 leaves=360 sample_paths=['transformer_blocks.0.img_mlp', 'transformer_blocks.0.txt_mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen_timestep_proj_embeddings` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=4 sample_paths=['time_text_embed'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `ada_layer_norm_continuous` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=3 sample_paths=['norm_out'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `timestep_embedding` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=3 sample_paths=['time_text_embed.timestep_embedder'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `qwen_embed_rope` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['pos_embed'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `timesteps` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['time_text_embed.time_proj'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
