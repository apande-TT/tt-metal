# Bring-up plan: `facebook/hf-seamless-m4t-large`

Backend template: **XTTS-v2 (multilingual TTS)** at `models/demos/xtts_v2` (canonical HF id: `/local/ttuser/apande/models/XTTS-v2-hf`).
New `model_type` = `seamless_m4t`; sibling `model_type` = `xtts`.

**Summary:** 2 REUSE · 23 NEW component(s).

> **Notes:**
> - new model_type=`seamless_m4t` differs from sibling model_type=`xtts` — expect attention + encoder stacks to be NEW even if other shapes line up.

## Components

| Status | Component | Sibling tt-file (reuse target) | HF reference (for NEW) |
|---|---|---|---|
| **NEW** | `seamless_m4_t_speech_encoder` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_encoder` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_encoder_layer` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_decoder` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_decoder_layer` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_encoder` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **REUSE** | `seamless_m4_t_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `seamless_m4_t_encoder_layer` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_feed_forward` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_feed_forward_network` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_convolution_module` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_text_to_unit_for_conditional_generation` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_text_to_unit_model` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **REUSE** | `seamless_m4_t_conformer_self_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `seamless_m4_t_code_hifi_gan` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_hifi_gan` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `hifi_gan_residual_block` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `g_l_u` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_adapter` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_adapter_layer` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_variance_predictor` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_feature_projection` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_scaled_word_embedding` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_sinusoidal_positional_embedding` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |
| **NEW** | `seamless_m4_t_conformer_rel_positional_embedding` | `—` | `transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py` |

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

### `seamless_m4_t_speech_encoder` — NEW
_module-tree: occ=1 leaves=725 sample_paths=['speech_encoder']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_encoder` — NEW
_module-tree: occ=1 leaves=699 sample_paths=['speech_encoder.encoder']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_encoder_layer` — NEW
_module-tree: occ=24 leaves=696 sample_paths=['speech_encoder.encoder.layers.0', 'speech_encoder.encoder.layers.1']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_decoder` — NEW
_module-tree: occ=2 leaves=546 sample_paths=['text_decoder', 't2u_model.model.decoder']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_decoder_layer` — NEW
_module-tree: occ=30 leaves=540 sample_paths=['text_decoder.layers.0', 'text_decoder.layers.1']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_encoder` — NEW
_module-tree: occ=2 leaves=364 sample_paths=['text_encoder', 't2u_model.model.encoder']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=90 leaves=360 sample_paths=['text_encoder.layers.0.self_attn', 'text_encoder.layers.1.self_attn']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_encoder_layer` — NEW
_module-tree: occ=30 leaves=360 sample_paths=['text_encoder.layers.0', 'text_encoder.layers.1']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_feed_forward` — NEW
_module-tree: occ=50 leaves=250 sample_paths=['speech_encoder.encoder.layers.0.ffn1', 'speech_encoder.encoder.layers.0.ffn2']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_feed_forward_network` — NEW
_module-tree: occ=60 leaves=240 sample_paths=['text_encoder.layers.0.ffn', 'text_encoder.layers.1.ffn']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_convolution_module` — NEW
_module-tree: occ=24 leaves=192 sample_paths=['speech_encoder.encoder.layers.0.conv_module', 'speech_encoder.encoder.layers.1.conv_module']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_text_to_unit_for_conditional_generation` — NEW
_module-tree: occ=1 leaves=185 sample_paths=['t2u_model']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_text_to_unit_model` — NEW
_module-tree: occ=1 leaves=184 sample_paths=['t2u_model.model']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_self_attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=25 leaves=149 sample_paths=['speech_encoder.encoder.layers.0.self_attn', 'speech_encoder.encoder.layers.1.self_attn']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_code_hifi_gan` — NEW
_module-tree: occ=1 leaves=107 sample_paths=['vocoder']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_hifi_gan` — NEW
_module-tree: occ=1 leaves=97 sample_paths=['vocoder.hifi_gan']_

| field | new model | sibling |
|---|---|---|

### `hifi_gan_residual_block` — NEW
_module-tree: occ=15 leaves=90 sample_paths=['vocoder.hifi_gan.resblocks.0', 'vocoder.hifi_gan.resblocks.1']_

| field | new model | sibling |
|---|---|---|

### `g_l_u` — NEW
_module-tree: occ=25 leaves=25 sample_paths=['speech_encoder.encoder.layers.0.conv_module.glu', 'speech_encoder.encoder.layers.1.conv_module.glu']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_adapter` — NEW
_module-tree: occ=1 leaves=17 sample_paths=['speech_encoder.adapter']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_adapter_layer` — NEW
_module-tree: occ=1 leaves=17 sample_paths=['speech_encoder.adapter.layers.0']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_variance_predictor` — NEW
_module-tree: occ=1 leaves=7 sample_paths=['vocoder.dur_predictor']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_feature_projection` — NEW
_module-tree: occ=1 leaves=3 sample_paths=['speech_encoder.feature_projection']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_scaled_word_embedding` — NEW
_module-tree: occ=3 leaves=3 sample_paths=['text_encoder.embed_tokens', 'text_decoder.embed_tokens']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_sinusoidal_positional_embedding` — NEW
_module-tree: occ=3 leaves=3 sample_paths=['text_encoder.embed_positions', 'text_decoder.embed_positions']_

| field | new model | sibling |
|---|---|---|

### `seamless_m4_t_conformer_rel_positional_embedding` — NEW
_module-tree: occ=1 leaves=1 sample_paths=['speech_encoder.encoder.embed_positions']_

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
