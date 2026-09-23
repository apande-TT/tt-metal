# Bring-up plan: `mistralai/Voxtral-4B-TTS-2603`

Backend template: **Mistral-Small-3.1 (mistral3 VLM)** at `models/tt_transformers/demo/simple_text_demo.py` (canonical HF id: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`).
New `model_type` = `voxtral_tts`; sibling `model_type` = `mistral3`.

**Summary:** 8 REUSE · 19 NEW component(s).

> **Notes:**
> - new model_type=`voxtral_tts` differs from sibling model_type=`mistral3` — expect attention + encoder stacks to be NEW even if other shapes line up.
> - Top sibling candidates (per-component reuse targets are pulled from whichever sibling provides them, not only the first): tt_transformers / simple_text_demo (multimodal) (score 40: category 'VLM' default (generic runner)); Mistral-Small-3.1 (mistral3 VLM) (score 30: category 'VLM' default); qwen25_vl (auto-upstream) (score 30: category 'VLM' default)

## Sibling candidates (ranked)

Top backends by match score — components pull their reuse target from whichever of these provides it, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 2 | `Mistral-Small-3.1 (mistral3 VLM)` (selected) | 30 | category 'VLM' default |
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
| **NEW** | `mistral_model` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `mistral_decoder_layer` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **REUSE** | `mistral_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **REUSE** | `mistral_m_l_p` | `models/tt_transformers/tt/mlp.py` | `—` |
| **NEW** | `voxtral_t_t_s_audio_tokenizer` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `codec_transformer` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `codec_transformer_block` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **REUSE** | `mistral_r_m_s_norm` | `models/common/rmsnorm.py` | `—` |
| **REUSE** | `codec_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `flow_matching_audio_transformer` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `feed_forward` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `acoustic_transformer_block` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **REUSE** | `bidirectional_attention` | `models/tt_transformers/tt/attention.py` | `—` |
| **NEW** | `parametrization_list` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `weight_norm` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `causal_conv_transpose1d` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `parametrized_conv_transpose1d` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `causal_conv1d` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `mistral_audio_codebook` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `parametrized_conv1d` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `acoustic_codebook` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **REUSE** | `mistral_rotary_embedding` | `models/tt_transformers/tt/rope.py` | `—` |
| **NEW** | `multi_vocab_embeddings` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `semantic_codebook` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |
| **NEW** | `time_embedding` | `—` | `transformers/src/transformers/models/voxtral_tts/modeling_voxtral_tts.py` |

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
| hidden_act | — | silu |
| hidden_size | — | 5120 |
| intermediate_size | — | 32768 |
| max_position_embeddings | 128000 | 131072 |
| num_attention_heads | — | 32 |
| num_hidden_layers | — | 40 |
| num_key_value_heads | — | 8 |
| vocab_size | 131072 | 131072 |

### `attention` — REUSE
_reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0._

| field | new model | sibling |
|---|---|---|
| hidden_act | — | silu |
| hidden_size | — | 5120 |
| intermediate_size | — | 32768 |
| max_position_embeddings | 128000 | 131072 |
| num_attention_heads | — | 32 |
| num_hidden_layers | — | 40 |
| num_key_value_heads | — | 8 |
| vocab_size | 131072 | 131072 |

### `mlp` — REUSE
_reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh._

| field | new model | sibling |
|---|---|---|
| hidden_act | — | silu |
| hidden_size | — | 5120 |
| intermediate_size | — | 32768 |
| max_position_embeddings | 128000 | 131072 |
| num_attention_heads | — | 32 |
| num_hidden_layers | — | 40 |
| num_key_value_heads | — | 8 |
| vocab_size | 131072 | 131072 |

### `layer` — ADAPT
_reuse_registry: llama_layernorm -> models/tt_transformers/tt/multimodal/llama_layernorm.py::TtLayerNorm (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | — | silu |
| hidden_size | — | 5120 |
| intermediate_size | — | 32768 |
| max_position_embeddings | 128000 | 131072 |
| num_attention_heads | — | 32 |
| num_hidden_layers | — | 40 |
| num_key_value_heads | — | 8 |
| vocab_size | 131072 | 131072 |

### `encoder_stack` — ADAPT
_reuse_registry: llama_vision_encoder -> models/tt_transformers/tt/multimodal/llama_vision_encoder.py::TtLlamaVisionEncoder (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | — | silu |
| hidden_size | — | 5120 |
| intermediate_size | — | 32768 |
| max_position_embeddings | 128000 | 131072 |
| num_attention_heads | — | 32 |
| num_hidden_layers | — | 40 |
| num_key_value_heads | — | 8 |
| vocab_size | 131072 | 131072 |

### `decoder_head` — ADAPT
_reuse_registry: lm_head -> models/tt_transformers/tt/lm_head.py::LMHead (ADAPT). auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted._

| field | new model | sibling |
|---|---|---|
| hidden_act | — | silu |
| hidden_size | — | 5120 |
| intermediate_size | — | 32768 |
| max_position_embeddings | 128000 | 131072 |
| num_attention_heads | — | 32 |
| num_hidden_layers | — | 40 |
| num_key_value_heads | — | 8 |
| vocab_size | 131072 | 131072 |

### `mistral_model` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=263 sample_paths=['model'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral_decoder_layer` — NEW
_[supplemental module-tree pass] module-tree: occ=26 leaves=260 sample_paths=['model.layers.0', 'model.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=26 leaves=104 sample_paths=['model.layers.0.self_attn', 'model.layers.1.self_attn'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral_m_l_p` — REUSE
_[supplemental module-tree pass] reuse_registry: swiglu_mlp -> models/tt_transformers/tt/mlp.py::MLP (REUSE). derived from compatibility.py BUILDING_BLOCKS 'SwiGLU MLP'. hidden_act dispatched via activation_map; supports silu/gelu/relu/quick_gelu/gelu_pytorch_tanh. | module-tree: occ=26 leaves=104 sample_paths=['model.layers.0.mlp', 'model.layers.1.mlp'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `voxtral_t_t_s_audio_tokenizer` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=97 sample_paths=['audio_tokenizer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `codec_transformer` — NEW
_[supplemental module-tree pass] module-tree: occ=4 leaves=88 sample_paths=['audio_tokenizer.decoder_blocks.1', 'audio_tokenizer.decoder_blocks.3'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `codec_transformer_block` — NEW
_[supplemental module-tree pass] module-tree: occ=8 leaves=88 sample_paths=['audio_tokenizer.decoder_blocks.1.layers.0', 'audio_tokenizer.decoder_blocks.1.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral_r_m_s_norm` — REUSE
_[supplemental module-tree pass] reuse_registry: rmsnorm_text -> models/common/rmsnorm.py::RMSNorm (REUSE). derived from compatibility.py BUILDING_BLOCKS 'RMSNorm (text)'. ttnn.rms_norm requires TILE layout; distributed RMSNorm handles multi-chip. | module-tree: occ=53 leaves=53 sample_paths=['model.layers.0.input_layernorm', 'model.layers.0.post_attention_layernorm'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `codec_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=8 leaves=48 sample_paths=['audio_tokenizer.decoder_blocks.1.layers.0.attention', 'audio_tokenizer.decoder_blocks.1.layers.1.attention'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `flow_matching_audio_transformer` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=34 sample_paths=['acoustic_transformer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `feed_forward` — NEW
_[supplemental module-tree pass] module-tree: occ=11 leaves=33 sample_paths=['acoustic_transformer.layers.0.feed_forward', 'acoustic_transformer.layers.1.feed_forward'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `acoustic_transformer_block` — NEW
_[supplemental module-tree pass] module-tree: occ=3 leaves=27 sample_paths=['acoustic_transformer.layers.0', 'acoustic_transformer.layers.1'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `bidirectional_attention` — REUSE
_[supplemental module-tree pass] reuse_registry: gqa_attention -> models/tt_transformers/tt/attention.py::Attention (REUSE). derived from compatibility.py BUILDING_BLOCKS 'GQA attention'. Requires num_attention_heads % num_key_value_heads == 0. | module-tree: occ=3 leaves=12 sample_paths=['acoustic_transformer.layers.0.attention', 'acoustic_transformer.layers.1.attention'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `parametrization_list` — NEW
_[supplemental module-tree pass] module-tree: occ=5 leaves=5 sample_paths=['audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight', 'audio_tokenizer.decoder_blocks.2.conv.parametrizations.weight'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `weight_norm` — NEW
_[supplemental module-tree pass] module-tree: occ=5 leaves=5 sample_paths=['audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.0', 'audio_tokenizer.decoder_blocks.2.conv.parametrizations.weight.0'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `causal_conv_transpose1d` — NEW
_[supplemental module-tree pass] module-tree: occ=3 leaves=3 sample_paths=['audio_tokenizer.decoder_blocks.2', 'audio_tokenizer.decoder_blocks.4'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `parametrized_conv_transpose1d` — NEW
_[supplemental module-tree pass] module-tree: occ=3 leaves=3 sample_paths=['audio_tokenizer.decoder_blocks.2.conv', 'audio_tokenizer.decoder_blocks.4.conv'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `causal_conv1d` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=2 sample_paths=['audio_tokenizer.decoder_blocks.0', 'audio_tokenizer.output_proj'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral_audio_codebook` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=2 sample_paths=['audio_tokenizer.quantizer'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `parametrized_conv1d` — NEW
_[supplemental module-tree pass] module-tree: occ=2 leaves=2 sample_paths=['audio_tokenizer.decoder_blocks.0.conv', 'audio_tokenizer.output_proj.conv'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `acoustic_codebook` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['audio_tokenizer.quantizer.acoustic_codebook'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `mistral_rotary_embedding` — REUSE
_[supplemental module-tree pass] reuse_registry: standard_rope -> models/tt_transformers/tt/rope.py::RotaryEmbedding (REUSE). derived from compatibility.py BUILDING_BLOCKS 'Standard RoPE'. | module-tree: occ=1 leaves=1 sample_paths=['model.rotary_emb'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `multi_vocab_embeddings` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['audio_tokenizer.audio_token_embedding'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `semantic_codebook` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['audio_tokenizer.quantizer.semantic_codebook'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

### `time_embedding` — NEW
_[supplemental module-tree pass] module-tree: occ=1 leaves=1 sample_paths=['acoustic_transformer.time_embedding'] (primary extractor's template did not cover this class — falling back to module-tree discovery + op_classifier classification)._

| field | new model | sibling |
|---|---|---|

## Bring-up checklist

1. For each **REUSE** row above, import the sibling tt-module directly in the scaffolded demo's `tt/` instead of editing the cloned copy. The global PCC gate enforces correctness — if it fails, the brain auto-promotes REUSE to NEW via `force_adapt_all`.
2. For each **NEW** row, open the matching file under `_stubs/` and replace the `NotImplementedError` (or torch fallback) with a TTNN port driven by the linked HF reference. If a sibling tt-file with the same role exists, reuse its layout and update shape constants.
4. Once every component passes its PCC test, run `python -m scripts.tt_hw_planner prepare $MODEL --execute` to confirm the assembled model runs end-to-end.
