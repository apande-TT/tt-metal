<!-- BEGIN bringup -->
# Bring-up run report — `mistralai/Voxtral-4B-TTS-2603`

_Generated: 2026-09-22 11:54:21 UTC_

_Topology: single-device (1 chip)._

## Outcome

**Converged** after 1 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `Mistral-Small-3.1 (mistral3 VLM)`  (TEMPLATE-FALLBACK (model_type mismatch — closest sibling by category))
- **Closest template:** `models/tt_transformers/demo/simple_text_demo.py`
- **Target model_type:** `voxtral_tts`
- **Sibling / template base:** `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (model_type=`mistral3`)

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 2 | `Mistral-Small-3.1 (mistral3 VLM)` (selected) | 30 | category 'VLM' default |
| 3 | `qwen25_vl (auto-upstream)` | 30 | category 'VLM' default |

## Placement summary

- **ON_DEVICE** (31): graduated, native ttnn, PCC verified
  - `acoustic_codebook`, `acoustic_transformer_block`, `attention`, `bidirectional_attention`, `causal_conv1d`, `causal_conv_transpose1d`, `codec_attention`, `codec_transformer`, `codec_transformer_block`, `decoder_head`, `encoder_stack`, `feed_forward`, `flow_matching_audio_transformer`, `layer`, `mistral_attention`, `mistral_audio_codebook`, `mistral_decoder_layer`, `mistral_m_l_p`, `mistral_model`, `mistral_r_m_s_norm`, `mistral_rotary_embedding`, `mlp`, `multi_vocab_embeddings`, `parametrization_list`, `parametrized_conv1d`, `parametrized_conv_transpose1d`, `semantic_codebook`, `time_embedding`, `token_embed`, `voxtral_t_t_s_audio_tokenizer`, `weight_norm`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `acoustic_codebook` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_acoustic_codebook.py::test_acoustic_codebook` |
| `acoustic_transformer_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_acoustic_transformer_block.py::test_acoustic_transformer_block` |
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_attention.py::test_attention` |
| `bidirectional_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_bidirectional_attention.py::test_bidirectional_attention` |
| `causal_conv1d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_causal_conv1d.py::test_causal_conv1d` |
| `causal_conv_transpose1d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_causal_conv_transpose1d.py::test_causal_conv_transpose1d` |
| `codec_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_codec_attention.py::test_codec_attention` |
| `codec_transformer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_codec_transformer.py::test_codec_transformer` |
| `codec_transformer_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_codec_transformer_block.py::test_codec_transformer_block` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `feed_forward` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_feed_forward.py::test_feed_forward` |
| `flow_matching_audio_transformer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_flow_matching_audio_transformer.py::test_flow_matching_audio_transformer` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_layer.py::test_layer` |
| `mistral_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_attention.py::test_mistral_attention` |
| `mistral_audio_codebook` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_audio_codebook.py::test_mistral_audio_codebook` |
| `mistral_decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_decoder_layer.py::test_mistral_decoder_layer` |
| `mistral_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_m_l_p.py::test_mistral_m_l_p` |
| `mistral_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_model.py::test_mistral_model` |
| `mistral_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_r_m_s_norm.py::test_mistral_r_m_s_norm` |
| `mistral_rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_rotary_embedding.py::test_mistral_rotary_embedding` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mlp.py::test_mlp` |
| `multi_vocab_embeddings` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_multi_vocab_embeddings.py::test_multi_vocab_embeddings` |
| `parametrization_list` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_parametrization_list.py::test_parametrization_list` |
| `parametrized_conv1d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_parametrized_conv1d.py::test_parametrized_conv1d` |
| `parametrized_conv_transpose1d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_parametrized_conv_transpose1d.py::test_parametrized_conv_transpose1d` |
| `semantic_codebook` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_semantic_codebook.py::test_semantic_codebook` |
| `time_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_time_embedding.py::test_time_embedding` |
| `token_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_token_embed.py::test_token_embed` |
| `voxtral_t_t_s_audio_tokenizer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_voxtral_t_t_s_audio_tokenizer.py::test_voxtral_t_t_s_audio_tokenizer` |
| `weight_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_weight_norm.py::test_weight_norm` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_acoustic_codebook.py::test_acoustic_codebook -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_acoustic_transformer_block.py::test_acoustic_transformer_block -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_bidirectional_attention.py::test_bidirectional_attention -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_causal_conv1d.py::test_causal_conv1d -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_causal_conv_transpose1d.py::test_causal_conv_transpose1d -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_codec_attention.py::test_codec_attention -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_codec_transformer.py::test_codec_transformer -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_codec_transformer_block.py::test_codec_transformer_block -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_feed_forward.py::test_feed_forward -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_flow_matching_audio_transformer.py::test_flow_matching_audio_transformer -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_attention.py::test_mistral_attention -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_audio_codebook.py::test_mistral_audio_codebook -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_decoder_layer.py::test_mistral_decoder_layer -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_m_l_p.py::test_mistral_m_l_p -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_model.py::test_mistral_model -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_r_m_s_norm.py::test_mistral_r_m_s_norm -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mistral_rotary_embedding.py::test_mistral_rotary_embedding -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_multi_vocab_embeddings.py::test_multi_vocab_embeddings -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_parametrization_list.py::test_parametrization_list -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_parametrized_conv1d.py::test_parametrized_conv1d -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_parametrized_conv_transpose1d.py::test_parametrized_conv_transpose1d -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_semantic_codebook.py::test_semantic_codebook -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_time_embedding.py::test_time_embedding -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_token_embed.py::test_token_embed -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_voxtral_t_t_s_audio_tokenizer.py::test_voxtral_t_t_s_audio_tokenizer -svv
python -m pytest models/tt_transformers/demo/voxtral_4b_tts_2603/tests/pcc/test_weight_norm.py::test_weight_norm -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e mistralai/Voxtral-4B-TTS-2603`
<!-- END bringup -->
