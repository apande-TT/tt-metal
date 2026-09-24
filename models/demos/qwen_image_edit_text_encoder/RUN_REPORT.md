<!-- BEGIN bringup -->
# Bring-up run report — `/tmp/tt_hw_planner_components/qwen_image_edit_text_encoder`

_Generated: 2026-09-23 10:37:06 UTC_

_Topology: TP=4 x DP=2 (mesh 2x4, 8 chips) — run emit-e2e / optimize with `--mesh 2x4`._

## Outcome

**Converged** after 2 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `qwen25_vl (auto-upstream)`
- **Closest template:** `models/demos/qwen25_vl`
- **Target model_type:** `qwen2_5_vl`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `qwen25_vl (auto-upstream)` (selected) | 100 | exact model_type 'qwen2_5_vl' |
| 2 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 3 | `Mistral-Small-3.1 (mistral3 VLM)` | 30 | category 'VLM' default |

## Placement summary

- **ON_DEVICE** (19): graduated, native ttnn, PCC verified
  - `attention`, `decoder_head`, `encoder_stack`, `layer`, `m_l_p`, `mlp`, `token_embed`, `v_l_attention`, `v_l_decoder_layer`, `v_l_m_l_p`, `v_l_patch_merger`, `v_l_r_m_s_norm`, `v_l_rotary_embedding`, `v_l_text_model`, `v_l_vision_attention`, `v_l_vision_block`, `vision_patch_embed`, `vision_rotary_embedding`, `vision_transformer_pretrained_model`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_attention.py::test_attention` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_layer.py::test_layer` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_mlp.py::test_mlp` |
| `token_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_token_embed.py::test_token_embed` |
| `v_l_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_attention.py::test_v_l_attention` |
| `v_l_decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_decoder_layer.py::test_v_l_decoder_layer` |
| `v_l_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_m_l_p.py::test_v_l_m_l_p` |
| `v_l_patch_merger` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_patch_merger.py::test_v_l_patch_merger` |
| `v_l_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_r_m_s_norm.py::test_v_l_r_m_s_norm` |
| `v_l_rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_rotary_embedding.py::test_v_l_rotary_embedding` |
| `v_l_text_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_text_model.py::test_v_l_text_model` |
| `v_l_vision_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_vision_attention.py::test_v_l_vision_attention` |
| `v_l_vision_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_vision_block.py::test_v_l_vision_block` |
| `vision_patch_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_vision_patch_embed.py::test_vision_patch_embed` |
| `vision_rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_vision_rotary_embedding.py::test_vision_rotary_embedding` |
| `vision_transformer_pretrained_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/qwen_image_edit_text_encoder/tests/pcc/test_vision_transformer_pretrained_model.py::test_vision_transformer_pretrained_model` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_token_embed.py::test_token_embed -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_attention.py::test_v_l_attention -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_decoder_layer.py::test_v_l_decoder_layer -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_m_l_p.py::test_v_l_m_l_p -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_patch_merger.py::test_v_l_patch_merger -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_r_m_s_norm.py::test_v_l_r_m_s_norm -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_rotary_embedding.py::test_v_l_rotary_embedding -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_text_model.py::test_v_l_text_model -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_vision_attention.py::test_v_l_vision_attention -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_v_l_vision_block.py::test_v_l_vision_block -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_vision_patch_embed.py::test_vision_patch_embed -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_vision_rotary_embedding.py::test_vision_rotary_embedding -svv
python -m pytest models/demos/qwen_image_edit_text_encoder/tests/pcc/test_vision_transformer_pretrained_model.py::test_vision_transformer_pretrained_model -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /tmp/tt_hw_planner_components/qwen_image_edit_text_encoder`
<!-- END bringup -->
