<!-- BEGIN bringup -->
# Bring-up run report — `mistralai/Devstral-Small-2-24B-Instruct-2512`

_Generated: 2026-10-09 23:45:49 UTC_

_Topology: TP=4 x DP=1 (mesh 1x4, 4 chips) — run emit-e2e / optimize with `--mesh 1x4`._

## Outcome

**Converged** after 10 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `Mistral-Small-3.1 (mistral3 VLM)`  (EXACT (model_type match))
- **Closest template:** `models/tt_transformers/demo/simple_text_demo.py`
- **Target model_type:** `mistral3`
- **Sibling / template base:** `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (model_type=`mistral3`)

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `Mistral-Small-3.1 (mistral3 VLM)` (selected) | 100 | exact model_type 'mistral3' |
| 2 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 3 | `qwen25_vl (auto-upstream)` | 30 | category 'VLM' default |

## Placement summary

- **ON_DEVICE** (22): graduated, native ttnn, PCC verified
  - `attention`, `decoder_head`, `encoder_stack`, `layer`, `ministral3_attention`, `ministral3_decoder_layer`, `ministral3_m_l_p`, `ministral3_model`, `ministral3_r_m_s_norm`, `ministral3_rotary_embedding`, `mistral3_multi_modal_projector`, `mistral3_patch_merger`, `mistral3_r_m_s_norm`, `mlp`, `pixtral_attention`, `pixtral_attention_layer`, `pixtral_m_l_p`, `pixtral_r_m_s_norm`, `pixtral_rotary_embedding`, `pixtral_transformer`, `pixtral_vision_model`, `token_embed`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_attention.py::test_attention` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_layer.py::test_layer` |
| `ministral3_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_attention.py::test_ministral3_attention` |
| `ministral3_decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_decoder_layer.py::test_ministral3_decoder_layer` |
| `ministral3_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_m_l_p.py::test_ministral3_m_l_p` |
| `ministral3_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_model.py::test_ministral3_model` |
| `ministral3_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_r_m_s_norm.py::test_ministral3_r_m_s_norm` |
| `ministral3_rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_rotary_embedding.py::test_ministral3_rotary_embedding` |
| `mistral3_multi_modal_projector` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mistral3_multi_modal_projector.py::test_mistral3_multi_modal_projector` |
| `mistral3_patch_merger` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mistral3_patch_merger.py::test_mistral3_patch_merger` |
| `mistral3_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mistral3_r_m_s_norm.py::test_mistral3_r_m_s_norm` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mlp.py::test_mlp` |
| `pixtral_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_attention.py::test_pixtral_attention` |
| `pixtral_attention_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_attention_layer.py::test_pixtral_attention_layer` |
| `pixtral_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_m_l_p.py::test_pixtral_m_l_p` |
| `pixtral_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_r_m_s_norm.py::test_pixtral_r_m_s_norm` |
| `pixtral_rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_rotary_embedding.py::test_pixtral_rotary_embedding` |
| `pixtral_transformer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_transformer.py::test_pixtral_transformer` |
| `pixtral_vision_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_vision_model.py::test_pixtral_vision_model` |
| `token_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_token_embed.py::test_token_embed` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_attention.py::test_ministral3_attention -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_decoder_layer.py::test_ministral3_decoder_layer -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_m_l_p.py::test_ministral3_m_l_p -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_model.py::test_ministral3_model -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_r_m_s_norm.py::test_ministral3_r_m_s_norm -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_ministral3_rotary_embedding.py::test_ministral3_rotary_embedding -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mistral3_multi_modal_projector.py::test_mistral3_multi_modal_projector -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mistral3_patch_merger.py::test_mistral3_patch_merger -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mistral3_r_m_s_norm.py::test_mistral3_r_m_s_norm -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_attention.py::test_pixtral_attention -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_attention_layer.py::test_pixtral_attention_layer -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_m_l_p.py::test_pixtral_m_l_p -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_r_m_s_norm.py::test_pixtral_r_m_s_norm -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_rotary_embedding.py::test_pixtral_rotary_embedding -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_transformer.py::test_pixtral_transformer -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_pixtral_vision_model.py::test_pixtral_vision_model -svv
python -m pytest models/tt_transformers/demo/devstral_small_2_24b_instruct_2512/tests/pcc/test_token_embed.py::test_token_embed -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e mistralai/Devstral-Small-2-24B-Instruct-2512`
<!-- END bringup -->

<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 22, ungraduated: 0

fresh capture: invalid terminated by SIGKILL (rc=-9) -- the step was killed, not failed. TRACE_STAGE_BYTES[encode]=37031624976 ops=6042
<!-- END trace-gate -->
