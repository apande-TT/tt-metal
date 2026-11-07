<!-- BEGIN bringup -->
# Bring-up run report — `zai-org/GLM-5.3-Flash`

_Generated: 2026-11-06 12:47:27 UTC_

_Topology: TP=32 x DP=1 (mesh 1x32, 32 chips) — run emit-e2e / optimize with `--mesh 1x32`._

## Outcome

**Converged** after bring-up.

## Backend & template match

- **Backend picked:** `GLM-5 Next (glm5_next hybrid linear-attn/MoE VLM)`  (EXACT (model_type match))
- **Closest template:** `models/demos/glm5_next`
- **Target model_type:** `glm5_next`
- **Sibling / template base:** `zai-org/GLM-5.3-Flash` (model_type=`glm5_next`)

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `GLM-5 Next (glm5_next hybrid linear-attn/MoE VLM)` (selected) | 100 | exact model_type 'glm5_next' |
| 2 | `tt_transformers / simple_text_demo (multimodal)` | 40 | category 'VLM' default (generic runner) |
| 3 | `Mistral-Small-3.1 (mistral3 VLM)` | 30 | category 'VLM' default |

## Placement summary

- **ON_DEVICE** (23): graduated, native ttnn, PCC verified
  - `r_m_s_norm`, `text_attention`, `text_decoder_layer`, `text_experts`, `text_forget_gate`, `text_hyper_connection`, `text_hyper_head`, `text_indexer`, `text_linear_attention`, `text_m_l_p`, `text_mo_e`, `text_model`, `text_r_m_s_norm`, `text_r_m_s_norm_gated`, `text_topk_router`, `text_unweighted_r_m_s_norm`, `vision_attention`, `vision_block`, `vision_m_l_p`, `vision_model`, `vision_patch_embed`, `vision_patch_merger`, `vision_rotary_embedding`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `text_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_attention.py::test_text_attention` |
| `text_decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_decoder_layer.py::test_text_decoder_layer` |
| `text_experts` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_experts.py::test_text_experts` |
| `text_forget_gate` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_forget_gate.py::test_text_forget_gate` |
| `text_hyper_connection` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_hyper_connection.py::test_text_hyper_connection` |
| `text_hyper_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_hyper_head.py::test_text_hyper_head` |
| `text_indexer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_indexer.py::test_text_indexer` |
| `text_linear_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_linear_attention.py::test_text_linear_attention` |
| `text_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_m_l_p.py::test_text_m_l_p` |
| `text_mo_e` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_mo_e.py::test_text_mo_e` |
| `text_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_model.py::test_text_model` |
| `text_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_r_m_s_norm.py::test_text_r_m_s_norm` |
| `text_r_m_s_norm_gated` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_r_m_s_norm_gated.py::test_text_r_m_s_norm_gated` |
| `text_topk_router` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_topk_router.py::test_text_topk_router` |
| `text_unweighted_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_text_unweighted_r_m_s_norm.py::test_text_unweighted_r_m_s_norm` |
| `vision_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_vision_attention.py::test_vision_attention` |
| `vision_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_vision_block.py::test_vision_block` |
| `vision_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_vision_m_l_p.py::test_vision_m_l_p` |
| `vision_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_vision_model.py::test_vision_model` |
| `vision_patch_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_vision_patch_embed.py::test_vision_patch_embed` |
| `vision_patch_merger` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_vision_patch_merger.py::test_vision_patch_merger` |
| `vision_rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/demos/glm_5_3_flash/tests/pcc/test_vision_rotary_embedding.py::test_vision_rotary_embedding` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_attention.py::test_text_attention -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_decoder_layer.py::test_text_decoder_layer -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_experts.py::test_text_experts -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_forget_gate.py::test_text_forget_gate -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_hyper_connection.py::test_text_hyper_connection -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_hyper_head.py::test_text_hyper_head -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_indexer.py::test_text_indexer -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_linear_attention.py::test_text_linear_attention -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_m_l_p.py::test_text_m_l_p -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_mo_e.py::test_text_mo_e -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_model.py::test_text_model -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_r_m_s_norm.py::test_text_r_m_s_norm -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_r_m_s_norm_gated.py::test_text_r_m_s_norm_gated -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_topk_router.py::test_text_topk_router -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_text_unweighted_r_m_s_norm.py::test_text_unweighted_r_m_s_norm -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_vision_attention.py::test_vision_attention -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_vision_block.py::test_vision_block -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_vision_m_l_p.py::test_vision_m_l_p -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_vision_model.py::test_vision_model -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_vision_patch_embed.py::test_vision_patch_embed -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_vision_patch_merger.py::test_vision_patch_merger -svv
python -m pytest models/demos/glm_5_3_flash/tests/pcc/test_vision_rotary_embedding.py::test_vision_rotary_embedding -svv
```

End-to-end / demo:
```bash
python -m pytest models/demos/glm_5_3_flash/tests/e2e/test_e2e_image_text_to_text.py -svv
python -m pytest models/demos/glm_5_3_flash/tests/e2e/test_e2e_trace.py -svv
python -m pytest models/demos/glm_5_3_flash/tests/e2e/test_image_text_to_text_perf.py -svv
python -m pytest models/demos/glm_5_3_flash/demo/demo.py::test_demo -svv
python -m pytest models/demos/glm_5_3_flash/demo/demo_image_text_to_text.py::test_demo -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e zai-org/GLM-5.3-Flash`
<!-- END bringup -->

<!-- BEGIN emit-e2e -->
# E2E report — `zai-org/GLM-5.3-Flash`

_Generated: 2026-11-06 12:47:27 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 23/23 on device (100%), 0/23 on CPU (0%)
- Graduated (ON_DEVICE) : 14/23 (60%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=9  ADAPT-wired=1  NEW-native=13  NEW-partial-CPU=0
- on CPU    : NEW-fallback=0  REUSE/ADAPT-not-wired=0
- operations: 23/23 on device (100%), 0/23 on CPU (0%)  (component-level estimate; run with --op-synth for op-level granularity)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `image_text_to_text` | n/a | `models/demos/glm_5_3_flash/demo/demo_image_text_to_text.py` | `models/demos/glm_5_3_flash/tests/e2e/test_e2e_image_text_to_text.py` | `models/demos/glm_5_3_flash/tests/e2e/test_image_text_to_text_perf.py` |

## Reproduce

### image_text_to_text
```bash
python models/demos/glm_5_3_flash/demo/demo_image_text_to_text.py
pytest models/demos/glm_5_3_flash/tests/e2e/test_e2e_image_text_to_text.py -svv
pytest models/demos/glm_5_3_flash/tests/e2e/test_image_text_to_text_perf.py -svv
```
<!-- END emit-e2e -->

<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 23, ungraduated: 0
<!-- END trace-gate -->
