<!-- BEGIN bringup -->
# Bring-up run report — `Aleph-Alpha/Kolibri-1`

_Generated: 2026-10-08 12:48:43 UTC_

_Topology: TP=4 x DP=1 (mesh 1x4, 4 chips) — run emit-e2e / optimize with `--mesh 1x4`._

## Outcome

**Converged** after 1 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `tt_transformers / simple_text_demo`
- **Closest template:** `models/tt_transformers/demo/simple_text_demo.py`
- **Target model_type:** `kolibri1`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo` (selected) | 40 | category 'LLM' default (generic runner) |
| 2 | `NemotronH (nemotron_h hybrid Mamba2/MoE)` | 30 | category 'LLM' default |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Placement summary

- **ON_DEVICE** (9): graduated, native ttnn, PCC verified
  - `attention`, `decoder_head`, `decoder_layer`, `f_p8_linear`, `m_l_p`, `r_m_s_norm`, `router`, `sparse_moe_block`, `token_embed`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (1): retry next run
  - `model`
- **CPU_REUSE** (3): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device
  - `encoder_stack`, `layer`, `mlp`

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_attention.py::test_attention` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_decoder_layer.py::test_decoder_layer` |
| `f_p8_linear` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_f_p8_linear.py::test_f_p8_linear` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `router` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_router.py::test_router` |
| `sparse_moe_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_sparse_moe_block.py::test_sparse_moe_block` |
| `token_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_token_embed.py::test_token_embed` |
| `model` | [wait] | PENDING | retry next run | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_model.py::test_model` |
| `encoder_stack` | [ cpu ] | CPU_REUSE | REUSE/ADAPT tag not wired to a ttnn module — runs on CPU (eager runner) | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ cpu ] | CPU_REUSE | REUSE/ADAPT tag not wired to a ttnn module — runs on CPU (eager runner) | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_layer.py::test_layer` |
| `mlp` | [ cpu ] | CPU_REUSE | REUSE/ADAPT tag not wired to a ttnn module — runs on CPU (eager runner) | `models/tt_transformers/demo/kolibri_1/tests/pcc/test_mlp.py::test_mlp` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_decoder_layer.py::test_decoder_layer -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_f_p8_linear.py::test_f_p8_linear -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_router.py::test_router -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_sparse_moe_block.py::test_sparse_moe_block -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_token_embed.py::test_token_embed -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_model.py::test_model -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_transformers/demo/kolibri_1/tests/pcc/test_mlp.py::test_mlp -svv
```

## Next steps

- **1 component(s) not graduated** — resume where it left off (already-graduated components are kept):
  - `python -m scripts.tt_hw_planner promote Aleph-Alpha/Kolibri-1 --box <BOX> --mesh <MESH>`
<!-- END bringup -->
