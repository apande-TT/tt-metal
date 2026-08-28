<!-- BEGIN bringup -->
# Bring-up run report — `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`

_Generated: 2026-08-27 17:56:45 UTC_

_Topology: TP=8 x DP=1 (mesh 1x8, 8 chips) — run emit-e2e / optimize with `--mesh 1x8`._

## Outcome

**Converged** after 1 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `tt_transformers / simple_text_demo`
- **Closest template:** `models/tt_transformers/demo/simple_text_demo.py`
- **Target model_type:** `qwen3`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo` (selected) | 95 | LLM: Target is a Qwen3ForCausalLM decoder-only causal LM (the FLUX.2 Klein text encoder is a Qwen3 LLM). This backend's fingerprint is exactly decoder-only causal LM and tt_transformers is the generic-rout |
| 2 | `qwen3_vl (auto-upstream)` | 89 | LLM: Same Qwen3 lineage: model_type 'qwen3_vl' shares the qwen3 stem and its language-model trunk IS a Qwen3 decoder stack (same RMSNorm, q/k-norm attention, SwiGLU MLP, RoPE). Closest same-family backend  |
| 3 | `qwen25_vl (auto-upstream)` | 76 | LLM: Qwen lineage (qwen2_5_vl) one version back; its decoder LM trunk is near-identical to Qwen3 apart from q/k-norm, so attention/MLP/norm component implementations port with minimal change. |

## Placement summary

- **ON_DEVICE** (10): graduated, native ttnn, PCC verified
  - `attention`, `decoder_head`, `decoder_layer`, `encoder_stack`, `layer`, `m_l_p`, `mlp`, `r_m_s_norm`, `rotary_embedding`, `token_embed`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_attention.py::test_attention` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_decoder_layer.py::test_decoder_layer` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_layer.py::test_layer` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_mlp.py::test_mlp` |
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_rotary_embedding.py::test_rotary_embedding` |
| `token_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_token_embed.py::test_token_embed` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_decoder_layer.py::test_decoder_layer -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_rotary_embedding.py::test_rotary_embedding -svv
python -m pytest models/tt_transformers/demo/flux_2_klein_9b_text_encoder/tests/pcc/test_token_embed.py::test_token_embed -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
<!-- END bringup -->
