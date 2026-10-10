<!-- BEGIN bringup -->
# Bring-up run report — `openbmb/MiniCPM5-2B`

_Generated: 2026-10-10 21:16:17 UTC_

_Topology: single-device (1 chip)._

## Outcome

**Converged** after bring-up.

## Backend & template match

- **Backend picked:** `tt_transformers / simple_text_demo`
- **Closest template:** `models/tt_transformers/demo/simple_text_demo.py`
- **Target model_type:** `llama`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo` (selected) | 40 | category 'LLM' default (generic runner) |
| 2 | `NemotronH (nemotron_h hybrid Mamba2/MoE)` | 30 | category 'LLM' default |
| 3 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |

## Placement summary

- **ON_DEVICE** (10): graduated, native ttnn, PCC verified
  - `attention`, `decoder_head`, `decoder_layer`, `encoder_stack`, `layer`, `m_l_p`, `mlp`, `r_m_s_norm`, `rotary_embedding`, `token_embed`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_attention.py::test_attention` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_decoder_layer.py::test_decoder_layer` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_layer.py::test_layer` |
| `m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_m_l_p.py::test_m_l_p` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_mlp.py::test_mlp` |
| `r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm` |
| `rotary_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_rotary_embedding.py::test_rotary_embedding` |
| `token_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `minicpm5_2b/tests/pcc/test_token_embed.py::test_token_embed` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest minicpm5_2b/tests/pcc/test_attention.py::test_attention -svv
python -m pytest minicpm5_2b/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest minicpm5_2b/tests/pcc/test_decoder_layer.py::test_decoder_layer -svv
python -m pytest minicpm5_2b/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest minicpm5_2b/tests/pcc/test_layer.py::test_layer -svv
python -m pytest minicpm5_2b/tests/pcc/test_m_l_p.py::test_m_l_p -svv
python -m pytest minicpm5_2b/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest minicpm5_2b/tests/pcc/test_r_m_s_norm.py::test_r_m_s_norm -svv
python -m pytest minicpm5_2b/tests/pcc/test_rotary_embedding.py::test_rotary_embedding -svv
python -m pytest minicpm5_2b/tests/pcc/test_token_embed.py::test_token_embed -svv
```

End-to-end / demo:
```bash
python -m pytest minicpm5_2b/tests/e2e/test_e2e_minicpm5_2b.py -svv
python -m pytest minicpm5_2b/tests/e2e/test_text_generation_perf.py -svv
python -m pytest minicpm5_2b/demo/demo.py::test_demo -svv
python -m pytest minicpm5_2b/demo/demo_text_generation.py::test_demo -svv
python -m pytest minicpm5_2b/demo/device.py::test_demo -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e openbmb/MiniCPM5-2B`
<!-- END bringup -->

<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 10, ungraduated: 0
<!-- END trace-gate -->

<!-- BEGIN emit-e2e -->
# E2E report — `openbmb/MiniCPM5-2B`

_Generated: 2026-10-10 21:16:17 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 10/10 on device (100%), 0/10 on CPU (0%)
- Graduated (ON_DEVICE) : 6/10 (60%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=4  ADAPT-wired=5  NEW-native=1  NEW-partial-CPU=0
- on CPU    : NEW-fallback=0  REUSE/ADAPT-not-wired=0
- operations: 10/10 on device (100%), 0/10 on CPU (0%)  (component-level estimate; run with --op-synth for op-level granularity)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `text_generation` | n/a | `minicpm5_2b/demo/demo_text_generation.py` | (none) | `minicpm5_2b/tests/e2e/test_text_generation_perf.py` |

## Reproduce

### text_generation
```bash
python minicpm5_2b/demo/demo_text_generation.py
pytest minicpm5_2b/tests/e2e/test_text_generation_perf.py -svv
```
<!-- END emit-e2e -->
