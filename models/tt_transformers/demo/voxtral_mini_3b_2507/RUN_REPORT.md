<!-- BEGIN bringup -->
# Bring-up run report — `mistralai/Voxtral-Mini-3B-2507`

_Generated: 2026-08-05 19:03:21 UTC_

_Topology: single-device (1 chip)._

## Outcome

**Converged** after bring-up.

## Backend & template match

- **Backend picked:** `tt_transformers / simple_text_demo`
- **Closest template:** `models/tt_transformers/demo/simple_text_demo.py`
- **Target model_type:** `voxtral`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_transformers / simple_text_demo` (selected) | 40 | category 'LLM' default (generic runner) |
| 2 | `falcon7b_common (auto-upstream)` | 30 | category 'LLM' default |
| 3 | `gemma4 (auto-upstream)` | 30 | category 'LLM' default |

## Placement summary

- **ON_DEVICE** (15): graduated, native ttnn, PCC verified
  - `attention`, `avg_pool1d`, `decoder_head`, `encoder_stack`, `layer`, `llama_decoder_layer`, `llama_m_l_p`, `llama_model`, `llama_r_m_s_norm`, `mlp`, `token_embed`, `voxtral_attention`, `voxtral_encoder`, `voxtral_encoder_layer`, `voxtral_multi_modal_projector`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (2): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device
  - `llama_attention`, `llama_rotary_embedding`

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_attention.py::test_attention` |
| `avg_pool1d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_avg_pool1d.py::test_avg_pool1d` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_layer.py::test_layer` |
| `llama_decoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_llama_decoder_layer.py::test_llama_decoder_layer` |
| `llama_m_l_p` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_llama_m_l_p.py::test_llama_m_l_p` |
| `llama_model` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_llama_model.py::test_llama_model` |
| `llama_r_m_s_norm` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_llama_r_m_s_norm.py::test_llama_r_m_s_norm` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_mlp.py::test_mlp` |
| `token_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_token_embed.py::test_token_embed` |
| `voxtral_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_voxtral_attention.py::test_voxtral_attention` |
| `voxtral_encoder` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_voxtral_encoder.py::test_voxtral_encoder` |
| `voxtral_encoder_layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_voxtral_encoder_layer.py::test_voxtral_encoder_layer` |
| `voxtral_multi_modal_projector` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `voxtral_mini_3b_2507/tests/pcc/test_voxtral_multi_modal_projector.py::test_voxtral_multi_modal_projector` |
| `llama_attention` | [ cpu ] | CPU_REUSE | REUSE/ADAPT tag not wired to a ttnn module — runs on CPU (eager runner) | `voxtral_mini_3b_2507/tests/pcc/test_llama_attention.py::test_llama_attention` |
| `llama_rotary_embedding` | [ cpu ] | CPU_REUSE | REUSE/ADAPT tag not wired to a ttnn module — runs on CPU (eager runner) | `voxtral_mini_3b_2507/tests/pcc/test_llama_rotary_embedding.py::test_llama_rotary_embedding` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_attention.py::test_attention -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_avg_pool1d.py::test_avg_pool1d -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_layer.py::test_layer -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_llama_decoder_layer.py::test_llama_decoder_layer -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_llama_m_l_p.py::test_llama_m_l_p -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_llama_model.py::test_llama_model -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_llama_r_m_s_norm.py::test_llama_r_m_s_norm -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_token_embed.py::test_token_embed -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_voxtral_attention.py::test_voxtral_attention -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_voxtral_encoder.py::test_voxtral_encoder -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_voxtral_encoder_layer.py::test_voxtral_encoder_layer -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_voxtral_multi_modal_projector.py::test_voxtral_multi_modal_projector -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_llama_attention.py::test_llama_attention -svv
python -m pytest voxtral_mini_3b_2507/tests/pcc/test_llama_rotary_embedding.py::test_llama_rotary_embedding -svv
```

End-to-end / demo:
```bash
python -m pytest voxtral_mini_3b_2507/tests/e2e/test_e2e_pipeline.py -svv
python -m pytest voxtral_mini_3b_2507/tests/e2e/test_trace_contract.py -svv
python -m pytest voxtral_mini_3b_2507/demo/demo.py::test_demo -svv
python -m pytest voxtral_mini_3b_2507/demo/demo_audio_chat.py::test_demo -svv
python -m pytest voxtral_mini_3b_2507/demo/demo_transcription.py::test_demo -svv
```

## Next steps

- **All NEW components graduated** — 2 REUSE/ADAPT component(s) run on CPU (not wired to a ttnn module). Wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e mistralai/Voxtral-Mini-3B-2507`
<!-- END bringup -->

<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 15, ungraduated: 2

fresh capture: no perf test to capture
<!-- END trace-gate -->

<!-- BEGIN emit-e2e -->
# E2E report — `mistralai/Voxtral-Mini-3B-2507`

_Generated: 2026-08-05 19:03:21 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 15/17 on device (88%), 2/17 on CPU (11%)
- Graduated (ON_DEVICE) : 10/17 (58%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=5  ADAPT-wired=4  NEW-native=6  NEW-partial-CPU=0
- on CPU    : NEW-fallback=0  REUSE/ADAPT-not-wired=2
- REUSE/ADAPT tagged but NOT wired to a ttnn module in this demo (runs on CPU via eager runner): llama_attention, llama_rotary_embedding
- operations: 628/630 on device (99%), 2/630 on CPU (0%)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `audio_chat` | n/a | `voxtral_mini_3b_2507/demo/demo_audio_chat.py` | (none) | (none) |
| `transcription` | n/a | `voxtral_mini_3b_2507/demo/demo_transcription.py` | (none) | (none) |

## Reproduce

### audio_chat
```bash
python voxtral_mini_3b_2507/demo/demo_audio_chat.py
```

### transcription
```bash
python voxtral_mini_3b_2507/demo/demo_transcription.py
```
<!-- END emit-e2e -->
