<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 0, ungraduated: 0
<!-- END trace-gate -->

<!-- BEGIN bringup -->
# Bring-up run report — `mistralai/Voxtral-4B-TTS-2603`

_Generated: 2026-09-20 01:03:52 UTC_

## Outcome

**Converged** after bring-up.

## Placement summary

- **ON_DEVICE** (0): graduated, native ttnn, PCC verified
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
```

End-to-end / demo:
```bash
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_acoustic.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_hidden_states.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_text_generation.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_text_generation_perf.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_trace_and_host_ops.py -svv
python -m pytest voxtral_4b_tts_2603/demo/demo.py::test_demo -svv
python -m pytest voxtral_4b_tts_2603/demo/demo_acoustic.py::test_demo -svv
python -m pytest voxtral_4b_tts_2603/demo/demo_hidden_states.py::test_demo -svv
python -m pytest voxtral_4b_tts_2603/demo/demo_text_generation.py::test_demo -svv
```

## Next steps
<!-- END bringup -->

<!-- BEGIN emit-e2e -->
# E2E report — `mistralai/Voxtral-4B-TTS-2603`

_Generated: 2026-09-20 01:03:52 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 10/11 on device (90%), 1/11 on CPU (9%)
- Graduated (ON_DEVICE) : 5/11 (45%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=5  ADAPT-wired=3  NEW-native=2  NEW-partial-CPU=0
- on CPU    : NEW-fallback=0  REUSE/ADAPT-not-wired=1
- REUSE/ADAPT tagged but NOT wired to a ttnn module in this demo (runs on CPU via eager runner): encoder_stack
- operations: 10/11 on device (90%), 1/11 on CPU (9%)  (component-level estimate; run with --op-synth for op-level granularity)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `acoustic` | n/a | `voxtral_4b_tts_2603/demo/demo_acoustic.py` | `voxtral_4b_tts_2603/tests/e2e/test_e2e_acoustic.py` | (none) |
| `hidden_states` | n/a | `voxtral_4b_tts_2603/demo/demo_hidden_states.py` | `voxtral_4b_tts_2603/tests/e2e/test_e2e_hidden_states.py` | (none) |
| `text_generation` | n/a | `voxtral_4b_tts_2603/demo/demo_text_generation.py` | `voxtral_4b_tts_2603/tests/e2e/test_e2e_text_generation.py` | `voxtral_4b_tts_2603/tests/e2e/test_text_generation_perf.py` |

## Reproduce

### acoustic
```bash
python voxtral_4b_tts_2603/demo/demo_acoustic.py
pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_acoustic.py -svv
```

### hidden_states
```bash
python voxtral_4b_tts_2603/demo/demo_hidden_states.py
pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_hidden_states.py -svv
```

### text_generation
```bash
python voxtral_4b_tts_2603/demo/demo_text_generation.py
pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_text_generation.py -svv
pytest voxtral_4b_tts_2603/tests/e2e/test_text_generation_perf.py -svv
```
<!-- END emit-e2e -->
