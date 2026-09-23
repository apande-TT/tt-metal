<!-- BEGIN trace-gate -->
# Trace gate

verdict: **EAGER_WAIVED**

trace not engaged; eager permitted because ungraduated module(s) present: flow_matching_audio_transformer

graduated on-device: 30, ungraduated: 1

fresh capture: no perf test to capture
<!-- END trace-gate -->

<!-- BEGIN bringup -->
# Bring-up run report — `mistralai/Voxtral-4B-TTS-2603`

_Generated: 2026-09-22 20:39:41 UTC_

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
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_component_pcc.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_text_continuation.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_text_to_speech.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_gates.py -svv
python -m pytest voxtral_4b_tts_2603/tests/e2e/test_trace_and_host_ops.py -svv
python -m pytest voxtral_4b_tts_2603/demo/demo.py::test_demo -svv
python -m pytest voxtral_4b_tts_2603/demo/demo_text_continuation.py::test_demo -svv
python -m pytest voxtral_4b_tts_2603/demo/demo_text_to_speech.py::test_demo -svv
```

## Next steps
<!-- END bringup -->

<!-- BEGIN emit-e2e -->
# E2E report — `mistralai/Voxtral-4B-TTS-2603`

_Generated: 2026-09-22 20:39:41 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: 30/31 on device (96%), 1/31 on CPU (3%)
- Graduated (ON_DEVICE) : 22/31 (70%) actually graduated (native stub, PCC-verified)
- on device : REUSE-wired=8  ADAPT-wired=4  NEW-native=18  NEW-partial-CPU=0
- on CPU    : NEW-fallback=1  REUSE/ADAPT-not-wired=0
- operations: 30/31 on device (96%), 1/31 on CPU (3%)  (component-level estimate; run with --op-synth for op-level granularity)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `text_continuation` | n/a | `voxtral_4b_tts_2603/demo/demo_text_continuation.py` | `voxtral_4b_tts_2603/tests/e2e/test_e2e_text_continuation.py` | (none) |
| `text_to_speech` | n/a | `voxtral_4b_tts_2603/demo/demo_text_to_speech.py` | `voxtral_4b_tts_2603/tests/e2e/test_e2e_text_to_speech.py` | (none) |

## Reproduce

### text_continuation
```bash
python voxtral_4b_tts_2603/demo/demo_text_continuation.py
pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_text_continuation.py -svv
```

### text_to_speech
```bash
python voxtral_4b_tts_2603/demo/demo_text_to_speech.py
pytest voxtral_4b_tts_2603/tests/e2e/test_e2e_text_to_speech.py -svv
```
<!-- END emit-e2e -->
