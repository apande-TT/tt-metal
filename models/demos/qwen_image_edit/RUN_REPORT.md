<!-- BEGIN trace-gate -->
# Trace gate

verdict: **PASS**

trace engaged

graduated on-device: 0, ungraduated: 0

fresh capture: invalid E       23. _PyFunction_Vectorcall
E       25. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x2779e8) [0x5579aa7a69e8]
E       26. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x2c40c9) [0x5579aa7f30c9]
E       27. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x29259f) [0x5579aa7c159f]
E       28. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x2fd0b9) [0x5579aa82c0b9]
E       30. _PyFunction_Vectorcall
E       31. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x2fd0b9) [0x5579aa82c0b9]
E       33. _PyFunction_Vectorcall
E       35. _PyFunction_Vectorcall
E       37. _PyFunction_Vectorcall
E       38. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x273350) [0x5579aa7a2350]
E       40. _PyFunction_Vectorcall
E       41. _PyObject_Call_Prepend
E       42. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x3a2b46) [0x5579aa8d1b46]
E       43. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x2fd2a2) [0x5579aa82c2a2]
E       45. _PyFunction_Vectorcall
E       46. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x273350) [0x5579aa7a2350]
E       47. /home/ubuntu/apande/tt-metal/python_env/bin/python(+0x2fd0b9) [0x5579aa82c0b9]
E       49. _PyFunction_Vectorcall
E       51. _PyFunction_Vectorcall
E       53. _PyFunction_Vectorcall
E       55. _PyFunction_Vectorcall
E       57. _PyFunction_Vectorcall
E       59. _PyFunction_Vectorcall
E       61. _PyFunction_Vectorcall
<!-- END trace-gate -->

<!-- BEGIN bringup -->
# Bring-up run report — `Qwen/Qwen-Image-Edit`

_Generated: 2026-09-24 16:02:28 UTC_

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
python -m pytest qwen_image_edit/tests/e2e/test_e2e_image_edit.py -svv
python -m pytest qwen_image_edit/tests/e2e/test_image_edit_perf.py -svv
python -m pytest qwen_image_edit/demo/demo.py::test_demo -svv
python -m pytest qwen_image_edit/demo/demo_image_edit.py::test_demo -svv
```

## Next steps
<!-- END bringup -->

<!-- BEGIN emit-e2e -->
# E2E report — `Qwen/Qwen-Image-Edit`

_Generated: 2026-09-24 16:02:28 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: (no tracked components)
- operations: (no tracked components)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `image_edit` | n/a | `qwen_image_edit/demo/demo_image_edit.py` | `qwen_image_edit/tests/e2e/test_e2e_image_edit.py` | `qwen_image_edit/tests/e2e/test_image_edit_perf.py` |

## Reproduce

### image_edit
```bash
python qwen_image_edit/demo/demo_image_edit.py
pytest qwen_image_edit/tests/e2e/test_e2e_image_edit.py -svv
pytest qwen_image_edit/tests/e2e/test_image_edit_perf.py -svv
```
<!-- END emit-e2e -->
