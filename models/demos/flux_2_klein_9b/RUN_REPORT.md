<!-- BEGIN trace-gate -->
# Trace gate

verdict: **EAGER_WAIVED**

trace not engaged; eager permitted because ungraduated module(s) present: ?

graduated on-device: 0, ungraduated: 0

fresh capture: no perf test to capture
<!-- END trace-gate -->

<!-- BEGIN bringup -->
# Bring-up run report — `black-forest-labs/FLUX.2-klein-9B`

_Generated: 2026-08-28 14:05:23 UTC_

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
python -m pytest flux_2_klein_9b/tests/e2e/test_e2e_batch_vae.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_e2e_pipeline.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_gates.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_latent_plumbing.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_layout_parity.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_stage_text_encoder.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_stage_transformer.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_stage_vae.py -svv
python -m pytest flux_2_klein_9b/tests/e2e/test_trace_contract.py -svv
python -m pytest flux_2_klein_9b/demo/demo.py::test_demo -svv
python -m pytest flux_2_klein_9b/demo/demo_image_edit.py::test_demo -svv
python -m pytest flux_2_klein_9b/demo/demo_text_generation.py::test_demo -svv
python -m pytest flux_2_klein_9b/demo/demo_text_to_image.py::test_demo -svv
python -m pytest flux_2_klein_9b/demo/demo_vae_roundtrip.py::test_demo -svv
```

## Next steps
<!-- END bringup -->

<!-- BEGIN emit-e2e -->
# E2E report — `black-forest-labs/FLUX.2-klein-9B`

_Generated: 2026-08-28 14:05:23 UTC_

**Verdict: PASS**

## Pipeline placement (on-device vs CPU fallback)

- components: (no tracked components)
- operations: (no tracked components)
- CPU-fallback modules: (none — fully on device)

## Per task / demo

| task | e2e PCC | demo (real input→output) | e2e PCC test | trace perf test |
|---|---|---|---|---|
| `image_edit` | n/a | `flux_2_klein_9b/demo/demo_image_edit.py` | (none) | (none) |
| `text_generation` | n/a | `flux_2_klein_9b/demo/demo_text_generation.py` | (none) | (none) |
| `text_to_image` | n/a | `flux_2_klein_9b/demo/demo_text_to_image.py` | (none) | (none) |
| `vae_roundtrip` | n/a | `flux_2_klein_9b/demo/demo_vae_roundtrip.py` | (none) | (none) |

## Reproduce

### image_edit
```bash
python flux_2_klein_9b/demo/demo_image_edit.py
```

### text_generation
```bash
python flux_2_klein_9b/demo/demo_text_generation.py
```

### text_to_image
```bash
python flux_2_klein_9b/demo/demo_text_to_image.py
```

### vae_roundtrip
```bash
python flux_2_klein_9b/demo/demo_vae_roundtrip.py
```
<!-- END emit-e2e -->
