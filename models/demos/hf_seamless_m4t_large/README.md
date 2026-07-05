# facebook/hf-seamless-m4t-large — TT end-to-end pipeline

This package contains a runnable TTNN pipeline for
`facebook/hf-seamless-m4t-large`, exposing the 5 task heads defined by the
HF `AutoModel` registry.

## Layout

- `tt/pipeline.py` — the single shared chained forward pass. Imported by
  both `demo/` and `tests/e2e/` so a passing test guarantees a working demo.
- `demo/demo_<task>.py` — runnable per-task entrypoints
  (`python -m models.demos.hf_seamless_m4t_large.demo.demo_<task>`).
- `tests/e2e/test_e2e_<task>.py` — per-task e2e gates.
- `_stubs/` — the 42 graduated TTNN stubs (built by `tt_hw_planner bringup`).
- `_captured/` — HF golden tensors for per-component PCC.
- `e2e_plan.json` — the planner's task_heads plan (Command 1 output).

## Task heads

| head  | HF class                            | input             | output    | demo                | e2e test                    |
|-------|-------------------------------------|-------------------|-----------|---------------------|-----------------------------|
| T2TT  | `SeamlessM4TForTextToText`          | text `input_ids`  | text ids  | `demo/demo_t2tt.py` | `tests/e2e/test_e2e_t2tt.py` |
| S2TT  | `SeamlessM4TForSpeechToText`        | `input_features`  | text ids  | `demo/demo_s2tt.py` | `tests/e2e/test_e2e_s2tt.py` |
| T2ST  | `SeamlessM4TForTextToSpeech`        | text `input_ids`  | waveform  | `demo/demo_t2st.py` | `tests/e2e/test_e2e_t2st.py` |
| S2ST  | `SeamlessM4TForSpeechToSpeech`      | `input_features`  | waveform  | `demo/demo_s2st.py` | `tests/e2e/test_e2e_s2st.py` |
| BASE  | `SeamlessM4TModel`                  | text or audio     | text or waveform (dispatch) | `demo/demo_base.py` | `tests/e2e/test_e2e_base.py` |

## Running

Under this repo's TT env:

```
./python_env/bin/python -m pytest models/demos/hf_seamless_m4t_large/tests/e2e/test_e2e_t2tt.py -s
./python_env/bin/python -m pytest models/demos/hf_seamless_m4t_large/tests/e2e/test_e2e_s2tt.py -s
./python_env/bin/python -m pytest models/demos/hf_seamless_m4t_large/tests/e2e/test_e2e_t2st.py -s
./python_env/bin/python -m pytest models/demos/hf_seamless_m4t_large/tests/e2e/test_e2e_s2st.py -s
./python_env/bin/python -m pytest models/demos/hf_seamless_m4t_large/tests/e2e/test_e2e_base.py -s
```

Demo runs:

```
./python_env/bin/python -m models.demos.hf_seamless_m4t_large.demo.demo_t2tt --text "Hello" --tgt-lang fra
./python_env/bin/python -m models.demos.hf_seamless_m4t_large.demo.demo_s2tt --tgt-lang eng
./python_env/bin/python -m models.demos.hf_seamless_m4t_large.demo.demo_t2st --text "Hello" --tgt-lang eng
./python_env/bin/python -m models.demos.hf_seamless_m4t_large.demo.demo_s2st --tgt-lang eng
./python_env/bin/python -m models.demos.hf_seamless_m4t_large.demo.demo_base --generate-speech --tgt-lang eng
```

## Gate contract

- **Gate 1 — native TTNN**: every routed graduated stub runs `ttnn.*` primitives (not torch fallback).
- **Gate 2 — every graduated stub invoked**: 42 stubs, all touched by at least one head; each head's chain calls all its listed direct + probe stubs.
- **Gate 3 — final PCC ≥ 0.95**: TT output vs HF `model.generate()` capped to the same `max_new_tokens` cap (N=16 for text-out heads, N=8 for speech-out).

The pipeline **always** prints `e2e PCC=<value>` on its own line before the final assert, on both pass and fail runs.

## Trace + 2CQ contract

`tt/pipeline.py` exposes `PIPELINE_STAGES = ["encode", "prefill", "decode", "t2u_prefill", "t2u_decode", "vocode"]` and, for each stage, `<stage>_trace_setup`, `<stage>_trace_step`, `<stage>_write_inputs`. `Pipeline.trace_capture_selftest(device)` walks the list, captures one host-free step per stage in isolation, and releases the trace before the next stage.
