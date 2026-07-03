# DEMO entry: `demo.py` for `ACE-Step/acestep-v15-base`

## What you are doing

Write the top-level entry script that loads a real input, runs the model on TT hardware end-to-end, and prints / saves the output. Use the TTNN modules the user already has under `models/demos/hf_eager/acestep_v15_base/_stubs/` (NEW components) and `models/demos/hf_eager/acestep_v15_base/tt/` (ADAPT components). Do NOT write any new module bodies in this file — only the pipeline.

Category guidance (NLP): text — tokenize, forward, sample / argmax.

## Components available to wire

NEW (TTNN ports the user generated separately):
  - `ace_step_di_t_model` -> import from `_stubs/ace_step_di_t_model.py`
  - `ace_step_di_t_layer` -> import from `_stubs/ace_step_di_t_layer.py`
  - `ace_step_encoder_layer` -> import from `_stubs/ace_step_encoder_layer.py`
  - `ace_step_condition_encoder` -> import from `_stubs/ace_step_condition_encoder.py`
  - `ace_step_lyric_encoder` -> import from `_stubs/ace_step_lyric_encoder.py`
  - `ace_step_timbre_encoder` -> import from `_stubs/ace_step_timbre_encoder.py`
  - `ace_step_audio_tokenizer` -> import from `_stubs/ace_step_audio_tokenizer.py`
  - `audio_token_detokenizer` -> import from `_stubs/audio_token_detokenizer.py`
  - `attention_pooler` -> import from `_stubs/attention_pooler.py`
  - `timestep_embedding` -> import from `_stubs/timestep_embedding.py`
  - `lambda` -> import from `_stubs/lambda.py`
  - `residual_f_s_q` -> import from `_stubs/residual_f_s_q.py`
  - `f_s_q` -> import from `_stubs/f_s_q.py`

ADAPT (sibling-cloned files, shape-adjusted):
  (no ADAPT components)

## Model config snapshot

```json
{}
```

## Required structure of `demo.py`

The file must contain, in this order:
  1. The 3-line SPDX header (rule 1 of project conventions).
  2. Imports: argparse, pytest, torch, ttnn, the HF processor /
     tokenizer appropriate for the category, and the TTNN modules
     from `models/demos/hf_eager/acestep_v15_base/_stubs/` and `models/demos/hf_eager/acestep_v15_base/tt/`.
  3. A `test_demo(device_params, device)` pytest entry decorated
     with `@pytest.mark.parametrize('device_params', [{}], indirect=True)`
     that performs: load real input -> preprocess -> instantiate
     TTNN modules with weights from the HF state_dict -> forward
     on device -> postprocess -> assert sanity (output shape /
     value range).
  4. An `if __name__ == '__main__':` block exposing the same
     pipeline as a standalone CLI (`--input <path>`).

## No-comments rule (mandatory)

Do NOT add docstrings, inline `# ...` comments, or section
banners. The only comments allowed are the 3-line SPDX header
at the top. Use clearly named helpers (`_preprocess`, `_build_modules`,
`_run_inference`, `_postprocess`) so the pipeline reads itself.

## Output

Write the complete file to disk at:

  `models/demos/hf_eager/acestep_v15_base/_synth_responses/demo.py`

Make it runnable: `pytest models/demos/hf_eager/acestep_v15_base/demo.py -svv` should open the device, run the model end-to-end on a small input, and return a sane shape. Use real HF assets (processor / tokenizer) so the user can swap in their own input without modifying the file. After it lands, the user will run `bringup ACE-Step/acestep-v15-base --apply-all-responses` to install it.
