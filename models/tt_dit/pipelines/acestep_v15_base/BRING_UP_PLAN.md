# ACE-Step v1.5 — tt_dit pipeline bring-up plan

ACE-Step is a **flow-matching diffusion transformer (DiT)** for text-to-audio, not a
causal LM or generic TTS seq2seq model. Use this pipeline instead of
`models/demos/hf_eager/` for bring-up.

## Architecture mapping

| ACE-Step (HF) | tt-metal reuse | Scaffold file |
|---|---|---|
| Flow-matching denoising loop | `pipelines/flux1/pipeline_flux1.py`, `stable_diffusion_35_large/` | `pipeline_acestep.py` |
| CFG (2× DiT forward/step) | `pipelines/cfg.py` — `CFGCombiner` | `pipeline_acestep.py` |
| APG/ADG guidance | New host-side math | `guidance.py` (TODO) |
| Timestep / AdaLN (6× modulation) | `layers/embeddings.py` | DiT model (TODO) |
| DiT 24L GQA + sliding/full attn | `tt_transformers/tt/attention.py` | `transformer_acestep.py` (TODO) |
| Conv1d patchify (patch=2) | `layers/audio_ops.py` | DiT model (TODO) |
| Condition encoder stack | Standard transformer layers | `condition_encoder.py` (TODO) |
| ResidualFSQ codec | Host-side only | `fsq_codec.py` (TODO) |
| VAE + vocoder | `models/tt_dit/models/audio_vae/` | `audio_decode.py` (TODO) |

## Port order (bottom-up)

1. Condition encoders (lyric 8L, timbre 4L, attention pooler) — closest to REUSE/ADAPT
2. DiT block (AdaLN + conv1d patchify + alternating sliding/full attention)
3. Host sampling loop + CFG (+ APG/ADG)
4. FSQ tokenizer/detokenizer (host)
5. Audio VAE/vocoder tail

## v0 scope

Text → music only. Skip LM hints, cover/reference-audio modes, and `acestep_v15_base-5Hz-lm`.

## Planner routing

`scripts/tt_hw_planner/family_backends.py` registers this path for
`model_type=acestep_v15_base` / `ACE-Step/acestep_v15_base-v15-base`.
