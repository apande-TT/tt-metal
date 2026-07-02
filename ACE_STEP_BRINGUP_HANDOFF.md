# ACE-Step v1.5 → TTNN bring-up — handoff (2026-07-02)

Branch: `feature/tt-hw-planner`. This file is **uncommitted** scratch — commit, move, or delete as you like.

## State: DONE and committed
- `f36ff875dd` — added single-chip **`p150`** box (Blackhole, 1×1, 32 GB) to `scripts/tt_hw_planner/hardware.py`.
- `019e7cbe8d` — env-compat gate **respects the `transformers==5.10.2` pin** (no more downgrade thrashing) + **mesh-aware** grid checks (skipped for canonical single-chip 1×1).
- `6f9690e23b` — gate recognizes the existing `common.py` `_chat_template_ids` normalization (killed a stale-marker false positive).
- Tests: `scripts/tt_hw_planner/tests/test_env_gate_pin_mesh.py` (10, green). 5 pre-existing invariant failures are **red on `main` too** (see Gaps) — not regressions.

Verified: `auto-up ACE-Step/acestep-v15-base --box p150 --mesh 1,1` runs clean pre-flight → static analysis → scaffold → prepare (exit 0). **p150 hardware confirmed working** (opens as Blackhole, 1 chip).

## Env notes
- `transformers` pinned at **5.10.2** (repo pin, restored after the tool had downgraded it to 4.49.0).
- `vector_quantize_pytorch` was pip-installed (ACE-Step VAE dep).
- Untouched: pre-existing uncommitted `tt_metal/python_env/requirements-dev.txt` change (fiftyone/plotly — unrelated).

## KEY INSIGHT (from capability survey): don't chase `hf_eager` — reuse `tt_dit`
The tool routed ACE-Step to the generic `hf_eager` text backend (which doesn't exist) because it
didn't recognize it as a **diffusion transformer**. That was a red herring. tt-metal already has a
production **`models/tt_dit/`** subsystem that supplies ~most of what ACE-Step needs. ACE-Step should
be a **purpose-built pipeline like Flux1 / SD3.5**, NOT the generic eager harness.

### Capability map (surveyed 2026-07-02)
| ACE-Step block | In tt-metal? | Where |
|---|---|---|
| DiT + host-driven denoising loop + CFG | ✅ production | `models/tt_dit/pipelines/{flux1,stable_diffusion_35_large}/` (FlowMatchEuler scheduler; ACE-Step is flow-matching too) |
| Timestep/sinusoidal embed, AdaLN modulation, CFG combine | ✅ | `models/tt_dit/layers/embeddings.py`, `pipelines/cfg.py` |
| Conv1d / ConvTranspose1d (patchify + vocoder) | ✅ | `ttnn.conv1d` (`ttnn/cpp/.../conv/conv1d/`) + `models/tt_dit/layers/audio_ops.py` (Conv1dViaConv3d; ~0.9 dB loss, ok) |
| GQA + per-layer sliding(128)/full attention, RoPE θ1e6, RMSNorm, SiLU | ✅ production | `models/tt_transformers/tt/attention.py` (layer_types), `rope.py`; norms in `tt_dit/layers/normalization.py` |
| Audio VAE / DCAE / vocoder | ✅ (LTX-2 live) | `models/tt_dit/models/audio_vae/{vocoder_ltx.py,audio_decoder_ltx.py}` (BigVGAN-v2, fp32) |
| **FSQ / ResidualFSQ codebook** | ❌ | none — do **host-side** (torch codebook lookup, ship indices to device). Cheap, fine on host. |
| Text encoder (Qwen3-Embedding-0.6B) | ✅ (std transformer) | `models/tt_transformers/` |
| APG/ADG adaptive guidance (momentum projection) | ❌ (custom) | small host-side vector math — port from `modeling_acestep_v15_base.py` `apg_guidance.py` |

Net-new ACE-Step work is the *wiring*, not the primitives: the specific DiT block (6× AdaLN
modulation + conv1d patchify), the flow-matching ODE/SDE loop with APG/ADG (host), the condition
encoders (lyric 8L / timbre 4L / pooler — standard transformer layers), host-side FSQ decode, and
gluing text-enc → condition → DiT loop → detokenizer → VAE → vocoder.

### Pre-existing checkout gaps (not blockers for the tt_dit path)
- `models/demos/hf_eager/demo.py` absent — only relevant if you go the (wrong) generic-backend route.
- `_dispatch_safe_grid` missing in `models/tt_transformers/tt/model_config.py` (2 red tests; non-canonical meshes only).

## ACE-Step v1.5 architecture (confirmed from cached modeling code)
It is a **multi-stage, host-orchestrated music-generation pipeline**, not a single forward pass:
1. Text encoder (Qwen3-Embedding-0.6B, in umbrella `Ace-Step1.5`).
2. Condition stack: `AceStepLyricEncoder` (8L), `AceStepTimbreEncoder` (4L), `AttentionPooler` (2L), `AceStepConditionEncoder`.
3. (Optional) `acestep-5Hz-lm` autoregressive LM (0.6B/1.7B/4B) → 25 Hz audio-token hints.
4. **Core `AceStepDiTModel`**: 24-layer Diffusion Transformer — hidden 2048, GQA 16/8, **alternating sliding(128)/full attention**, RoPE θ=1e6, RMSNorm, SiLU MLP, `TimestepEmbedding` (6× AdaLN modulation), `Conv1d`/`ConvTranspose1d` patchify (patch_size 2, in_ch 192).
5. **Sampling**: 30-step flow-matching **ODE/SDE denoising loop** (host `for` loop), DiT run twice/step for **CFG**, **APG/ADG** adaptive guidance (momentum-buffer projection).
6. Audio codec: `ResidualFSQ` (levels [8,8,8,5,5,5]) → DCAE/VAE decode → vocoder → waveform.

Model file: `~/.cache/huggingface/hub/models--ACE-Step--acestep-v15-base/snapshots/*/modeling_acestep_v15_base.py` (sampling loop ~L1790–1945).

## Suggested resume order (tomorrow)
0. Capability survey — **DONE** (results in the table above). Path decided: build a `tt_dit`-style pipeline.
1. **De-risk the p150 on-device path** with a model that already has a verified backend (independent of ACE-Step):
   `python -m scripts.tt_hw_planner up <small-Llama-or-Qwen> --box p150 --mesh 1,1 --execute`
   Confirms device init + kernel compile + PCC flow work on p150 before investing in ACE-Step.
2. **Study the reference pipeline**: read `models/tt_dit/pipelines/flux1/` (or `stable_diffusion_35_large/`) end-to-end — that's the template (host scheduler loop + on-device DiT forward + CFG). ACE-Step's flow-matching loop maps onto FlowMatchEuler cleanly.
3. **Stand up `models/tt_dit/pipelines/acestep/`** and port bottom-up, PCC-verifying each stage vs the torch reference in `modeling_acestep_v15_base.py`:
   a. DiT block — reuse tt_transformers attention (set `layer_types` to the alternating sliding/full pattern, sliding_window=128) + RMSNorm/SiLU MLP/RoPE; **new: 6× AdaLN timestep modulation** (adapt `tt_dit/layers/embeddings.py`) + **conv1d patchify/unpatchify** (`tt_dit/layers/audio_ops.py`).
   b. Condition encoders (lyric 8L, timbre 4L, AttentionPooler 2L) — standard transformer layers.
   c. Text encoder — Qwen3-Embedding-0.6B via `models/tt_transformers/`.
   d. Host-side sampling loop (30-step ODE/SDE) + **APG/ADG guidance** (port `apg_guidance.py` — small host vector math).
   e. Detokenizer + **FSQ decode on host** → **DCAE/VAE decode + vocoder** (reuse `tt_dit/models/audio_vae/`).
4. Decide scope: start with **base text→music** (skip the optional `acestep-5Hz-lm` hint model and cover/reference-audio modes) to get a first end-to-end waveform, then add the LM-hints and cover paths.

## Handy re-entry commands
```
git -C /local/ttuser/dvartanians/ace/tt-metal log --oneline -4
python -m pytest scripts/tt_hw_planner/tests/test_env_gate_pin_mesh.py -q
python -m scripts.tt_hw_planner auto-up ACE-Step/acestep-v15-base --box p150 --mesh 1,1   # reaches prepare (exit 0)
```
Full logs from this session: `generated/acestep-v15-base_p150_run4.log` (clean scaffold), `..._execute.log` (device opened; hf_eager demo missing).
