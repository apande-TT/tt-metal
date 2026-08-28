<!-- BEGIN bringup -->
# Bring-up run report — `/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`

_Generated: 2026-08-27 20:04:07 UTC_

_Topology: TP=8 x DP=1 (mesh 1x8, 8 chips) — run emit-e2e / optimize with `--mesh 1x8`._

## Outcome

**Converged** after 1 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `tt_dit/flux2 (auto-upstream)`
- **Closest template:** `models/tt_dit/pipelines/flux2`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_dit/flux2 (auto-upstream)` (selected) | 97 | LLM: Same model: target is the FLUX.2 Klein 9B VAE (AutoencoderKLFlux2), a sub-component of the flux2 pipeline; this backend is the exact flux2 lineage and hosts the tt_dit VAE/encoder-decoder plumbing to  |
| 2 | `tt_dit/flux1 (auto-upstream)` | 88 | LLM: Same FLUX lineage (flux1 -> flux2); its autoencoder/VAE encoder-decoder ResNet+attention stack and tt_dit block reuse are the closest structural precedent for the FLUX.2 VAE. |
| 3 | `Stable Diffusion 1.4` | 76 | LLM: Only backend whose fingerprint explicitly includes a VAE (diffusion UNet+VAE): its AutoencoderKL encoder/decoder — ResnetBlock2D, Downsample/Upsample2D, GroupNorm, mid-block self-attention — is struct |

## Placement summary

- **ON_DEVICE** (15): graduated, native ttnn, PCC verified
  - `attention`, `decoder`, `decoder_head`, `down_encoder_block2_d`, `downsample2_d`, `encoder`, `encoder_stack`, `layer`, `mlp`, `patch_embed`, `resnet_block2_d`, `self_attention`, `u_net_mid_block2_d`, `up_decoder_block2_d`, `upsample2_d`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_attention.py::test_attention` |
| `decoder` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_decoder.py::test_decoder` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `down_encoder_block2_d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_down_encoder_block2_d.py::test_down_encoder_block2_d` |
| `downsample2_d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_downsample2_d.py::test_downsample2_d` |
| `encoder` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_encoder.py::test_encoder` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_layer.py::test_layer` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_mlp.py::test_mlp` |
| `patch_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_patch_embed.py::test_patch_embed` |
| `resnet_block2_d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_resnet_block2_d.py::test_resnet_block2_d` |
| `self_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_self_attention.py::test_self_attention` |
| `u_net_mid_block2_d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_u_net_mid_block2_d.py::test_u_net_mid_block2_d` |
| `up_decoder_block2_d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_up_decoder_block2_d.py::test_up_decoder_block2_d` |
| `upsample2_d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_upsample2_d.py::test_upsample2_d` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_decoder.py::test_decoder -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_down_encoder_block2_d.py::test_down_encoder_block2_d -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_downsample2_d.py::test_downsample2_d -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_encoder.py::test_encoder -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_patch_embed.py::test_patch_embed -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_resnet_block2_d.py::test_resnet_block2_d -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_self_attention.py::test_self_attention -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_u_net_mid_block2_d.py::test_u_net_mid_block2_d -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_up_decoder_block2_d.py::test_up_decoder_block2_d -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_upsample2_d.py::test_upsample2_d -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /tmp/tt_hw_planner_components/flux_2_klein_9b_vae`
<!-- END bringup -->
