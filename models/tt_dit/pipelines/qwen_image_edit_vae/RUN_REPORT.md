<!-- BEGIN bringup -->
# Bring-up run report — `/tmp/tt_hw_planner_components/qwen_image_edit_vae`

_Generated: 2026-09-25 03:55:11 UTC_

_Topology: TP=32 x DP=1 (mesh 1x32, 32 chips) — run emit-e2e / optimize with `--mesh 1x32`._

## Outcome

**Converged** after 3 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `tt_dit/qwenimage (auto-upstream)`
- **Closest template:** `models/tt_dit/pipelines/qwenimage`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `tt_dit/qwenimage (auto-upstream)` (selected) | 95 | LLM: Same model family. The Qwen-Image-Edit VAE is AutoencoderKLQwenImage, the VAE that the tt_dit/qwenimage pipeline already uses. The encoder and decoder can be reused directly. |
| 2 | `tt_dit/wan (auto-upstream)` | 89 | LLM: The Qwen-Image VAE is derived from the Wan 2.1 VAE: a causal 3D-conv autoencoder with RMS norm, residual blocks and mid-block attention. The tt_dit Wan VAE kernels map to it block for block. |
| 3 | `tt_dit/mochi (auto-upstream)` | 54 | LLM: Its video VAE decoder uses causal 3D convolutions and conv-parallel tiling patterns that apply to a temporal-causal VAE. |

## Placement summary

- **ON_DEVICE** (17): graduated, native ttnn, PCC verified
  - `decoder_head`, `encoder_stack`, `layer`, `mlp`, `patch_embed`, `qwen_image_attention_block`, `qwen_image_causal_conv3d`, `qwen_image_decoder3d`, `qwen_image_encoder3d`, `qwen_image_mid_block`, `qwen_image_r_m_s`, `qwen_image_resample`, `qwen_image_residual_block`, `qwen_image_up_block`, `qwen_image_upsample`, `self_attention`, `zero_pad2d`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_layer.py::test_layer` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_mlp.py::test_mlp` |
| `patch_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_patch_embed.py::test_patch_embed` |
| `qwen_image_attention_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_attention_block.py::test_qwen_image_attention_block` |
| `qwen_image_causal_conv3d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_causal_conv3d.py::test_qwen_image_causal_conv3d` |
| `qwen_image_decoder3d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_decoder3d.py::test_qwen_image_decoder3d` |
| `qwen_image_encoder3d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_encoder3d.py::test_qwen_image_encoder3d` |
| `qwen_image_mid_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_mid_block.py::test_qwen_image_mid_block` |
| `qwen_image_r_m_s` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_r_m_s.py::test_qwen_image_r_m_s` |
| `qwen_image_resample` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_resample.py::test_qwen_image_resample` |
| `qwen_image_residual_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_residual_block.py::test_qwen_image_residual_block` |
| `qwen_image_up_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_up_block.py::test_qwen_image_up_block` |
| `qwen_image_upsample` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_upsample.py::test_qwen_image_upsample` |
| `self_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_self_attention.py::test_self_attention` |
| `zero_pad2d` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_zero_pad2d.py::test_zero_pad2d` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_patch_embed.py::test_patch_embed -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_attention_block.py::test_qwen_image_attention_block -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_causal_conv3d.py::test_qwen_image_causal_conv3d -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_decoder3d.py::test_qwen_image_decoder3d -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_encoder3d.py::test_qwen_image_encoder3d -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_mid_block.py::test_qwen_image_mid_block -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_r_m_s.py::test_qwen_image_r_m_s -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_resample.py::test_qwen_image_resample -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_residual_block.py::test_qwen_image_residual_block -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_up_block.py::test_qwen_image_up_block -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_qwen_image_upsample.py::test_qwen_image_upsample -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_self_attention.py::test_self_attention -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_vae/tests/pcc/test_zero_pad2d.py::test_zero_pad2d -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /tmp/tt_hw_planner_components/qwen_image_edit_vae`
<!-- END bringup -->
