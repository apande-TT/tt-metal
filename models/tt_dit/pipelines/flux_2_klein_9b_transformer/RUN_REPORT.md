<!-- BEGIN bringup -->
# Bring-up run report — `/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer`

_Generated: 2026-08-27 19:04:56 UTC_

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
| 1 | `hf_eager universal (Image / diffusion)` | 40 | category 'Image' default (generic runner) |
| 2 | `Stable Diffusion 1.4` | 30 | category 'Image' default |
| 3 | `stable_diffusion_xl_base (auto-upstream)` | 30 | category 'Image' default |

## Placement summary

- **ON_DEVICE** (18): graduated, native ttnn, PCC verified
  - `ada_layer_norm_continuous`, `decoder_head`, `encoder_stack`, `flux2_attention`, `flux2_feed_forward`, `flux2_modulation`, `flux2_parallel_self_attention`, `flux2_pos_embed`, `flux2_single_transformer_block`, `flux2_swi_g_l_u`, `flux2_timestep_guidance_embeddings`, `flux2_transformer_block`, `layer`, `mlp`, `patch_embed`, `self_attention`, `timestep_embedding`, `timesteps`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `ada_layer_norm_continuous` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `flux2_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_attention.py::test_flux2_attention` |
| `flux2_feed_forward` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_feed_forward.py::test_flux2_feed_forward` |
| `flux2_modulation` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_modulation.py::test_flux2_modulation` |
| `flux2_parallel_self_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_parallel_self_attention.py::test_flux2_parallel_self_attention` |
| `flux2_pos_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_pos_embed.py::test_flux2_pos_embed` |
| `flux2_single_transformer_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_single_transformer_block.py::test_flux2_single_transformer_block` |
| `flux2_swi_g_l_u` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_swi_g_l_u.py::test_flux2_swi_g_l_u` |
| `flux2_timestep_guidance_embeddings` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_timestep_guidance_embeddings.py::test_flux2_timestep_guidance_embeddings` |
| `flux2_transformer_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_transformer_block.py::test_flux2_transformer_block` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_layer.py::test_layer` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_mlp.py::test_mlp` |
| `patch_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_patch_embed.py::test_patch_embed` |
| `self_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_self_attention.py::test_self_attention` |
| `timestep_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_timestep_embedding.py::test_timestep_embedding` |
| `timesteps` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_timesteps.py::test_timesteps` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_attention.py::test_flux2_attention -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_feed_forward.py::test_flux2_feed_forward -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_modulation.py::test_flux2_modulation -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_parallel_self_attention.py::test_flux2_parallel_self_attention -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_pos_embed.py::test_flux2_pos_embed -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_single_transformer_block.py::test_flux2_single_transformer_block -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_swi_g_l_u.py::test_flux2_swi_g_l_u -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_timestep_guidance_embeddings.py::test_flux2_timestep_guidance_embeddings -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_flux2_transformer_block.py::test_flux2_transformer_block -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_patch_embed.py::test_patch_embed -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_self_attention.py::test_self_attention -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_timestep_embedding.py::test_timestep_embedding -svv
python -m pytest models/tt_dit/pipelines/flux_2_klein_9b_transformer/tests/pcc/test_timesteps.py::test_timesteps -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /tmp/tt_hw_planner_components/flux_2_klein_9b_transformer`
<!-- END bringup -->
