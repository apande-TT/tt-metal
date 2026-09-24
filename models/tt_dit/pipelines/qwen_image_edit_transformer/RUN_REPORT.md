<!-- BEGIN bringup -->
# Bring-up run report — `/tmp/tt_hw_planner_components/qwen_image_edit_transformer`

_Generated: 2026-09-23 10:45:03 UTC_

_Topology: TP=8 x DP=1 (mesh 1x8, 8 chips) — run emit-e2e / optimize with `--mesh 1x8`._

## Outcome

**Converged** after 1 iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `tt_dit/qwenimage (auto-upstream)`
- **Closest template:** `models/tt_dit/pipelines/qwenimage`

## Sibling candidates (ranked)

Top backends by match score — the demo can compose per-component reuse across these, not only rank 1.

| Rank | Backend | Score | Match reason |
|---|---|---|---|
| 1 | `hf_eager universal (Image / diffusion)` | 40 | category 'Image' default (generic runner) |
| 2 | `Stable Diffusion 1.4` | 30 | category 'Image' default |
| 3 | `stable_diffusion_xl_base (auto-upstream)` | 30 | category 'Image' default |

## Placement summary

- **ON_DEVICE** (14): graduated, native ttnn, PCC verified
  - `ada_layer_norm_continuous`, `attention`, `decoder_head`, `encoder_stack`, `feed_forward`, `layer`, `mlp`, `patch_embed`, `qwen_embed_rope`, `qwen_image_transformer_block`, `qwen_timestep_proj_embeddings`, `self_attention`, `timestep_embedding`, `timesteps`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|
| `ada_layer_norm_continuous` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous` |
| `attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_attention.py::test_attention` |
| `decoder_head` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_decoder_head.py::test_decoder_head` |
| `encoder_stack` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_encoder_stack.py::test_encoder_stack` |
| `feed_forward` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_feed_forward.py::test_feed_forward` |
| `layer` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_layer.py::test_layer` |
| `mlp` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_mlp.py::test_mlp` |
| `patch_embed` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_patch_embed.py::test_patch_embed` |
| `qwen_embed_rope` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_qwen_embed_rope.py::test_qwen_embed_rope` |
| `qwen_image_transformer_block` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_qwen_image_transformer_block.py::test_qwen_image_transformer_block` |
| `qwen_timestep_proj_embeddings` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_qwen_timestep_proj_embeddings.py::test_qwen_timestep_proj_embeddings` |
| `self_attention` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_self_attention.py::test_self_attention` |
| `timestep_embedding` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_timestep_embedding.py::test_timestep_embedding` |
| `timesteps` | [ ok ] | ON_DEVICE | graduated — native ttnn, PCC-verified | `models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_timesteps.py::test_timesteps` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_attention.py::test_attention -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_decoder_head.py::test_decoder_head -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_encoder_stack.py::test_encoder_stack -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_feed_forward.py::test_feed_forward -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_layer.py::test_layer -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_mlp.py::test_mlp -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_patch_embed.py::test_patch_embed -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_qwen_embed_rope.py::test_qwen_embed_rope -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_qwen_image_transformer_block.py::test_qwen_image_transformer_block -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_qwen_timestep_proj_embeddings.py::test_qwen_timestep_proj_embeddings -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_self_attention.py::test_self_attention -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_timestep_embedding.py::test_timestep_embedding -svv
python -m pytest models/tt_dit/pipelines/qwen_image_edit_transformer/tests/pcc/test_timesteps.py::test_timesteps -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e /tmp/tt_hw_planner_components/qwen_image_edit_transformer`
<!-- END bringup -->
