"""tt_hw_planner: native TTNN port of the Qwen-Image VAE mid block (`encoder.mid_block`).

Component: qwen_image_mid_block  (torch reference: diffusers QwenImageMidBlock)

QwenImageMidBlock (residual -> [attention -> residual] x num_layers) is architecturally identical
to the Wan2.1 VAE mid block, so this reuses tt_dit's native `WanMidBlock`.

Tensor-parallel scheme (TP over the 1xN mesh): same as encoder_stack, which contains this block --
the parallel axis is SPATIAL. The activation is partitioned along W across the mesh (the causal
convs exchange halos with their neighbours; the single-head attention gathers H/W internally for a
replicated SDPA and re-partitions), all weights stay replicated, and the output is all-gathered
along W. The gathered output equals the single-device result.

On a Galaxy the 1x32 line is run as an 8x4 grid (see _mesh.py: CCLs along the 32-chip line
deadlock): the activation is zero-padded up to the grid, partitioned H over the 8-axis and W over
the 4-axis, and the gathered output is cropped back to the logical size.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanMidBlock
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import ResidentPort

# Mesh axes carrying the H and W partitions of the (8x4) grid.
_H_AXIS = 0
_W_AXIS = 1


class TtQwenImageMidBlock(ResidentPort):
    BODY_ATTR = "block"  # inside encoder3d/decoder3d the port is entered via forward_sharded

    def __init__(self, device, torch_module):
        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        self.num_convs = 2 * len(torch_module.resnets)  # two causal convs per residual block
        self.block = WanMidBlock(
            dim=torch_module.dim,
            num_layers=len(torch_module.attentions),
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        self.block.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, x, feat_cache=None, feat_idx=None, **_ignored):
        # x: replicated TILE [B, C, T, H, W] (BCTHW, like the torch reference).
        B, C, T, H, W = x.shape
        pc = self.parallel_config

        x = ttnn.permute(x, (0, 2, 3, 4, 1))  # BTHWC
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        # zero-pad H/W up to the grid; the Wan body masks the padding via logical_h / logical_w
        x, Hp = pad_to_multiple(x, 2, pc.height_parallel.factor)[0], H + (-H) % pc.height_parallel.factor
        x, Wp = pad_to_multiple(x, 3, pc.width_parallel.factor)[0], W + (-W) % pc.width_parallel.factor
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Fresh causal-conv cache, exactly as the reference's first (and only) chunk sees it.
        tt_feat_cache = [None] * self.num_convs
        tt_feat_idx = [0]
        out = self.block(x, H, feat_cache=tt_feat_cache, feat_idx=tt_feat_idx, logical_w=W)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
        ob, ot, oh, ow, oc = out.shape
        lh, lw = H * oh // Hp, W * ow // Wp
        if (oh, ow) != (lh, lw):
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (ob, ot, lh, lw, oc))

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW


def build(device, torch_module):
    return TtQwenImageMidBlock(device, torch_module)
