"""tt_hw_planner: native TTNN port of the Qwen-Image VAE mid-block attention.

Component: qwen_image_attention_block  (torch reference: diffusers QwenImageAttentionBlock,
`encoder.mid_block.attentions.0`)

QwenImageAttentionBlock is architecturally identical to the Wan2.1 VAE attention block (RMS norm,
1x1 to_qkv, single-head spatial SDPA per frame, 1x1 proj, residual), so this reuses tt_dit's native
`WanAttentionBlock`.

Tensor-parallel scheme (TP over the 1xN mesh): the block is single-head (no heads to split) and a
column split of the 384 -> 1152 qkv projection is not tile-aligned per chip, so -- as in
encoder_stack, which contains this block -- the parallel axis is SPATIAL. The activation arrives
W-partitioned across the mesh; the block all-gathers H/W for the (replicated) SDPA and re-partitions
its output, weights stay replicated, the residual is added per shard, and the output is
all-gathered along W. The gathered output equals the single-device result.

On a Galaxy the 1x32 line is run as an 8x4 grid (see _mesh.py: CCLs along the 32-chip line
deadlock): the activation is zero-padded up to the grid, partitioned H over the 8-axis and W over
the 4-axis, and the gathered output is cropped back to the logical size.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanAttentionBlock
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import ResidentPort

# Mesh axes carrying the H and W partitions of the (8x4) grid.
_H_AXIS = 0
_W_AXIS = 1


class TtQwenImageAttentionBlock(ResidentPort):
    BODY_ATTR = "block"  # inside encoder3d/decoder3d the port is entered via forward_sharded

    def __init__(self, device, torch_module):
        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        self.block = WanAttentionBlock(
            dim=torch_module.dim,
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        self.block.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, x, **_ignored):
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

        out = self.block(x, H, logical_w=W)

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
    return TtQwenImageAttentionBlock(device, torch_module)
