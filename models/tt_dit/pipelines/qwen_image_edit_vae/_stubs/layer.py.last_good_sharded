"""tt_hw_planner: native TTNN port of one Qwen-Image VAE encoder layer (`encoder.down_blocks[0]`).

Component: layer  (torch reference: diffusers QwenImageResidualBlock)

The VAE has no transformer `encoder.layers`; its per-layer unit is the residual block
(RMS norm + SiLU -> causal conv3d -> RMS norm + SiLU -> causal conv3d, plus residual). This stub
replaces the llama-layernorm scaffold it was seeded with by tt_dit's native `WanResidualBlock`,
which is architecturally identical.

Tensor-parallel scheme: same as encoder_stack -- a conv block has no large matmul to column/row
split, so the parallel axis is SPATIAL. The 1x32 line is run as an 8x4 grid (see _mesh.py: CCLs along
the 32-chip line deadlock); the activation is zero-padded up to the grid and partitioned H over the
8-axis, W over the 4-axis (WanCausalConv3d exchanges halos with its neighbours and masks the padding
via logical_h/logical_w), norm/conv weights stay replicated, and the output is all-gathered along W
then H and cropped to the logical size. The gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanResidualBlock
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid

# Mesh axes carrying the H and W partitions of the (8x4) grid.
_H_AXIS = 0
_W_AXIS = 1


class TtQwenImageResidualLayer:
    def __init__(self, device, torch_module):
        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        self.block = WanResidualBlock(
            in_dim=torch_module.in_dim,
            out_dim=torch_module.out_dim,
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
        # zero-pad H/W up to the grid; the Wan block masks the padding via logical_h / logical_w
        x, _ = pad_to_multiple(x, 2, pc.height_parallel.factor)
        x, _ = pad_to_multiple(x, 3, pc.width_parallel.factor)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Fresh causal-conv cache, exactly as the reference's first (and only) chunk sees it.
        tt_feat_cache = [None, None]
        tt_feat_idx = [0]
        out = self.block(x, H, feat_cache=tt_feat_cache, feat_idx=tt_feat_idx, logical_w=W)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
        ob, ot, oh, ow, oc = out.shape
        if (oh, ow) != (H, W):
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (ob, ot, H, W, oc))

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW


def build(device, torch_module):
    return TtQwenImageResidualLayer(device, torch_module)
