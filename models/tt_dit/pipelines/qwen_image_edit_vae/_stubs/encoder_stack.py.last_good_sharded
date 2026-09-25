"""tt_hw_planner: native TTNN port of the Qwen-Image VAE encoder (`AutoencoderKLQwenImage.encoder`).

Component: encoder_stack  (torch reference: diffusers QwenImageEncoder3d)

QwenImageEncoder3d is architecturally identical to the Wan2.1 VAE encoder (causal 3D convs,
RMS norms, resample down-blocks, mid-block attention), so this reuses tt_dit's native
`WanEncoder3D` rather than the llama-vision-encoder scaffold this stub was seeded with.

Tensor-parallel scheme: a conv VAE has no large matmul to column/row split -- its cost is the
spatial convolutions -- so the parallel axis is SPATIAL. The 1x32 line is run as an 8x4 grid (see
_mesh.py: CCLs along the 32-chip line deadlock, and a 32-way W split leaves each chip only 2 image
columns ahead of the 8x downsample). The image is zero-padded up to the grid and partitioned H over
the 8-axis, W over the 4-axis (each chip convolves its own tile; WanCausalConv3d exchanges halos with
its neighbours and masks the padding via logical_h/logical_w), conv/norm weights stay replicated, and
the latent is all-gathered along W then H and cropped to the logical size. The math is unchanged:
the gathered output equals the single-device encode.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder3D
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid
from models.tt_dit.utils.conv3d import aligned_channels, count_convs

# Mesh axes carrying the H and W partitions of the (8x4) grid.
_H_AXIS = 0
_W_AXIS = 1


class TtQwenImageEncoder3d:
    def __init__(self, device, torch_module):
        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)
        self.out_channels = torch_module.conv_out.out_channels

        self.encoder = WanEncoder3D(
            in_channels=torch_module.conv_in.in_channels,
            dim=torch_module.dim,
            z_dim=torch_module.z_dim,
            dim_mult=list(torch_module.dim_mult),
            num_res_blocks=torch_module.num_res_blocks,
            attn_scales=list(torch_module.attn_scales),
            temperal_downsample=list(torch_module.temperal_downsample),
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        self.encoder.load_torch_state_dict(torch_module.state_dict())
        self.num_convs = count_convs(self.encoder)

    def __call__(self, x, feat_cache=None, feat_idx=None, **_ignored):
        # x: replicated TILE [B, C=3, T, H, W] (BCTHW, like the torch reference).
        B, C, T, H, W = x.shape
        pc = self.parallel_config

        x = ttnn.permute(x, (0, 2, 3, 4, 1))  # BTHWC
        c_pad = aligned_channels(C) - C
        if c_pad:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, 0), (0, 0), (0, c_pad)], value=0.0)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        # zero-pad H/W up to the grid; the Wan stack masks the padding via logical_h / logical_w
        x, _ = pad_to_multiple(x, 2, pc.height_parallel.factor)
        x, _ = pad_to_multiple(x, 3, pc.width_parallel.factor)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)

        # Fresh causal-conv cache, exactly as the reference's first (and only) chunk sees it.
        tt_feat_cache = [None] * self.num_convs
        tt_feat_idx = [0]
        out, logical_h, logical_w = self.encoder(x, H, feat_cache=tt_feat_cache, feat_idx=tt_feat_idx, logical_w=W)

        out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
        ob, ot, oh, ow, oc = out.shape
        lh, lw = logical_h or oh, logical_w or ow
        if (oh, ow) != (lh, lw):
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (ob, ot, lh, lw, oc))

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW
        if out.shape[1] != self.out_channels:
            ob, _, ot, oh, ow = out.shape
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (ob, self.out_channels, ot, oh, ow))
        return out


def build(device, torch_module):
    return TtQwenImageEncoder3d(device, torch_module)
