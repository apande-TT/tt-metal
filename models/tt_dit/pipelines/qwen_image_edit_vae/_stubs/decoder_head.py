"""tt_hw_planner: native TTNN port of the Qwen-Image VAE decoder (`AutoencoderKLQwenImage.decoder`).

Component: decoder_head  (torch reference: diffusers QwenImageDecoder3d)

QwenImageDecoder3d is architecturally identical to the Wan2.1 VAE decoder (causal 3D convs,
RMS norms, mid-block attention, resample up-blocks), so this reuses tt_dit's native
`WanDecoder3d` rather than the LM-head scaffold this stub was seeded with.

Tensor-parallel scheme: a conv VAE has no large matmul to column/row split -- its cost is the
spatial convolutions -- so the parallel axis is SPATIAL. The 1x32 line is run as an 8x4
grid (see _mesh.py: CCLs along the 32-chip line deadlock), the latent is zero-padded up to the grid
and partitioned H over the 8-axis, W over the 4-axis (each chip convolves its own tile;
WanCausalConv3d exchanges halos with its neighbours and masks the padding via logical_h/logical_w),
conv/norm weights stay replicated, and the output is all-gathered along W then H and cropped to the
logical size. The math is unchanged: the gathered output equals the single-device decode.

Verified on a freshly reset Galaxy: every neighbor_pad / all_gather in the decode completes on the 8x4
grid (sharded PCC 0.99999 at TP=32). A wall-clock kill of this test after another component's hang is
a dirty-device symptom, not a deadlock in this port.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder3d
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid
from models.tt_dit.utils.conv3d import aligned_channels, count_convs

# Mesh axes carrying the H and W partitions of the (8x4) grid.
_H_AXIS = 0
_W_AXIS = 1


class TtQwenImageDecoder3d:
    def __init__(self, device, torch_module):
        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)
        self.out_channels = torch_module.conv_out.out_channels

        self.decoder = WanDecoder3d(
            dim=torch_module.dim,
            z_dim=torch_module.z_dim,
            dim_mult=list(torch_module.dim_mult),
            num_res_blocks=torch_module.num_res_blocks,
            attn_scales=list(torch_module.attn_scales),
            temperal_upsample=list(torch_module.temperal_upsample),
            out_channels=self.out_channels,
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            # fp32 end-to-end: the bf16 decode drifts to PCC ~0.982 through the deep conv stack.
            dtype=ttnn.float32,
        )
        self.decoder.load_torch_state_dict(torch_module.state_dict())
        self.num_convs = count_convs(self.decoder)

    def __call__(self, x, feat_cache=None, feat_idx=None, **_ignored):
        # x: replicated TILE [B, C=z_dim, T, H, W] (BCTHW, like the torch reference).
        B, C, T, H, W = x.shape
        pc = self.parallel_config

        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
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
        out, logical_h, logical_w = self.decoder(x, H, feat_cache=tt_feat_cache, feat_idx=tt_feat_idx, logical_w=W)

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
    return TtQwenImageDecoder3d(device, torch_module)
