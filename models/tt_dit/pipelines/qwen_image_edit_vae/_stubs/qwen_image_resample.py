"""tt_hw_planner: native TTNN port of the Qwen-Image VAE resample (`encoder.down_blocks.2`).

Component: qwen_image_resample  (torch reference: diffusers QwenImageResample, downsample2d here)

QwenImageResample (ZeroPad2d(0,1,0,1) + stride-2 3x3 conv for downsample; nearest-upsample + conv
for upsample; plus a temporal conv in the 3d modes) is architecturally identical to the Wan2.1 VAE
resample, so this reuses tt_dit's native `WanResample` (which realises the padded stride-2 conv as a
stride-1 conv sampled at [1::2, 1::2]).

Tensor-parallel scheme (TP over the 1xN mesh): same as encoder_stack, which contains this block --
the parallel axis is SPATIAL. The activation is partitioned along W across the mesh (the conv
exchanges halos with its neighbours; each W shard is even-width and starts at an even column, so the
per-shard [1::2] subsample is the global one), weights stay replicated, and the output is
all-gathered along W. The gathered output equals the single-device result.

On a Galaxy the 1x32 line is run as an 8x4 grid (see _mesh.py: CCLs along the 32-chip line
deadlock): the activation is zero-padded up to the grid, partitioned H over the 8-axis and W over
the 4-axis, and the gathered output is cropped back to the logical size.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanResample
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import ResidentPort

# Mesh axes carrying the H and W partitions of the (8x4) grid.
_H_AXIS = 0
_W_AXIS = 1


class TtQwenImageResample(ResidentPort):
    BODY_ATTR = "block"  # inside encoder3d/decoder3d the port is entered via forward_sharded

    @classmethod
    def around_resample(cls, body, device, parallel_config, ccl_manager, op_module=None):
        """Resident resample: the body's spatial x2 runs as the graduated op ports.

        upsample*: `qwen_image_upsample` (nearest x2 on the W shard) feeds the body's 3x3 conv.
        downsample*: `zero_pad2d` pads the bottom row explicitly (H is whole on every chip in this
        layout), then a 3x3 conv with NO H padding (W still halo-padded, the same weights) runs, sampled
        at [0::2, 1::2]. That is HF's ZeroPad2d(0, 1, 0, 1) + stride-2 conv exactly: output row i reads
        padded rows 2i..2i+2, and output column j reads columns 2j..2j+2, with the global right zero
        column coming from the halo exchange. The temporal (3d) path of the body is untouched.
        """
        self = cls.around(body, device, parallel_config, ccl_manager)
        self.upsample = self.zero_pad = self.conv_valid_h = None
        if op_module is None:
            return self
        if body.is_upsample:
            from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_upsample import TtQwenImageUpsample

            self.upsample = TtQwenImageUpsample.resident(device, op_module)
            body.spatial_upsample = lambda x_NHWC: self.upsample.forward_sharded(x_NHWC)
        elif parallel_config.height_parallel.factor == 1:
            from models.tt_dit.models.vae.vae_wan2_1 import WanConv2d
            from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.zero_pad2d import TtZeroPad2d

            self.zero_pad = TtZeroPad2d.resident(device, op_module)
            conv = body.conv
            self.conv_valid_h = WanConv2d(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                padding=(0, 0, 1),
                mesh_device=device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=conv.dtype,
            )
            # same prepared weights / blocking as the graduated conv (no second copy of the weights)
            self.conv_valid_h.conv_config = conv.conv_config
            self.conv_valid_h.weight = conv.weight
            self.conv_valid_h.bias = conv.bias
            body.spatial_downsample = lambda x, logical_h, logical_w=0: self.spatial_downsample(x, logical_h, logical_w)
        return self

    def spatial_downsample(self, x_BTHWC, logical_h, logical_w=0):
        x = self.zero_pad.forward_sharded(x_BTHWC)  # bottom zero row
        y = self.conv_valid_h(x, logical_h + 1, logical_w=logical_w)  # [B, T, H - 1, W, C']
        return y[:, :, 0::2, 1::2, :]

    def __init__(self, device, torch_module):
        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        self.mode = torch_module.mode
        spatial_conv = torch_module.resample[-1]
        self.block = WanResample(
            dim=torch_module.dim,
            mode=self.mode,
            resample_out_dim=spatial_conv.out_channels,
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

        # Fresh causal-conv cache, exactly as the reference's first (and only) chunk sees it.
        tt_feat_cache = [None]
        tt_feat_idx = [0]
        out, _logical_h, _logical_w = self.block(x, H, feat_cache=tt_feat_cache, feat_idx=tt_feat_idx, logical_w=W)

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
    return TtQwenImageResample(device, torch_module)
