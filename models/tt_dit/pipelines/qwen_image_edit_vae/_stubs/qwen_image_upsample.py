"""tt_hw_planner: native TTNN port of the Qwen-Image VAE spatial upsample.

Component: qwen_image_upsample  (torch reference: diffusers QwenImageUpsample,
`decoder.up_blocks.0.upsamplers.0.resample.0`)

QwenImageUpsample is nn.Upsample(scale_factor=2, mode="nearest-exact") on NCHW frames. For an
integer scale, nearest-exact (src = floor((dst + 0.5) / s)) picks the same source pixel as nearest
(src = floor(dst / s)), so this is `ttnn.upsample` (nearest) on the NHWC tensor -- the same op
WanResample uses for its upsample.

Tensor-parallel scheme (TP over the 1xN mesh): the op has no weights and is pixel-local, so the
parallel axis is SPATIAL, as in decoder_head. The frame is partitioned along W across the mesh;
nearest-upsampling a W shard yields exactly the matching shard of the upsampled frame (no halo
needed), and the output is all-gathered along W. The gathered output equals the single-device result.

On a Galaxy the 1x32 line is run as an 8x4 grid (see _mesh.py: CCLs along the 32-chip line
deadlock): the activation is zero-padded up to the grid, partitioned H over the 8-axis and W over
the 4-axis, and the gathered output is cropped back to the logical size.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid

# NHWC layout: H is dim 1, W is dim 2; H is split over mesh axis 0, W over mesh axis 1.
_H_DIM = 1
_W_DIM = 2


class TtQwenImageUpsample:
    def __init__(self, device, torch_module):
        assert torch_module.mode in ("nearest", "nearest-exact"), f"unsupported upsample mode {torch_module.mode}"
        assert torch_module.size is None, "only scale_factor upsampling is used by this VAE"
        scale = torch_module.scale_factor
        scale = tuple(scale) if isinstance(scale, (tuple, list)) else (scale, scale)
        assert all(
            float(s).is_integer() for s in scale
        ), f"nearest-exact == nearest only for integer scales, got {scale}"
        self.scale = tuple(int(s) for s in scale)

        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        self.shape = shape
        self.tp = shape[0] * shape[1]
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1) if self.tp > 1 else None

    @classmethod
    def resident(cls, device, torch_module):
        """The op port used inside the decoder (forward_sharded only): no CCLManager / L1 semaphores."""
        self = cls.__new__(cls)
        assert torch_module.mode in ("nearest", "nearest-exact") and torch_module.size is None
        scale = torch_module.scale_factor
        scale = tuple(scale) if isinstance(scale, (tuple, list)) else (scale, scale)
        assert all(float(s).is_integer() for s in scale)
        self.scale = tuple(int(s) for s in scale)
        self.device, self.tp, self.ccl_manager = device, 1, None
        return self

    def forward_sharded(self, x_NHWC):
        """Inside the decoder: nearest x2 of a ROW_MAJOR NHWC W-shard is exactly the matching shard of the
        upsampled frame (no halo), so the op runs on the shard as is."""
        return ttnn.upsample(x_NHWC, scale_factor=self.scale)

    def __call__(self, x, **_ignored):
        # x: replicated TILE [N, C, H, W] (NCHW, like the torch reference).
        N, C, H, W = x.shape
        splits = [(dim, ax) for dim, ax in ((_H_DIM, 0), (_W_DIM, 1)) if self.shape[ax] > 1]

        x = ttnn.permute(x, (0, 2, 3, 1))  # NHWC
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        # zero-pad H/W up to the grid; nearest upsampling maps each padded row/col to its own rows/cols,
        # so the padding lands wholly past the logical edge and is cropped after the gather.
        for dim, ax in splits:
            x, _ = pad_to_multiple(x, dim, self.shape[ax])
            x = ttnn.mesh_partition(x, dim=dim, cluster_axis=ax)

        out = ttnn.upsample(x, scale_factor=self.scale)

        for dim, ax in reversed(splits):
            out = self.ccl_manager.all_gather(out, dim=dim, mesh_axis=ax, use_hyperparams=False)
        oh, ow = H * self.scale[0], W * self.scale[1]
        if (out.shape[_H_DIM], out.shape[_W_DIM]) != (oh, ow):
            out = ttnn.slice(out, (0, 0, 0, 0), (N, oh, ow, C))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 3, 1, 2))  # NCHW


def build(device, torch_module=None):
    return TtQwenImageUpsample(device, torch_module)
