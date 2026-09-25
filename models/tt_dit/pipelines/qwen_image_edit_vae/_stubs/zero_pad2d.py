"""tt_hw_planner: native TTNN port of the Qwen-Image VAE downsample zero-pad.

Component: zero_pad2d  (torch reference: nn.ZeroPad2d((0, 1, 0, 1)), `encoder.down_blocks.2.resample.0`)

nn.ZeroPad2d(left, right, top, bottom) zero-pads the last two (H, W) axes of an NCHW frame; here it
adds one zero row at the bottom and one zero column on the right before the stride-2 downsample conv.
This is a single `ttnn.pad` on the row-major NCHW tensor.

Tensor-parallel scheme (TP over the 1xN mesh): the op has no weights and pads every channel
identically, so the parallel axis is the CHANNEL axis -- a W split would put the right-hand pad on
the last chip only. Channels are partitioned across the mesh, each chip pads its own channels, and
the output is all-gathered along C. The gathered output equals the single-device result.

On a Galaxy the 1x32 line is run as an 8x4 grid (see _mesh.py: CCLs along the 32-chip line
deadlock): the activation is zero-padded up to the grid, partitioned H over the 8-axis and W over
the 4-axis, and the gathered output is cropped back to the logical size.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._mesh import mesh_shape, pad_to_multiple, physical_grid

# NCHW layout: C is dim 1.
_C_DIM = 1


class TtZeroPad2d:
    def __init__(self, device, torch_module):
        left, right, top, bottom = torch_module.padding
        assert float(torch_module.value) == 0.0, f"expected zero padding, got value={torch_module.value}"
        self.pad_h = (top, bottom)
        self.pad_w = (left, right)

        self.device = device = physical_grid(device)
        shape = mesh_shape(device)
        # Channels are split over every mesh axis with more than one chip, outermost first.
        self.tp_axes = [ax for ax in (0, 1) if shape[ax] > 1]
        self.tp = shape[0] * shape[1]
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1) if self.tp > 1 else None

    @classmethod
    def resident(cls, device, torch_module):
        """The op port used inside the encoder (forward_sharded only): no CCLManager / L1 semaphores."""
        self = cls.__new__(cls)
        left, right, top, bottom = torch_module.padding
        assert float(torch_module.value) == 0.0, f"expected zero padding, got value={torch_module.value}"
        self.pad_h, self.pad_w = (top, bottom), (left, right)
        self.device, self.tp, self.ccl_manager = device, 1, None
        return self

    def forward_sharded(self, x_BTHWC):
        """Inside the encoder: x is a ROW_MAJOR BTHWC tensor with H whole on every chip and W split over
        the mesh. The bottom zero row is padded here, locally. The right zero column falls on the global W
        edge, which the following W-sharded conv's halo exchange zero-fills; that conv samples odd columns,
        so its left halo column is never read."""
        top, bottom = self.pad_h
        assert tuple(self.pad_w) == (0, 1), f"downsample pad expected (0, 1) on W, got {self.pad_w}"
        B, T, H, W, C = x_BTHWC.shape
        x = ttnn.reshape(x_BTHWC, (B * T, H, W, C))
        x = ttnn.pad(x, [(0, 0), (top, bottom), (0, 0), (0, 0)], value=0.0)
        return ttnn.reshape(x, (B, T, H + top + bottom, W, C))

    def __call__(self, x, **_ignored):
        # x: replicated TILE [N, C, H, W] (NCHW, like the torch reference).
        N, C, H, W = x.shape

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x, _ = pad_to_multiple(x, _C_DIM, self.tp)
        for ax in self.tp_axes:
            x = ttnn.mesh_partition(x, dim=_C_DIM, cluster_axis=ax)

        out = ttnn.pad(x, [(0, 0), (0, 0), self.pad_h, self.pad_w], value=0.0)

        for ax in reversed(self.tp_axes):
            out = self.ccl_manager.all_gather(out, dim=_C_DIM, mesh_axis=ax, use_hyperparams=False)
        if out.shape[_C_DIM] != C:
            _, _, oh, ow = out.shape
            out = ttnn.slice(out, (0, 0, 0, 0), (N, C, oh, ow))
        return out


def build(device, torch_module=None):
    return TtZeroPad2d(device, torch_module)
