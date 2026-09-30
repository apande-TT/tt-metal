"""tt_hw_planner: native TTNN port of the Qwen-Image VAE channel RMS norm.

Component: qwen_image_r_m_s  (torch reference: diffusers QwenImageRMS_norm, `encoder.down_blocks.0.norm1`)

QwenImageRMS_norm computes F.normalize(x, dim=C) * sqrt(C) * gamma (+ bias), which is exactly an RMS
norm over the channel axis -- x / sqrt(mean_C(x^2)) * gamma. This reuses tt_dit's native `RMSNorm`
(the one WanResidualBlock uses, eps=1e-12) on the channels-last token matrix.

Tensor-parallel scheme (TP over the 1xN mesh): a norm's gamma is never sharded, and the reduction is
over channels, so the parallel axis is the voxel (token) axis. The flattened tokens are partitioned
across the mesh, gamma stays replicated, each chip normalises its own voxels, and the tokens are
all-gathered back in order. The gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.layers.normalization import RMSNorm
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import ResidentPort

_TOKEN_DIM = 2


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImageRMSNorm(ResidentPort):
    BODY_ATTR = "norm"  # inside encoder3d/decoder3d the port is entered via forward_sharded

    # precise: on a float32 activation, the norm is spelled out in float32 (sum_C x^2 -> rsqrt -> scale,
    # then the body's fused activation) instead of the fused dit_rms_norm kernel. Measured on the decoder's
    # last up block (96 ch, HF's own input): fused max abs error 3.9e-2, spelled out 4.8e-6. Off by default
    # (the graduated path); the e2e pipeline switches it on for the float32 decoder.
    precise = False

    def forward_sharded(self, x, *args, **kwargs):
        if not self.precise or x.dtype != ttnn.float32:
            return super().forward_sharded(x, *args, **kwargs)
        body = self.norm
        if getattr(self, "_scale32", None) is None:
            c = body.embedding_dim
            w = body.weight.data if body.weight is not None else None
            # F.normalize(x, dim=C) * sqrt(C) * gamma
            self._scale32 = ttnn.multiply(ttnn.typecast(w, ttnn.float32), float(c) ** 0.5) if w is not None else None
            self._bias32 = ttnn.typecast(body.bias.data, ttnn.float32) if body.bias is not None else None
            self._cfg32 = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        shape = list(x.shape)
        lead = 1
        for d in shape[:-2]:
            lead *= d
        layout = x.layout  # the fused kernel returns its input's layout; so does this path
        x3 = ttnn.reshape(x, (lead, shape[-2], shape[-1]))
        if layout != ttnn.TILE_LAYOUT:
            x3 = ttnn.to_layout(x3, ttnn.TILE_LAYOUT)
        ss = ttnn.sum(ttnn.multiply(x3, x3), dim=-1, keepdim=True, compute_kernel_config=self._cfg32)
        y = ttnn.multiply(x3, ttnn.rsqrt(ttnn.maximum(ss, 1e-24)))  # F.normalize eps = 1e-12 on the norm
        if self._scale32 is not None:
            y = ttnn.multiply(y, self._scale32)
        if self._bias32 is not None:
            y = ttnn.add(y, self._bias32)
        act = body.fused_activation
        if act is not None:
            assert act == ttnn.UnaryOpType.SILU, f"fused activation {act} not handled in precise mode"
            y = ttnn.silu(y)
        if layout != ttnn.TILE_LAYOUT:
            y = ttnn.to_layout(y, layout)
        return ttnn.reshape(y, shape)

    def __init__(self, device, torch_module):
        assert torch_module.channel_first, "only the channel-first (BCTHW) variant is used by this VAE"
        self.device = device
        mesh_shape = _mesh_shape(device)
        # Tokens are split across the larger mesh axis (the N of a 1xN mesh).
        self.tp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
        self.tp = mesh_shape[self.tp_axis]
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1) if self.tp > 1 else None

        gamma = torch_module.gamma.detach().float().reshape(-1)
        self.dim = gamma.shape[0]
        # `bias` is the float 0.0 when the module was built with bias=False.
        has_bias = not isinstance(torch_module.bias, float)
        self.norm = RMSNorm(
            embedding_dim=self.dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=has_bias,
            mesh_device=device,
            dtype=ttnn.bfloat16,
        )
        state = {"weight": gamma}
        if has_bias:
            state["bias"] = torch_module.bias.detach().float().reshape(-1)
        self.norm.load_torch_state_dict(state)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x, **_ignored):
        # x: replicated TILE [B, C, T, H, W] (BCTHW, like the torch reference).
        B, C, T, H, W = x.shape
        n_tokens = B * T * H * W
        assert n_tokens % self.tp == 0, f"{n_tokens} voxels must divide the TP={self.tp} mesh"

        x = ttnn.permute(x, (0, 2, 3, 4, 1))  # BTHWC
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (1, 1, n_tokens, C))
        if self.tp > 1:
            x = ttnn.mesh_partition(x, dim=_TOKEN_DIM, cluster_axis=self.tp_axis)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        out = self.norm(x, compute_kernel_config=self.compute_kernel_config)

        if self.tp > 1:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
            out = self.ccl_manager.all_gather(out, dim=_TOKEN_DIM, mesh_axis=self.tp_axis, use_hyperparams=False)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.reshape(out, (B, T, H, W, C))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW


def build(device, torch_module=None):
    return TtQwenImageRMSNorm(device, torch_module)
