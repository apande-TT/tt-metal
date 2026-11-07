# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn RMSNorm for `zai-org/GLM-5.3-Flash` (e.g. visual.blocks.N.norm1).

y = x * rsqrt(mean(x^2, -1) + eps) * weight, computed on device with ttnn.rms_norm.
"""
from __future__ import annotations

import torch

import ttnn

TILE = 32


def _is_mesh(device):
    try:
        return device.get_num_devices() > 1
    except Exception:
        return False


class TtRMSNorm:
    def __init__(self, device, weight, eps, dim):
        self.device = device
        self.weight = weight
        self.eps = eps
        self.dim = dim
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def build(cls, device, torch_module):
        w = getattr(torch_module, "weight", None)
        eps = getattr(torch_module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(torch_module, "eps", 1e-6)
        dim = int(w.shape[-1])
        w = w.detach().to(torch.bfloat16)
        # ttnn.rms_norm expects the gamma in ROW_MAJOR as [dim/32, 32].
        w_rm = w.reshape(dim // TILE, TILE)
        kwargs = dict(
            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if _is_mesh(device):
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        tt_w = ttnn.from_torch(w_rm, **kwargs)
        return cls(device, tt_w, float(eps), dim)

    def __call__(self, x, *args, **kwargs):
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
        )


def build(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)


def r_m_s_norm(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)
