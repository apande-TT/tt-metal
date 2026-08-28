# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of Qwen3RMSNorm.

Component `r_m_s_norm` of `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
(`model.layers.0.self_attn.q_norm`):

    y = x * rsqrt(mean(x^2) + eps) * weight

normalising over the LAST axis only. This instance is the per-head query
norm, so its axis is head_dim = 128.

No tensor-parallel split: this is a per-element scale over an axis that no
TP scheme cuts (the head axis is what gets sharded, not head_dim), and its
gamma is a 1-D vector, not a matmul weight. Under TP it stays REPLICATED --
every chip normalises its own head slice with the same gamma, which is
exactly what the sharded attention/decoder ports do with it.

The gamma is staged in the (1, 1, dim/32, 32) ROW_MAJOR form `ttnn.rms_norm`
expects for a per-channel weight.
"""
from __future__ import annotations

import torch

import ttnn

TILE = 32


def _is_mesh(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


class TtRMSNorm:
    """Native ttnn RMSNorm; replicated on every chip of a mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)

        w = torch_module.state_dict()["weight"].detach()
        self.dim = int(w.numel())
        self.eps = float(getattr(torch_module, "variance_epsilon", 0.0) or getattr(torch_module, "eps", 0.0) or 1e-6)
        self.weight = ttnn.from_torch(
            w.reshape(1, 1, self.dim // TILE, TILE).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, *args, **kwargs):
        x = kwargs.pop("hidden_states", None)
        for name in ("x", "input", "inputs"):
            if x is None:
                x = kwargs.pop(name, None)
        if x is None:
            for a in args:
                if a is not None:
                    x = a
                    break
        if x is None:
            raise ValueError("r_m_s_norm stub: no input tensor supplied")
        if torch.is_tensor(x):
            x = ttnn.from_torch(
                x.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight, compute_kernel_config=self.compute_kernel_config)

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("r_m_s_norm stub needs the torch reference module to source its weights")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)


def r_m_s_norm(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)
