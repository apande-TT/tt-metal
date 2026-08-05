# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for LlamaMLP (language_model.layers[i].mlp).

SwiGLU MLP: gate_proj -> silu * up_proj -> down_proj.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _to_device(t, device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class TtLlamaMLP:
    def __init__(self, device, torch_module):
        self.device = device
        self.gate_weight = _to_device(torch_module.gate_proj.weight.T.contiguous().float(), device)
        self.up_weight = _to_device(torch_module.up_proj.weight.T.contiguous().float(), device)
        self.down_weight = _to_device(torch_module.down_proj.weight.T.contiguous().float(), device)

    def __call__(self, x, **kwargs):
        gate = ttnn.linear(x, self.gate_weight, compute_kernel_config=_HIFI4_CFG)
        gate = ttnn.silu(gate)
        up = ttnn.linear(x, self.up_weight, compute_kernel_config=_HIFI4_CFG)
        x = ttnn.multiply(gate, up)
        x = ttnn.linear(x, self.down_weight, compute_kernel_config=_HIFI4_CFG)
        return x


def build(device, torch_module=None):
    return TtLlamaMLP(device, torch_module)
