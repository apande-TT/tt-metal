# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for MLP (language_model.layers[i].mlp).

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


def _as_bf16(t):
    """Narrow a weight to bfloat16 ON THE HOST, before it is ever uploaded.

    ``from_torch(t, dtype=ttnn.bfloat16, ...)`` does NOT convert a float32 `t` on the host: it
    uploads the fp32 bytes, converts the LAYOUT on device in fp32, and only then emits a device
    Typecast to bf16.  Every call site here hands over a `.float()` tensor, so the layout
    conversion was moving 4 bytes per element to produce a 2-byte tensor and paying for a whole
    extra device op to do it.  Handing over bf16 makes the requested dtype the dtype that arrives:
    the conversion moves half the bytes and the typecast has nothing left to do.

    The VALUES are the same either way -- the fp32 -> bf16 rounding happens regardless, on the host
    here instead of on the device one op later.
    """
    return t.bfloat16() if hasattr(t, "bfloat16") else t


def _to_device(t, device):
    t = _as_bf16(t)
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


class TtMlp:
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


def build(device, torch_module):
    return TtMlp(device, torch_module)
