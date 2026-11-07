# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Glm5NextTextRMSNormGated` (GLM-5.3-Flash KDA output norm, `o_norm`).

Reference (strict fp32): y = weight * x * rsqrt(mean(x^2, -1) + eps) * sigmoid(gate)
Computed on device in fp32 (same formulation the KDA port uses for o_norm). The weight is a
replicated per-channel vector; the op is elementwise over channels, so on a mesh every chip
computes the full (replicated) result.
"""
from __future__ import annotations

import torch

import ttnn


class TtTextRMSNormGated:
    def __init__(self, device, m):
        self.device = device
        self.eps = float(m.variance_epsilon)
        if getattr(m, "activation", "sigmoid") != "sigmoid":
            raise NotImplementedError(m.activation)
        self.w = self._put(m.weight.detach().reshape(1, 1, 1, -1))

    def _mesh(self):
        try:
            return self.device.get_num_devices() > 1
        except Exception:
            return False

    def _put(self, t):
        kw = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        if self._mesh():
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(t.to(torch.float32).contiguous(), **kw)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _dev(self, t):
        if isinstance(t, torch.Tensor):
            return self._put(t)
        if t.layout != ttnn.TILE_LAYOUT:
            t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
        return ttnn.typecast(t, ttnn.float32)

    def __call__(self, hidden_states, gate=None, **kwargs):
        if gate is None:
            raise ValueError("Glm5NextTextRMSNormGated requires `gate`")
        x, g = self._dev(hidden_states), self._dev(gate)
        var = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        y = ttnn.multiply(ttnn.multiply(x, ttnn.rsqrt(ttnn.add(var, self.eps))), self.w)
        return ttnn.multiply(y, ttnn.sigmoid(g))


def build(device, torch_module=None):
    return TtTextRMSNormGated.build(device, torch_module)


def text_r_m_s_norm_gated(device, torch_module=None):
    return TtTextRMSNormGated.build(device, torch_module)
