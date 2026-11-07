# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextVisionMLP` (GLM-5.3-Flash vision SwiGLU MLP).

Math: down( silu(clamp(gate x + bg, max=L)) * clamp(up x + bu, -L, L) ) + bd.
Wraps the `_VisionMLP` validated inside the vision block (vision_block.py): gate/up COLUMN-parallel
with sharded biases, down ROW-parallel + all-reduce, down bias added once after the reduction.
"""
from __future__ import annotations

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _Ctx
from models.demos.glm_5_3_flash._stubs.vision_block import _VisionMLP


class TtVisionMLP:
    def __init__(self, device, m):
        self.ctx = _Ctx(device)
        self.mlp = _VisionMLP(self.ctx, m)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, hidden_state, *args, **kwargs):
        x = hidden_state
        in_shape = list(x.shape)
        D = in_shape[-1]
        R = 1
        for d in in_shape[:-1]:
            R *= d
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, R, D]), ttnn.float32)
        return ttnn.reshape(self.mlp(x), in_shape)


def build(device, torch_module=None):
    return TtVisionMLP.build(device, torch_module)


def vision_m_l_p(device, torch_module=None):
    return TtVisionMLP.build(device, torch_module)
