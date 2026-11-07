# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextTextMLP` (GLM-5.3-Flash dense / shared-expert MLP).

Math (HF): down( silu(clamp(gate x, max=L)) * clamp(up x, -L, L) ),  L = swiglu_limit.

The implementation is the decoder-layer stub's `_MLP` (validated inside the full layer), wrapped
here so the two cannot drift. Tensor parallelism over N chips: gate_proj / up_proj COLUMN-parallel
(intermediate/N features per chip, the clamp + SwiGLU are elementwise so they stay local),
down_proj ROW-parallel, then an all-reduce (all_gather + on-device sum) -> replicated output.
"""
from __future__ import annotations

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _MLP, _Ctx


class TtTextMLP:
    def __init__(self, ctx, m):
        self.ctx = ctx
        self.mlp = _MLP(ctx, m)

    @classmethod
    def build(cls, device, torch_module):
        return cls(_Ctx(device), torch_module)

    def __call__(self, x, *args, **kwargs):
        in_shape = list(x.shape)
        D = in_shape[-1]
        rows = 1
        for d in in_shape[:-1]:
            rows *= d
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, rows, D]), ttnn.float32)
        return ttnn.reshape(self.mlp(x), in_shape)


def build(device, torch_module=None):
    return TtTextMLP.build(device, torch_module)


def text_m_l_p(device, torch_module=None):
    return TtTextMLP.build(device, torch_module)
