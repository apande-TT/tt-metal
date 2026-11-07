# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextVisionPatchMerger` (GLM-5.3-Flash vision -> LLM).

Reference: h = GELU(LayerNorm(proj(x)));  out = down( silu(clamp(gate h, max=L)) * clamp(up h, -L, L) )
(nn.GELU is the exact erf form; LayerNorm is affine; all Linears bias-free in this checkpoint.)

Tensor parallelism over N chips:
  * proj COLUMN-parallel, then an all_gather of its columns -- the LayerNorm needs the full width.
  * LayerNorm + GELU replicated (fp32).
  * gate/up COLUMN-parallel (clamp + SwiGLU elementwise, stay local), down ROW-parallel + all-reduce.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _Ctx
from models.demos.glm_5_3_flash._stubs.vision_attention import _padded_all_reduce
from models.demos.glm_5_3_flash._stubs.vision_patch_embed import _gather_cols


class _PatchMerger:
    def __init__(self, ctx, m):
        self.ctx = c = ctx
        for lin in (m.proj, m.gate_proj, m.up_proj, m.down_proj):
            if lin.bias is not None:
                raise NotImplementedError("biased merger projections not used by this checkpoint")
        if m.act1.__class__.__name__ != "GELU" or getattr(m.act1, "approximate", "none") != "none":
            raise NotImplementedError(f"merger activation {m.act1}")
        T = lambda w: w.detach().to(torch.float32).t()
        self.w_proj = c.put(T(m.proj.weight), shard_dim=-1)
        ln = m.post_projection_norm
        self.ln_w = c.f32(ln.weight.detach().reshape(1, 1, 1, -1))
        self.ln_b = c.f32(ln.bias.detach().reshape(1, 1, 1, -1))
        self.ln_eps = float(ln.eps)
        self.w_gate = c.put(T(m.gate_proj.weight), shard_dim=-1)
        self.w_up = c.put(T(m.up_proj.weight), shard_dim=-1)
        self.w_down = c.put(T(m.down_proj.weight), shard_dim=0)
        self.limit = float(m.swiglu_limit)

    def __call__(self, x):
        """x [1,1,R,D] replicated -> fp32 [1,1,R,D] replicated."""
        c = self.ctx
        h = _gather_cols(c, c.lin_hp(ttnn.typecast(x, ttnn.float32), self.w_proj))  # [1,1,R,D] fp32
        mu = ttnn.mean(h, dim=-1, keepdim=True)
        hc = ttnn.subtract(h, mu)
        var = ttnn.mean(ttnn.multiply(hc, hc), dim=-1, keepdim=True)
        h = ttnn.add(ttnn.multiply(ttnn.multiply(hc, ttnn.rsqrt(ttnn.add(var, self.ln_eps))), self.ln_w), self.ln_b)
        h = ttnn.gelu(h, fast_and_approximate_mode=False)
        gate = ttnn.clamp(c.lin_hp(h, self.w_gate), max=self.limit)
        up = ttnn.clamp(c.lin_hp(h, self.w_up), min=-self.limit, max=self.limit)
        mid = ttnn.multiply(ttnn.silu(gate), up)
        return _padded_all_reduce(c, c.lin_hp(mid, self.w_down))


class TtVisionPatchMerger:
    def __init__(self, device, m):
        self.ctx = _Ctx(device)
        self.pm = _PatchMerger(self.ctx, m)

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
        x = ttnn.reshape(x, [1, 1, R, D])
        return ttnn.reshape(self.pm(x), in_shape)


def build(device, torch_module=None):
    return TtVisionPatchMerger.build(device, torch_module)


def vision_patch_merger(device, torch_module=None):
    return TtVisionPatchMerger.build(device, torch_module)
