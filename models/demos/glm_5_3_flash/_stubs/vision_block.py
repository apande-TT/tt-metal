# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextVisionBlock` (GLM-5.3-Flash vision tower block).

Reference:  h = x + attn(RMSNorm1(x));  out = h + MLP(RMSNorm2(h))
            MLP(z) = down( silu(clamp(gate z + bg, max=L)) * clamp(up z + bu, -L, L) ) + bd

Tensor parallelism over N chips:
  * attention: the head-parallel `_VisionAttn` (see vision_attention.py).
  * MLP: gate/up COLUMN-parallel (their biases sharded the same way, so the clamp + SwiGLU stay
    local), down ROW-parallel, all-reduce (all_gather + sum), down bias added once after it.
  * RMSNorms and the residual stream: replicated (fp32).
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _Ctx
from models.demos.glm_5_3_flash._stubs.vision_attention import _padded_all_reduce


class _VisionMLP:
    def __init__(self, ctx, m):
        self.ctx = c = ctx
        I = int(m.intermediate_size) if hasattr(m, "intermediate_size") else m.gate_proj.out_features
        if I % c.n:
            raise ValueError(f"intermediate {I} not divisible by mesh {c.n}")
        T = lambda w: w.detach().to(torch.float32).t()
        b = lambda lin: lin.bias.detach().to(torch.float32) if lin.bias is not None else None
        self.w_gate = c.put(T(m.gate_proj.weight), shard_dim=-1)
        self.w_up = c.put(T(m.up_proj.weight), shard_dim=-1)
        self.w_down = c.put(T(m.down_proj.weight), shard_dim=0)
        bg, bu, bd = b(m.gate_proj), b(m.up_proj), b(m.down_proj)
        self.b_gate = c.f32(bg.reshape(1, 1, 1, -1), shard_dim=-1) if bg is not None else None
        self.b_up = c.f32(bu.reshape(1, 1, 1, -1), shard_dim=-1) if bu is not None else None
        self.b_down = c.f32(bd.reshape(1, 1, 1, -1)) if bd is not None else None
        self.limit = float(m.swiglu_limit)
        if m.act_fn.__class__.__name__ not in ("SiLU", "SiLUActivation"):
            raise NotImplementedError(f"activation {m.act_fn}")

    def __call__(self, x):
        """x fp32 [1,1,R,D] replicated -> fp32 [1,1,R,D] replicated."""
        c = self.ctx
        add_b = lambda t, b: ttnn.add(t, b) if b is not None else t
        gate = ttnn.clamp(add_b(c.lin_hp(x, self.w_gate), self.b_gate), max=self.limit)
        up = ttnn.clamp(add_b(c.lin_hp(x, self.w_up), self.b_up), min=-self.limit, max=self.limit)
        h = ttnn.multiply(ttnn.silu(gate), up)
        out = _padded_all_reduce(c, c.lin_hp(h, self.w_down))
        return add_b(out, self.b_down)


class _VisionBlock:
    """h = x + vision_attention(r_m_s_norm(x));  out = h + vision_m_l_p(r_m_s_norm(h)) -- composed of the
    graduated leaf ports (norms: r_m_s_norm stub; attention: vision_attention stub; MLP: vision_m_l_p stub)."""

    def __init__(self, ctx, m):
        from models.demos.glm_5_3_flash._stubs.r_m_s_norm import TtRMSNorm
        from models.demos.glm_5_3_flash._stubs.vision_attention import TtVisionAttention
        from models.demos.glm_5_3_flash._stubs.vision_m_l_p import TtVisionMLP

        self.ctx = c = ctx
        self.norm1 = TtRMSNorm.build(c.device, m.norm1)
        self.norm2 = TtRMSNorm.build(c.device, m.norm2)
        self.attn = TtVisionAttention(c.device, m.attn)
        self.mlp = TtVisionMLP(c.device, m.mlp)

    def __call__(self, x, N, cos, sin, cu_seqlens=None, mask=None, frames=1):
        """x fp32 [1,1,N,D] replicated; cos/sin [N,hd] or [1,1,N,hd]; mask optional [1,1,N,N] additive."""
        D = x.shape[-1]
        a = self.attn(
            ttnn.reshape(self.norm1(x), [N, D]),
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
            attn_mask=mask,
            frames=frames,
        )
        h = ttnn.add(x, ttnn.typecast(ttnn.reshape(a, [1, 1, N, D]), ttnn.float32))
        return ttnn.add(h, ttnn.typecast(self.mlp(self.norm2(h)), ttnn.float32))


class TtVisionBlock:
    def __init__(self, device, m):
        self.ctx = _Ctx(device)
        self.blk = _VisionBlock(self.ctx, m)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, hidden_states, cu_seqlens=None, position_embeddings=None, attn_mask=None, frames=1, **kwargs):
        if position_embeddings is None:
            raise ValueError("vision block requires position_embeddings=(cos, sin)")
        x = hidden_states
        N, D = x.shape[-2], x.shape[-1]
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, N, D]), ttnn.float32)
        cos, sin = position_embeddings
        return ttnn.reshape(self.blk(x, N, cos, sin, cu_seqlens, mask=attn_mask, frames=frames), [N, D])


def build(device, torch_module=None):
    return TtVisionBlock.build(device, torch_module)


def vision_block(device, torch_module=None):
    return TtVisionBlock.build(device, torch_module)
