# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextVisionModel` (GLM-5.3-Flash vision tower).

Reference (one image / video, grid_thw = [[t, h, w]] in patches, merge m = spatial_merge_size):
    x   = patch_embed(pixels)                              Conv3d(kernel=stride=patch) == Linear
    pos = get_vision_position_ids(grid_thw, m)             block-major (h, w) per token
    cos, sin = axial rope(pos)                             [N, hd] = cat(cat(fh, fw), cat(fh, fw))
    x   = blocks(x)   (attention block-diagonal per frame, from cu_seqlens)
    x   = post_layernorm(x)
    last = downsample(x)   Conv2d(m x m, stride m) over each m*m consecutive (block-major) tokens
    pooler = merger(last)
    returns (last_hidden_state=last, pooler_output=pooler)

Everything input-dependent is computed ON DEVICE from grid_thw (uploaded): each token's (h, w)
position by closed-form index arithmetic (verified equal to HF's get_vision_position_ids, incl.
multi-frame video), the rope angles hpos * [inv|0|inv|0] + wpos * [0|inv|0|inv], and the per-frame
attention mask (segment = token // (h*w)). The downsample conv is exactly a Linear over the m*m
consecutive tokens' channels (weight permuted to (dy, dx, c) order).

Tensor parallelism: blocks = vision_block's head/column/row-parallel `_VisionBlock`; patch embed,
downsample and the merger's proj are COLUMN-parallel + all_gather; merger MLP column/row-parallel.
Norms, rope tables, residuals replicated. Multiple images packed in one call are not ported.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _Ctx, _rs
from models.demos.glm_5_3_flash._stubs.vision_patch_embed import _gather_cols

MAX_TOKENS = 16384


class TtVisionModel:
    """Composed of the graduated vision ports: vision_patch_embed -> vision_rotary_embedding (rope tables from
    on-device positions) -> vision_block x depth (each r_m_s_norm / vision_attention / vision_m_l_p) ->
    r_m_s_norm (post) -> downsample -> vision_patch_merger."""

    def __init__(self, device, m):
        from models.demos.glm_5_3_flash._stubs.r_m_s_norm import TtRMSNorm
        from models.demos.glm_5_3_flash._stubs.vision_block import TtVisionBlock
        from models.demos.glm_5_3_flash._stubs.vision_patch_embed import TtVisionPatchEmbed
        from models.demos.glm_5_3_flash._stubs.vision_patch_merger import TtVisionPatchMerger
        from models.demos.glm_5_3_flash._stubs.vision_rotary_embedding import TtVisionRotaryEmbedding

        self.ctx = c = _Ctx(device)
        cfg = m.config
        self.m = int(cfg.spatial_merge_size)
        self.patch_embed = TtVisionPatchEmbed(device, m.patch_embed)
        self.rotary_pos_emb = TtVisionRotaryEmbedding(device, m.rotary_pos_emb)
        self.blocks = [TtVisionBlock(device, b) for b in m.blocks]
        self.post_layernorm = TtRMSNorm.build(device, m.post_layernorm)
        self.tok = c.f32(torch.arange(MAX_TOKENS, dtype=torch.float32).reshape(1, 1, MAX_TOKENS, 1))
        ds = m.downsample
        if tuple(ds.kernel_size) != (self.m, self.m) or tuple(ds.stride) != (self.m, self.m):
            raise NotImplementedError("downsample conv must be m x m, stride m")
        Wd = ds.weight.detach().to(torch.float32)  # [O, C, m, m]
        self.out_dim, self.C = int(Wd.shape[0]), int(Wd.shape[1])
        if self.out_dim % c.n:
            raise ValueError("downsample out channels not divisible by mesh")
        self.w_ds = c.put(Wd.permute(2, 3, 1, 0).reshape(self.m * self.m * self.C, self.out_dim), shard_dim=-1)
        self.b_ds = c.f32(ds.bias.detach().reshape(1, 1, 1, -1), shard_dim=-1) if ds.bias is not None else None
        self.merger = TtVisionPatchMerger(device, m.merger)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    @staticmethod
    def _floor_div(a, b):
        return ttnn.floor(ttnn.divide(ttnn.add(a, 0.5), b))  # integer operands: +0.5 guards rounding

    def _positions(self, grid_thw, N):
        c, m = self.ctx, self.m
        g = (
            c.f32(grid_thw.reshape(1, 1, 1, 3))
            if isinstance(grid_thw, torch.Tensor)
            else ttnn.typecast(ttnn.to_layout(ttnn.reshape(grid_thw, [1, 1, 1, 3]), ttnn.TILE_LAYOUT), ttnn.float32)
        )
        grm = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)
        col = lambda j: ttnn.to_layout(ttnn.slice(grm, [0, 0, 0, j], [1, 1, 1, j + 1]), ttnn.TILE_LAYOUT)
        h, w = col(1), col(2)
        hw = ttnn.multiply(h, w)
        i = ttnn.to_layout(
            ttnn.slice(ttnn.to_layout(self.tok, ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0, 0], [1, 1, N, 1]), ttnn.TILE_LAYOUT
        )
        frame = self._floor_div(i, hw)  # [1,1,N,1]
        ip = ttnn.subtract(i, ttnn.multiply(frame, hw))
        b = self._floor_div(ip, float(m * m))
        inside = ttnn.subtract(ip, ttnn.multiply(b, float(m * m)))
        bw = ttnn.multiply(w, 1.0 / m)
        br = self._floor_div(b, bw)
        bc = ttnn.subtract(b, ttnn.multiply(br, bw))
        dy = self._floor_div(inside, float(m))
        dx = ttnn.subtract(inside, ttnn.multiply(dy, float(m)))
        hp = ttnn.add(ttnn.multiply(br, float(m)), dy)
        wp = ttnn.add(ttnn.multiply(bc, float(m)), dx)
        return hp, wp, frame

    def __call__(self, hidden_states, grid_thw=None, frames=1, **kwargs):
        """hidden_states [N, C*T*P*P] patches; grid_thw [1,3] = (frames, h, w) (one image, or `frames`
        same-size images packed as frames: attention is block-diagonal per frame, exactly as separate
        images). Returns (last_hidden_state [R, out], pooler_output [R, out])."""
        if grid_thw is None:
            raise ValueError("vision model requires grid_thw")
        if grid_thw.shape[0] != 1:
            raise NotImplementedError("pack same-size images as frames of one grid row")
        c = self.ctx
        x = hidden_states
        N, K = x.shape[-2], x.shape[-1]
        if N > MAX_TOKENS:
            raise NotImplementedError(f"{N} tokens > {MAX_TOKENS}")
        x = self.patch_embed(x)  # [N,C] fp32 (graduated vision_patch_embed)

        hp, wp, frame = self._positions(grid_thw, N)
        pos = ttnn.concat(
            [ttnn.to_layout(hp, ttnn.ROW_MAJOR_LAYOUT), ttnn.to_layout(wp, ttnn.ROW_MAJOR_LAYOUT)], dim=-1
        )
        cos, sin = self.rotary_pos_emb(x, position_ids=ttnn.reshape(pos, [N, 2]))  # graduated vision rope
        for blk in self.blocks:  # per-frame attention: equal-size frames ride the batch axis
            x = blk(x, position_embeddings=(cos, sin), frames=int(frames))  # [N,C] fp32

        x = self.post_layernorm(ttnn.reshape(x, [1, 1, N, self.C]))
        R = N // (self.m * self.m)
        xg = _rs(ttnn.typecast(x, ttnn.float32), [1, 1, R, self.m * self.m * self.C])
        part = c.lin_hp(xg, self.w_ds)
        if self.b_ds is not None:
            part = ttnn.add(part, self.b_ds)
        last = _gather_cols(c, part)  # [1,1,R,out]
        pooled = self.merger(ttnn.reshape(last, [R, self.out_dim]))  # graduated vision_patch_merger
        return ttnn.reshape(last, [R, self.out_dim]), ttnn.reshape(pooled, [R, self.out_dim])


def build(device, torch_module=None):
    return TtVisionModel.build(device, torch_module)


def vision_model(device, torch_module=None):
    return TtVisionModel.build(device, torch_module)
