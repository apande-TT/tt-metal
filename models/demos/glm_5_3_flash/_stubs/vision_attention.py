# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, head-parallel port of `Glm5NextVisionAttention` (GLM-5.3-Flash vision tower).

Reference, packed tokens x [N, D], cu_seqlens image boundaries, (cos, sin) [N, hd]:
    q, k, v = split(x @ Wqkv + b)  per head        (qkv layout [3, H, hd])
    q, k    = RMSNorm_hd(q), RMSNorm_hd(k)         (fp32, per-head norms)
    q, k    = q*cos + rotate_half(q)*sin, ...       (fp32)
    o       = softmax(q k^T * hd^-0.5 + block-diag image mask) v   (bidirectional)
    out     = o @ Wproj + b

Tensor parallelism over N chips with H heads:
  * qkv COLUMN-parallel by head. If N > H (TP=32, H=16) heads cannot be split further -- slicing a
    head's dims would break its q/k RMSNorm -- so each head is computed by r = N/H chips
    (redundantly); the qkv columns are laid out chip-major so the plain dim=-1 shard gives chip d
    head d // r.
  * proj ROW-parallel; each chip's rows are scaled by 1/r (exact for r a power of two) so the
    all-reduce over the r copies sums to exactly one contribution; the proj bias is added once,
    after the reduction. All-reduce = all_gather + on-device sum (padded to >= 64 tiles per chip:
    all_gather deadlocks on the 1x32 Galaxy line below that).
  * q/k norms, rotary tables, biases-after-reduce: replicated.
Multiple images: the block-diagonal mask is built on device from cu_seqlens (segment id of token i
= number of boundaries <= i), against a token-index table built once in build().
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _Ctx
from models.demos.glm_5_3_flash._stubs.text_indexer import row_chunks

MAX_TOKENS = 16384


def _padded_all_reduce(ctx, x):
    """Sum [1,1,R,C] partials over the mesh -> replicated (bf16 transport, fp32 sum). The gathered
    [1,n,rows,C] tensor is formed over row chunks (rows are independent), so its size does not grow with R."""
    if ctx.n == 1:
        return x
    R, C = x.shape[-2], x.shape[-1]
    xb = ttnn.typecast(x, ttnn.bfloat16)
    spans = row_chunks(R, ctx.n * max(C, 64), align=32)
    parts = [
        _all_reduce_rows(ctx, xb if len(spans) == 1 else ttnn.slice(xb, [0, 0, r0, 0], [1, 1, r1, C]), r1 - r0, C)
        for r0, r1 in spans
    ]
    return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)


def _all_reduce_rows(ctx, xb, R, C):
    row_tiles = max(1, -(-R // 32))
    cp = max(C, -(-64 // row_tiles) * 32)
    if cp > C:
        xb = ttnn.pad(xb, [(0, 0), (0, 0), (0, 0), (0, cp - C)], 0.0)
    g = ttnn.all_gather(xb, dim=1, cluster_axis=ctx.axis, topology=ttnn.Topology.Linear)
    s = ttnn.sum(ttnn.typecast(g, ttnn.float32), dim=1, keepdim=True)
    if cp > C:
        s = ttnn.to_layout(
            ttnn.slice(ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0, 0], [1, 1, R, C]), ttnn.TILE_LAYOUT
        )
    return s


def _rms_hd(x, w, eps):
    var = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
    return ttnn.multiply(ttnn.multiply(x, ttnn.rsqrt(ttnn.add(var, eps))), w)


class _VisionAttn:
    def __init__(self, ctx, m):
        self.ctx = c = ctx
        n = c.n
        H, D = int(m.num_heads), int(m.dim)
        hd = D // H
        self.H, self.D, self.hd = H, D, hd
        if n >= H:
            if n % H:
                raise ValueError(f"mesh {n} not a multiple of heads {H}")
            self.r, self.hl = n // H, 1
        else:
            if H % n:
                raise ValueError(f"heads {H} not divisible by mesh {n}")
            self.r, self.hl = 1, H // n
        heads_of = lambda d: [(d // self.r) * self.hl + j for j in range(self.hl)]
        Wqkv = m.qkv.weight.detach().to(torch.float32)  # [3*D, D], rows = (3, H, hd)
        bqkv = m.qkv.bias.detach().to(torch.float32) if m.qkv.bias is not None else torch.zeros(3 * D)
        Wv = Wqkv.view(3, H, hd, D)
        bv = bqkv.view(3, H, hd)
        # chip block d = [q heads | k heads | v heads] of heads_of(d): gather (part, head) blocks in that order
        sel = [(part, h) for d in range(n) for part in range(3) for h in heads_of(d)]
        p_idx = torch.tensor([p for p, _ in sel])
        h_idx = torch.tensor([h for _, h in sel])
        self.w_qkv = c.put(Wv[p_idx, h_idx].reshape(-1, D).t(), shard_dim=-1)  # [D, n*3*hl*hd]
        self.b_qkv = c.f32(bv[p_idx, h_idx].reshape(1, 1, 1, -1), shard_dim=-1)
        Wp = m.proj.weight.detach().to(torch.float32).t()  # [D_in (head-major), D]
        heads = torch.tensor([h for d in range(n) for h in heads_of(d)])
        self.w_proj = c.put(Wp.view(H, hd, D)[heads].reshape(-1, D) / self.r, shard_dim=0)  # 1/r exact (power of two)
        self.b_proj = c.f32(m.proj.bias.detach().reshape(1, 1, 1, D)) if m.proj.bias is not None else None
        self.qn_w = c.f32(m.q_norm.weight.detach().reshape(1, 1, 1, hd))
        self.kn_w = c.f32(m.k_norm.weight.detach().reshape(1, 1, 1, hd))
        self.q_eps = float(m.q_norm.variance_epsilon)
        self.k_eps = float(m.k_norm.variance_epsilon)
        self.scale = float(m.scaling)
        self.tok = c.f32(torch.arange(MAX_TOKENS, dtype=torch.float32).reshape(1, 1, MAX_TOKENS, 1))

    def _heads(self, x, part, N):
        hd, hl = self.hd, self.hl
        base = part * hl * hd
        hs = [ttnn.slice(x, [0, 0, 0, base + j * hd], [1, 1, N, base + (j + 1) * hd]) for j in range(hl)]
        return hs[0] if hl == 1 else ttnn.concat(hs, dim=1)  # [1,hl,N,hd]

    def _rot(self, x, cos, sin):
        h = self.hd // 2
        s = list(x.shape)
        x1 = ttnn.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], h])
        x2 = ttnn.slice(x, [0, 0, 0, h], [s[0], s[1], s[2], self.hd])
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))

    def _upload(self, t, shape):
        if isinstance(t, torch.Tensor):
            return self.ctx.f32(t.reshape(shape))
        if t.layout != ttnn.TILE_LAYOUT:
            t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
        return ttnn.typecast(ttnn.reshape(t, shape), ttnn.float32)

    def _mask(self, cu_seqlens, N):
        if cu_seqlens is None:
            return None
        K = cu_seqlens.shape[0]
        if K <= 2:  # one image: full bidirectional attention, no mask
            return None
        if N > MAX_TOKENS:
            raise NotImplementedError(f"{N} tokens > {MAX_TOKENS}")
        cu = self._upload(cu_seqlens, [1, 1, 1, K])
        tok = ttnn.to_layout(
            ttnn.slice(ttnn.to_layout(self.tok, ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0, 0], [1, 1, N, 1]), ttnn.TILE_LAYOUT
        )
        seg = ttnn.sum(ttnn.ge(tok, cu), dim=-1, keepdim=True)  # [1,1,N,1]
        same = ttnn.eq(seg, ttnn.transpose(seg, -2, -1))  # [1,1,N,N]
        return ttnn.multiply(ttnn.subtract(same, 1.0), 1e9)

    def _frames(self, t, F):
        """[1,hl,N,hd] -> [F,hl,N/F,hd] (equal-size images on the batch axis)."""
        if F == 1:
            return t
        hl, N, hd = t.shape[1], t.shape[2], t.shape[3]
        r = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [hl, F, N // F, hd])
        return ttnn.to_layout(ttnn.permute(r, (1, 0, 2, 3)), ttnn.TILE_LAYOUT)

    def _unframes(self, t, F):
        if F == 1:
            return t
        Fr, hl, n, hd = t.shape
        r = ttnn.permute(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), (1, 0, 2, 3))
        return ttnn.to_layout(ttnn.reshape(r, [1, hl, Fr * n, hd]), ttnn.TILE_LAYOUT)

    def __call__(self, x, N, cos, sin, cu_seqlens=None, mask=None, frames=1):
        """x: fp32 [1,1,N,D] replicated; cos/sin fp32 [1,1,N,hd]. Attention is per image: ``frames`` equal-size
        images run on the batch axis (exact, no mask); else an optional prebuilt [1,1,N,N] additive mask or
        one built from cu_seqlens. Returns fp32 [1,1,N,D] replicated."""
        c = self.ctx
        F = int(frames)
        qkv = ttnn.add(c.lin_hp(x, self.w_qkv), self.b_qkv)  # [1,1,N,3*hl*hd]
        q = self._frames(self._rot(_rms_hd(self._heads(qkv, 0, N), self.qn_w, self.q_eps), cos, sin), F)
        k = self._frames(self._rot(_rms_hd(self._heads(qkv, 1, N), self.kn_w, self.k_eps), cos, sin), F)
        v = self._frames(self._heads(qkv, 2, N), F)
        if F == 1 and mask is None:
            mask = self._mask(cu_seqlens, N)
        if F > 1:
            mask = None
        kT = ttnn.transpose(k, -2, -1)
        n = q.shape[2]
        outs = []  # softmax is per query row, so the [F,hl,rows,n] scores are formed one row chunk at a time
        for s0, s1 in row_chunks(n, q.shape[0] * q.shape[1] * (-(-n // 32) * 32), align=32):
            whole = (s0, s1) == (0, n)
            qc = q if whole else ttnn.slice(q, [0, 0, s0, 0], [q.shape[0], q.shape[1], s1, self.hd])
            sc = ttnn.multiply(c.mm(qc, kT), self.scale)
            if mask is not None:
                sc = ttnn.add(sc, mask if whole else ttnn.slice(mask, [0, 0, s0, 0], [1, 1, s1, n]))
            outs.append(c.mm(ttnn.softmax(sc, dim=-1, numeric_stable=True), v))
        o = self._unframes(outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2), F)  # [1,hl,N,hd]
        if self.hl > 1:
            o = ttnn.concat([ttnn.slice(o, [0, j, 0, 0], [1, j + 1, N, self.hd]) for j in range(self.hl)], dim=-1)
        part = c.lin_hp(o, self.w_proj)  # [1,1,N,D] partial
        out = _padded_all_reduce(c, part)
        return ttnn.add(out, self.b_proj) if self.b_proj is not None else out


class TtVisionAttention:
    def __init__(self, device, m):
        self.ctx = _Ctx(device)
        self.attn = _VisionAttn(self.ctx, m)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(
        self,
        hidden_states,
        cu_seqlens=None,
        position_embeddings=None,
        max_seqlen=None,
        attn_mask=None,
        frames=1,
        **kwargs,
    ):
        if position_embeddings is None:
            raise ValueError("vision attention requires position_embeddings=(cos, sin)")
        x = hidden_states
        N, D = x.shape[-2], x.shape[-1]
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, N, D]), ttnn.float32)
        hd = self.attn.hd
        cos, sin = (self.attn._upload(t, [1, 1, N, hd]) for t in position_embeddings)
        out = self.attn(x, N, cos, sin, cu_seqlens, mask=attn_mask, frames=frames)
        return ttnn.reshape(out, [N, D])


def build(device, torch_module=None):
    return TtVisionAttention.build(device, torch_module)


def vision_attention(device, torch_module=None):
    return TtVisionAttention.build(device, torch_module)
