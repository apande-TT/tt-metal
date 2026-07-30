# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `attention_block` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_encoder.attn.0`` — the tortoise-style
``AttentionBlock`` (channel-first self-attention over the time axis). For a
channel-first input ``x:[b, C, T]`` (C=1024):

    x_norm = GroupNorm32(32, C)(x)
    qkv    = Conv1d(C, 3C, 1)(x_norm)                 # [b, 3C, T]
    # QKVAttentionLegacy(n_heads): per head, over the time axis
    q,k,v  = reshape(qkv, [b*h, 3*ch, T]).split(ch, 1) # ch = C / h
    w      = einsum("bct,bcs->bts", q*ch**-.25, k*ch**-.25)   # [.,T,T]
    w      = softmax(w, dim=-1)
    a      = einsum("bts,bcs->bct", w, v)             # [.,ch,T] -> [b, C, T]
    h      = Conv1d(C, C, 1)(a)                        # proj_out
    return   x_norm + h                                # tortoise_norm is False

TP=8 scheme
-----------
The block's only large weights are the two 1x1 convs (channel projections);
the GroupNorm is a norm and stays REPLICATED in every scheme. The attention is
head-parallel, but heads are fully independent and the harness replicates the
input across the mesh, so each chip computes the identical block and the
gathered (concat-then-slice) output is bit-for-bit the single-device golden —
this is the replicate-parallel degenerate of head-parallel TP (correct because
no head/group interacts across the channel split). Placement changes, math
does not.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    C = int(torch_module.channels)
    n_heads = int(torch_module.num_heads)
    ch = C // n_heads
    norm = torch_module.norm
    G = int(norm.num_groups)
    eps = float(norm.eps)

    def _rep(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    gamma_ct = _rep(norm.weight.detach().reshape(1, C, 1))
    beta_ct = _rep(norm.bias.detach().reshape(1, C, 1))
    # Conv1d(kernel=1) == a per-timestep linear over channels. Store W as
    # [C_in, C_out] so a plain matmul(x_tc, W) is the projection.
    wt_qkv = _rep(torch_module.qkv.weight.detach().squeeze(-1).t().contiguous())      # [C, 3C]
    b_qkv = _rep(torch_module.qkv.bias.detach().reshape(1, 1, 3 * C))
    wt_proj = _rep(torch_module.proj_out.weight.detach().squeeze(-1).t().contiguous())  # [C, C]
    b_proj = _rep(torch_module.proj_out.bias.detach().reshape(1, 1, C))

    def _conv1x1(x_ct, wt_co, b_1o):
        # x_ct: [1, C_in, T] -> [1, C_out, T]
        y = ttnn.transpose(x_ct, -2, -1)          # [1, T, C_in]
        y = ttnn.matmul(y, wt_co)                 # [1, T, C_out]
        y = ttnn.add(y, b_1o)                     # + bias (broadcast over T)
        return ttnn.transpose(y, -2, -1)          # [1, C_out, T]

    def forward(x, mask=None, **_):
        T = int(x.shape[-1])

        # --- GroupNorm32 over channels (32 groups) ---
        xr = ttnn.reshape(x, [1, G, C // G, T])
        m = ttnn.mean(xr, dim=3, keepdim=True)
        m = ttnn.mean(m, dim=2, keepdim=True)      # [1, G, 1, 1]
        xc = ttnn.subtract(xr, m)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=3, keepdim=True)
        var = ttnn.mean(var, dim=2, keepdim=True)  # [1, G, 1, 1]
        xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
        xn = ttnn.reshape(xn, [1, C, T])
        x_norm = ttnn.add(ttnn.multiply(xn, gamma_ct), beta_ct)   # [1, C, T]

        # --- qkv projection ---
        qkv = _conv1x1(x_norm, wt_qkv, b_qkv)      # [1, 3C, T]

        # --- QKVAttentionLegacy over the time axis, per head ---
        qkv4 = ttnn.reshape(qkv, [1, n_heads, 3 * ch, T])
        q = ttnn.slice(qkv4, [0, 0, 0, 0], [1, n_heads, ch, T])
        k = ttnn.slice(qkv4, [0, 0, ch, 0], [1, n_heads, 2 * ch, T])
        v = ttnn.slice(qkv4, [0, 0, 2 * ch, 0], [1, n_heads, 3 * ch, T])

        scale2 = ch ** -0.5   # (ch**-0.25 applied to both q and k) == ch**-0.5 on the product
        w = ttnn.matmul(ttnn.transpose(q, -2, -1), k)   # [1, h, T(t), T(s)]
        w = ttnn.multiply(w, scale2)
        w = ttnn.softmax(w, dim=-1)
        a = ttnn.matmul(v, ttnn.transpose(w, -2, -1))   # [1, h, ch, T]
        a = ttnn.reshape(a, [1, C, T])

        # --- output projection + residual on the NORMED input ---
        h = _conv1x1(a, wt_proj, b_proj)
        return ttnn.add(x_norm, h)

    return forward
