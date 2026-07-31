# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `conditioning_encoder` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_encoder`` — a ``ConditioningEncoder``:

    h = Conv1d(80, 1024, kernel=1)(x)        # x: [b, 80, T]
    h = Sequential(AttentionBlock(1024, heads) x 6)(h)
    return h                                  # [b, 1024, T]

Each ``AttentionBlock`` is the tortoise-style channel-first self-attention over
the time axis (GroupNorm32 -> qkv 1x1 conv -> per-head time attention ->
proj_out 1x1 conv -> ``x_norm + h`` residual); see ``_stubs/attention_block.py``
for the derivation.

TP=8 scheme
-----------
The only large weights are 1x1 conv channel projections; the GroupNorms stay
replicated in every scheme, and the per-head attention is head-independent.
The harness replicates the input across the mesh, so each chip runs the
identical encoder and the gathered (concat-then-slice) output is bit-for-bit
the single-device golden — the replicate-parallel degenerate of head-parallel
TP. Placement changes, math does not.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    def _rep(t):
        # A bias / norm scale has logical height 1, so a DEVICE tilize has to val-pad it
        # 1 -> 32 rows, which ttnn runs on a SINGLE core -- the profile's grid=tiny
        # TilizeDeviceOperation, hundreds of calls for a few KB each. Tilizing those on the
        # HOST is free by comparison. Real matrices are already tile-shaped (no val padding)
        # and keep the multicore device path, where host-tilizing megabytes is the worse trade.
        if t.dim() >= 2 and int(t.shape[-2]) == 1:
            return ttnn.to_device(
                ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(device)),
                device)
        return ttnn.from_torch(
            t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    # init Conv1d(80 -> 1024, kernel=1): stored as [C_in, C_out] for matmul.
    init = torch_module.init
    wt_init = _rep(init.weight.detach().squeeze(-1).t().contiguous())
    b_init = _rep(init.bias.detach().reshape(1, 1, init.out_channels))

    # Pre-extract every AttentionBlock's replicated weights + shape metadata.
    blocks = []
    for blk in torch_module.attn:
        C = int(blk.channels)
        n_heads = int(blk.num_heads)
        norm = blk.norm
        blocks.append(
            {
                "C": C,
                "n_heads": n_heads,
                "ch": C // n_heads,
                "G": int(norm.num_groups),
                "eps": float(norm.eps),
                "gamma": _rep(norm.weight.detach().reshape(1, C, 1)),
                "beta": _rep(norm.bias.detach().reshape(1, C, 1)),
                "wt_qkv": _rep(blk.qkv.weight.detach().squeeze(-1).t().contiguous()),
                "b_qkv": _rep(blk.qkv.bias.detach().reshape(1, 1, 3 * C)),
                "wt_proj": _rep(blk.proj_out.weight.detach().squeeze(-1).t().contiguous()),
                "b_proj": _rep(blk.proj_out.bias.detach().reshape(1, 1, C)),
            }
        )

    def _conv1x1(x_ct, wt_co, b_1o):
        y = ttnn.transpose(x_ct, -2, -1)      # [1, T, C_in]
        y = ttnn.matmul(y, wt_co)             # [1, T, C_out]
        y = ttnn.add(y, b_1o)
        return ttnn.transpose(y, -2, -1)      # [1, C_out, T]

    def _attn_block(h, p):
        C, n_heads, ch, G, eps = p["C"], p["n_heads"], p["ch"], p["G"], p["eps"]
        T = int(h.shape[-1])

        # GroupNorm32 over channels.
        xr = ttnn.reshape(h, [1, G, C // G, T])
        m = ttnn.mean(ttnn.mean(xr, dim=3, keepdim=True), dim=2, keepdim=True)
        xc = ttnn.subtract(xr, m)
        var = ttnn.mean(ttnn.mean(ttnn.multiply(xc, xc), dim=3, keepdim=True), dim=2, keepdim=True)
        xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
        xn = ttnn.reshape(xn, [1, C, T])
        x_norm = ttnn.add(ttnn.multiply(xn, p["gamma"]), p["beta"])

        # qkv projection + per-head time attention.
        qkv = _conv1x1(x_norm, p["wt_qkv"], p["b_qkv"])
        qkv4 = ttnn.reshape(qkv, [1, n_heads, 3 * ch, T])
        q = ttnn.slice(qkv4, [0, 0, 0, 0], [1, n_heads, ch, T])
        k = ttnn.slice(qkv4, [0, 0, ch, 0], [1, n_heads, 2 * ch, T])
        v = ttnn.slice(qkv4, [0, 0, 2 * ch, 0], [1, n_heads, 3 * ch, T])
        w = ttnn.multiply(ttnn.matmul(ttnn.transpose(q, -2, -1), k), ch ** -0.5)
        w = ttnn.softmax(w, dim=-1)
        a = ttnn.matmul(v, ttnn.transpose(w, -2, -1))   # [1, h, ch, T]
        a = ttnn.reshape(a, [1, C, T])

        h2 = _conv1x1(a, p["wt_proj"], p["b_proj"])
        return ttnn.add(x_norm, h2)

    def forward(x, **_):
        h = _conv1x1(x, wt_init, b_init)   # [1, 1024, T]
        for p in blocks:
            h = _attn_block(h, p)
        return h

    return forward
