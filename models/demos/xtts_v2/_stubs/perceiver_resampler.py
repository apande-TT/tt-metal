# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `gpt_conditioning_perceiver` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_perceiver`` — a lucidrains ``PerceiverResampler``
(dim=1024, 32 learned latents, 2 layers, heads=8, dim_head=64):

    latents = repeat(self.latents, "n d -> b n d")        # [b, 32, 1024]
    for attn, ff in self.layers:                          # 2 x
        latents = attn(latents, context=x) + latents      # cross-attention
        latents = ff(latents) + latents                   # GEGLU feed-forward
    return self.norm(latents)                             # RMSNorm

Each ``attn`` is the graduated Perceiver ``attention`` (q from latents, k/v from
the context); ``ff`` is Linear(1024->5460) -> GEGLU(->2730) -> Linear(2730->1024);
``norm`` is RMSNorm (L2-normalize over the feature dim, * scale(=sqrt(dim)=32) *
gamma).

TP=8 scheme (genuine tensor-parallel, math unchanged)
-----------------------------------------------------
The cross-attention is HEAD-parallel (heads == TP == 8, 1 head/chip), exactly
like the graduated ``attention`` stub: ``to_q``/``to_k``/``to_v`` are
COLUMN-parallel (``ShardTensorToMesh(dim=1)`` -> chip i computes head i), the
per-head outputs are reassembled with ``all_gather(dim=2)``, and the replicated
``to_out`` yields the identical full output. The GEGLU feed-forward's hidden dim
(5460 / gated 2730) is not divisible by TP=8, so it stays REPLICATED (as do the
latents, gamma, and norm) — every chip computes the identical residual-updated
latents, so the gathered output equals the single-device golden.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    latents_p = torch_module.latents.detach()             # [32, 1024]
    n_latents, dim = int(latents_p.shape[0]), int(latents_p.shape[1])

    def _rep(t):
        # A bias / norm scale has logical height 1, so a DEVICE tilize has to val-pad it
        # 1 -> 32 rows, which ttnn runs on a SINGLE core -- the profile's grid=tiny
        # TilizeDeviceOperation, hundreds of calls for a few KB each. Tilizing those on the
        # HOST is free by comparison. Real matrices are already tile-shaped (no val padding)
        # and keep the multicore device path, where host-tilizing megabytes is the worse trade.
        if t.dim() >= 2 and int(t.shape[-2]) == 1:
            return ttnn.to_device(
                ttnn.from_torch(t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(device)),
                device)
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def _shard_cols(w2d):
        # Linear computes x @ W^T; store [in, out] transpose and column-split it.
        return ttnn.from_torch(
            w2d.t().contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )

    def _rep_wt(w2d):
        return _rep(w2d.t().contiguous())

    latents = _rep(latents_p.reshape(1, n_latents, dim))

    layers = []
    for attn, ff in torch_module.layers:
        Wq = attn.to_q.weight.detach()                    # [inner, dim]
        Wkv = attn.to_kv.weight.detach()                  # [2*inner, dim]
        Wo = attn.to_out.weight.detach()                  # [dim, inner]
        inner = int(Wq.shape[0])
        heads = int(attn.heads)
        scale = float(getattr(attn, "scale", (inner // heads) ** -0.5))
        lin1, geglu, lin2 = ff[0], ff[1], ff[2]
        layers.append({
            "scale": scale,
            "wt_q": _shard_cols(Wq),
            "wt_k": _shard_cols(Wkv[:inner]),
            "wt_v": _shard_cols(Wkv[inner:]),
            "wt_o": _rep_wt(Wo),
            "wt_ff1": _rep_wt(lin1.weight.detach()),
            "b_ff1": _rep(lin1.bias.detach().reshape(1, 1, -1)),
            "wt_ff2": _rep_wt(lin2.weight.detach()),
            "b_ff2": _rep(lin2.bias.detach().reshape(1, 1, -1)),
        })

    nm = torch_module.norm
    rms_scale = float(nm.scale)
    rms_gamma = _rep(nm.gamma.detach().reshape(1, 1, -1)) if nm.gamma is not None else None

    def _attn(latents_t, ctx, L):
        # cross_attn_include_queries=True: keys/values attend over the latents
        # concatenated with the context along the sequence axis.
        kv_ctx = ttnn.concat([latents_t, ctx], dim=1)     # [1, n+S, dim]
        q = ttnn.matmul(latents_t, L["wt_q"])             # [1, n, head_dim] head i
        k = ttnn.matmul(kv_ctx, L["wt_k"])                # [1, n+S, head_dim]
        v = ttnn.matmul(kv_ctx, L["wt_v"])
        sim = ttnn.multiply(ttnn.matmul(q, ttnn.transpose(k, -2, -1)), L["scale"])
        a = ttnn.softmax(sim, dim=-1)
        out = ttnn.matmul(a, v)                           # [1, n, head_dim]
        out = ttnn.all_gather(out, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        return ttnn.matmul(out, L["wt_o"])                # [1, n, dim]

    def _ff(x, L):
        h = ttnn.add(ttnn.matmul(x, L["wt_ff1"]), L["b_ff1"])   # [1, n, 5460]
        half = int(h.shape[-1]) // 2
        n = int(h.shape[-2])
        a = ttnn.slice(h, [0, 0, 0], [1, n, half])
        g = ttnn.slice(h, [0, 0, half], [1, n, 2 * half])
        h = ttnn.multiply(a, ttnn.gelu(g, variant=ttnn.GeluVariant.Accurate))
        return ttnn.add(ttnn.matmul(h, L["wt_ff2"]), L["b_ff2"])  # [1, n, dim]

    def forward(x, mask=None, **_):
        lat = latents
        for L in layers:
            lat = ttnn.add(_attn(lat, x, L), lat)
            lat = ttnn.add(_ff(lat, L), lat)
        # RMSNorm: L2-normalize over the feature dim, * scale * gamma.
        sq = ttnn.sum(ttnn.multiply(lat, lat), dim=-1, keepdim=True)
        xn = ttnn.multiply(lat, ttnn.rsqrt(ttnn.add(sq, 1e-12)))
        out = ttnn.multiply(xn, rms_scale)
        if rms_gamma is not None:
            out = ttnn.multiply(out, rms_gamma)
        return out

    return forward
