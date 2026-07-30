# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `attention` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_perceiver.layers.0.0`` — the lucidrains-style
Perceiver ``Attention`` (dim=1024, heads=8, dim_head=64, dim_inner=512,
``causal=False``, no bias). Forward:

    q, k, v = to_q(x), *to_kv(context).chunk(2, -1)   # context defaults to x
    q,k,v   = rearrange(., "b n (h d) -> b h n d")
    out     = softmax(q·kᵀ · d**-0.5) · v
    return    to_out(rearrange(out, "b h n d -> b n (h d)"))

with weights ``to_q:[512,1024]``, ``to_kv:[1024,1024]``, ``to_out:[1024,512]``.

TP=8 scheme (genuine tensor-parallel, one head per chip)
--------------------------------------------------------
heads == TP == 8, so the clean scheme is HEAD-parallel:

  * ``to_q`` / ``to_k`` / ``to_v`` are COLUMN-parallel — their output feature
    axis is split across the 8 chips (``ShardTensorToMesh(dim=1)`` on the stored
    transposed weight). Because ``rearrange("b n (h d)")`` lays head ``h`` in
    output features ``[h*64:(h+1)*64]``, a 64-wide column slice IS exactly one
    head, so chip ``i`` computes head ``i``'s q/k/v from the replicated input.
  * Each chip runs the full single-head scaled-dot-product attention on its own
    head — heads never interact, so no collective is needed mid-attention.
  * The per-head outputs are reassembled with ``all_gather(dim=2)`` (device
    order 0..7 == head order 0..7 == ``rearrange("b h n d -> b n (h d)")``).
  * ``to_out`` is REPLICATED and applied to the gathered ``[b, n, 512]`` on every
    chip, so each chip ends with the identical full ``[b, n, 1024]`` — the
    gathered (concat-then-slice) output equals the single-device golden.

The math is unchanged; only the placement of the per-head projections differs.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    heads = int(torch_module.heads)
    Wq = torch_module.to_q.weight.detach()      # [inner, dim]
    Wkv = torch_module.to_kv.weight.detach()     # [2*inner, dim]
    Wo = torch_module.to_out.weight.detach()     # [dim, inner]
    inner = int(Wq.shape[0])
    Wk = Wkv[:inner]
    Wv = Wkv[inner:]

    def _to_wt(w2d):
        # Linear computes x @ W^T; store the transpose [in, out] so a plain
        # ttnn.matmul(x, wt) is the projection.
        return w2d.t().contiguous().to(torch.bfloat16)

    def _shard_cols(w2d):
        # Column-parallel: split the OUTPUT feature axis (dim=1 of the transpose)
        # across the mesh -> chip i holds head i's projection columns.
        return ttnn.from_torch(
            _to_wt(w2d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )

    def _replicate(w2d):
        return ttnn.from_torch(
            _to_wt(w2d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    wt_q = _shard_cols(Wq)   # [dim, inner] -> per chip [dim, head_dim]
    wt_k = _shard_cols(Wk)
    wt_v = _shard_cols(Wv)
    wt_o = _replicate(Wo)    # [inner, dim] replicated

    def forward(x, context=None, mask=None, **_):
        ctx = x if context is None else context
        if not isinstance(ctx, ttnn.Tensor):
            ctx = ttnn.from_torch(
                ctx.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

        # Column-parallel projections -> per chip: head i's q/k/v [1, seq, head_dim].
        q = ttnn.matmul(x, wt_q)
        k = ttnn.matmul(ctx, wt_k)
        v = ttnn.matmul(ctx, wt_v)

        # Single-head scaled dot-product attention (matches the Attend submodule,
        # which scales by q.shape[-1] ** -0.5 == head_dim ** -0.5).
        scale = q.shape[-1] ** -0.5
        sim = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        sim = ttnn.multiply(sim, scale)
        attn = ttnn.softmax(sim, dim=-1)
        out = ttnn.matmul(attn, v)  # [1, seq, head_dim] (head i)

        # Reassemble heads across the mesh, then the replicated output projection.
        out = ttnn.all_gather(out, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        out = ttnn.matmul(out, wt_o)  # [1, seq, dim]
        return out

    return forward
