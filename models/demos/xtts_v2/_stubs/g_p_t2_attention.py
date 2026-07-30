# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t2_attention` for coqui/XTTS-v2.

HF submodule: ``gpt.gpt.h.0.attn`` — a HuggingFace ``GPT2Attention`` (embed=1024,
heads=16, head_dim=64). Unit-tested as a clean prefill (past_key_values=None,
attention_mask=None), which is plain causal multi-head self-attention:

    q, k, v = c_attn(x).split(1024, dim=2)          # c_attn: Conv1D(1024 -> 3072)
    q,k,v   = view(b, T, 16, 64).transpose(1, 2)    # [b, 16, T, 64]
    attn    = softmax(q·kᵀ · scaling + causal_mask)
    out     = attn · v -> transpose -> reshape(b, T, 1024)
    out     = c_proj(out)                            # Conv1D(1024 -> 1024)

TP=8 scheme (genuine tensor-parallel, 2 heads per chip)
-------------------------------------------------------
16 heads over TP=8 == 2 heads/chip, so the clean scheme is HEAD-parallel and
mirrors the graduated Perceiver ``attention`` stub:

  * ``c_attn`` is the fused q|k|v Conv1D(1024 -> 3072). Split it into the three
    [1024,1024] blocks Wq|Wk|Wv; within each block head ``h`` occupies output
    columns ``[h*64:(h+1)*64]`` (head-major), so a plain COLUMN split
    ``ShardTensorToMesh(dim=1)`` on the stored transpose hands chip ``i`` the
    128 columns of heads ``{2i, 2i+1}`` — column-parallel. Biases are split the
    same way (shard the feature axis).
  * Each chip runs the full scaled-dot-product attention (with the causal mask)
    on ITS 2 heads — heads never interact, so no collective is needed mid-attn.
  * The per-chip [1, T, 128] head outputs are reassembled with
    ``all_gather(dim=2)`` (device order 0..7 == head-pair order == the
    ``reshape(b, T, 1024)`` head-major layout).
  * ``c_proj`` (Conv1D 1024 -> 1024) is REPLICATED and applied to the gathered
    [1, T, 1024] on every chip, so each chip ends with the identical golden
    output. The math is unchanged; only the placement of the fused QKV
    projection differs.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    n_heads = int(torch_module.num_heads)
    head_dim = int(torch_module.head_dim)
    embed = n_heads * head_dim
    scaling = float(getattr(torch_module, "scaling", head_dim ** -0.5))

    # HF Conv1D stores weight as [in, out]; matmul(x, W) is the projection, so no
    # transpose is needed. c_attn: [1024, 3072] laid out as [q|k|v], each block
    # head-major over output columns.
    Wc = torch_module.c_attn.weight.detach()          # [1024, 3072]
    bc = torch_module.c_attn.bias.detach()            # [3072]
    Wq, Wk, Wv = Wc[:, :embed], Wc[:, embed:2 * embed], Wc[:, 2 * embed:]
    bq, bk, bv = bc[:embed], bc[embed:2 * embed], bc[2 * embed:]

    def _shard_cols(w2d):
        # Column-parallel: split the OUTPUT feature axis (dim=1) across the mesh
        # -> chip i holds heads {2i, 2i+1} projection columns.
        return ttnn.from_torch(
            w2d.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )

    def _shard_bias(b1d):
        # Split the feature axis of the bias to match the sharded columns.
        return ttnn.from_torch(
            b1d.reshape(1, 1, -1).contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2),
        )

    def _replicate(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    wt_q, wt_k, wt_v = _shard_cols(Wq), _shard_cols(Wk), _shard_cols(Wv)
    bt_q, bt_k, bt_v = _shard_bias(bq), _shard_bias(bk), _shard_bias(bv)

    wt_cproj = _replicate(torch_module.c_proj.weight.detach())          # [1024, 1024]
    b_cproj = _replicate(torch_module.c_proj.bias.detach().reshape(1, 1, -1))

    def _to_heads(t, T):
        # [1, T, hl*head_dim] -> [1, hl, T, head_dim]; hl (heads on THIS chip) is
        # read from the sharded feature width so the code follows the mesh split.
        hl = int(t.shape[-1]) // head_dim
        t = ttnn.reshape(t, [1, T, hl, head_dim])
        return ttnn.permute(t, [0, 2, 1, 3])

    def forward(x, **_):
        T = int(x.shape[-2])

        # Column-parallel fused QKV -> per chip: heads {2i,2i+1} q/k/v [1, T, 128].
        q = ttnn.add(ttnn.matmul(x, wt_q), bt_q)
        k = ttnn.add(ttnn.matmul(x, wt_k), bt_k)
        v = ttnn.add(ttnn.matmul(x, wt_v), bt_v)

        q = _to_heads(q, T)                                 # [1, hl, T, head_dim]
        k = _to_heads(k, T)
        v = _to_heads(v, T)

        # Native on-device causal flash-attention: builds the causal mask + softmax
        # internally, so no torch runs in forward. Heads are independent -> per-chip.
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scaling,
        )                                                   # [1, hl, T, head_dim]
        hl = int(ctx.shape[1])

        ctx = ttnn.permute(ctx, [0, 2, 1, 3])               # [1, T, hl, head_dim]
        ctx = ttnn.reshape(ctx, [1, T, hl * head_dim])

        # Reassemble heads across the mesh -> full [1, T, 1024], then replicated proj.
        ctx = ttnn.all_gather(ctx, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        return ttnn.add(ttnn.matmul(ctx, wt_cproj), b_cproj)  # [1, T, 1024]

    return forward
