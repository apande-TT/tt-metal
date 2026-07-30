# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t2_block` for coqui/XTTS-v2.

HF submodule: ``gpt.gpt.h.0`` — a HuggingFace ``GPT2Block`` (embed=1024,
heads=16, head_dim=64, mlp inner=4096). Unit-tested as a clean prefill
(layer_past empty, attention_mask=None), the pre-LN transformer block:

    h = x + attn(ln_1(x))          # causal multi-head self-attention
    y = h + mlp(ln_2(h))           # c_fc -> gelu(new/tanh) -> c_proj

TP=8 scheme (genuine tensor-parallel, math unchanged)
-----------------------------------------------------
LayerNorms (ln_1, ln_2) are elementwise + reduce over the FULL hidden dim, so
they stay REPLICATED (the block's input arrives replicated across the mesh, so
every chip normalizes the identical row).

Attention — head-parallel, 2 heads/chip (mirrors the graduated ``g_p_t2_attention``
and ``attention`` stubs):
  * fused ``c_attn`` (Conv1D 1024->3072) split into q|k|v [1024,1024] blocks;
    each block is head-major over its output columns, so a COLUMN split
    ``ShardTensorToMesh(dim=1)`` hands chip ``i`` heads {2i,2i+1}. Biases split
    the same way.
  * per-chip on-device causal flash-attention over its 2 heads (no collective
    mid-attention — heads are independent);
  * ``all_gather(dim=2)`` reassembles heads to [1,T,1024]; the REPLICATED
    ``c_proj`` (Conv1D 1024->1024) then yields the identical full output.

MLP — column-then-gather (mirrors ``g_e_g_l_u``):
  * ``c_fc`` (Conv1D 1024->4096) is COLUMN-parallel (``ShardTensorToMesh(dim=1)``);
    each chip computes 512 of the 4096 hidden features + tanh-GELU locally;
  * ``all_gather(dim=2)`` reassembles the 4096 hidden dim; the REPLICATED
    ``c_proj`` (Conv1D 4096->1024) projects back.

Only the placement of the two large projections changes; the gathered output
equals the single-device golden.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    attn = torch_module.attn
    mlp = torch_module.mlp
    ln_1, ln_2 = torch_module.ln_1, torch_module.ln_2

    n_heads = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    embed = n_heads * head_dim
    scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
    ln1_eps = float(ln_1.eps)
    ln2_eps = float(ln_2.eps)

    def _rep(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def _shard(t, dim):
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
        )

    # --- LayerNorm weights (replicated) ---
    ln1_w = _rep(ln_1.weight.detach().reshape(1, 1, -1))
    ln1_b = _rep(ln_1.bias.detach().reshape(1, 1, -1))
    ln2_w = _rep(ln_2.weight.detach().reshape(1, 1, -1))
    ln2_b = _rep(ln_2.bias.detach().reshape(1, 1, -1))

    # --- Attention weights: fused c_attn split into head-major q|k|v blocks ---
    Wc = attn.c_attn.weight.detach()          # [1024, 3072] = [in, out(q|k|v)]
    bc = attn.c_attn.bias.detach()            # [3072]
    wt_q = _shard(Wc[:, :embed], dim=1)                 # column-parallel
    wt_k = _shard(Wc[:, embed:2 * embed], dim=1)
    wt_v = _shard(Wc[:, 2 * embed:], dim=1)
    bt_q = _shard(bc[:embed].reshape(1, 1, -1), dim=2)
    bt_k = _shard(bc[embed:2 * embed].reshape(1, 1, -1), dim=2)
    bt_v = _shard(bc[2 * embed:].reshape(1, 1, -1), dim=2)
    wt_attn_o = _rep(attn.c_proj.weight.detach())       # [1024, 1024] replicated
    b_attn_o = _rep(attn.c_proj.bias.detach().reshape(1, 1, -1))

    # --- MLP weights: c_fc column-parallel, c_proj replicated ---
    wt_fc = _shard(mlp.c_fc.weight.detach(), dim=1)     # [1024, 4096] col-parallel
    b_fc = _shard(mlp.c_fc.bias.detach().reshape(1, 1, -1), dim=2)
    wt_mlp_o = _rep(mlp.c_proj.weight.detach())         # [4096, 1024] replicated
    b_mlp_o = _rep(mlp.c_proj.bias.detach().reshape(1, 1, -1))

    def _to_heads(t, T):
        hl = int(t.shape[-1]) // head_dim               # heads on THIS chip
        t = ttnn.reshape(t, [1, T, hl, head_dim])
        return ttnn.permute(t, [0, 2, 1, 3])            # [1, hl, T, head_dim]

    def _attn(h, T):
        q = _to_heads(ttnn.add(ttnn.matmul(h, wt_q), bt_q), T)
        k = _to_heads(ttnn.add(ttnn.matmul(h, wt_k), bt_k), T)
        v = _to_heads(ttnn.add(ttnn.matmul(h, wt_v), bt_v), T)

        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scaling,
        )                                               # [1, hl, T, head_dim]
        hl = int(ctx.shape[1])
        ctx = ttnn.permute(ctx, [0, 2, 1, 3])           # [1, T, hl, head_dim]
        ctx = ttnn.reshape(ctx, [1, T, hl * head_dim])

        ctx = ttnn.all_gather(ctx, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        return ttnn.add(ttnn.matmul(ctx, wt_attn_o), b_attn_o)   # [1, T, 1024]

    def _mlp(h):
        ff = ttnn.add(ttnn.matmul(h, wt_fc), b_fc)      # [1, T, 512] per chip
        ff = ttnn.gelu(ff, variant=ttnn.GeluVariant.Tanh)
        ff = ttnn.all_gather(ff, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        return ttnn.add(ttnn.matmul(ff, wt_mlp_o), b_mlp_o)      # [1, T, 1024]

    def forward(x, *_, **__):
        T = int(x.shape[-2])

        h = ttnn.layer_norm(x, weight=ln1_w, bias=ln1_b, epsilon=ln1_eps)
        x = ttnn.add(x, _attn(h, T))

        h = ttnn.layer_norm(x, weight=ln2_w, bias=ln2_b, epsilon=ln2_eps)
        x = ttnn.add(x, _mlp(h))
        return x

    return forward
