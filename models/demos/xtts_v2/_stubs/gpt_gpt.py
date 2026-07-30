# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `gpt_gpt` for coqui/XTTS-v2.

HF submodule: ``gpt.gpt`` — a HuggingFace ``GPT2Model`` (30 x GPT2Block,
embed=1024, heads=16). It is unit-tested driven by ``inputs_embeds`` (the XTTS
positional embedding table ``wpe`` is ``null_position_embeddings`` -> zeros, and
``drop`` is identity in eval), so:

    last_hidden_state = ln_f( block_29( ... block_0(inputs_embeds) ) )

TP=8 scheme (genuine tensor-parallel, math unchanged)
-----------------------------------------------------
Every block is sharded exactly like the graduated ``g_p_t2_block`` stub:
LayerNorms (ln_1, ln_2, ln_f) REPLICATED; attention HEAD-parallel (2 heads/chip)
with column-parallel fused q|k|v, per-chip causal flash-attention, ``all_gather``
+ replicated ``c_proj``; MLP column-parallel ``c_fc`` + tanh-GELU + ``all_gather``
+ replicated ``c_proj``. 30 stacked layers compound bf16 rounding, so every
matmul runs at HiFi4 fidelity with fp32 accumulation. The gathered hidden state
equals the single-device golden.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module=None):
    transformer = torch_module
    blocks = transformer.h
    ln_f = transformer.ln_f

    attn0 = blocks[0].attn
    n_heads = int(attn0.num_heads)
    head_dim = int(attn0.head_dim)
    embed = n_heads * head_dim
    scaling = float(getattr(attn0, "scaling", head_dim ** -0.5))

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _mm(a, b):
        return ttnn.matmul(a, b, compute_kernel_config=kcfg)

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

    def _norm(mod):
        return (_rep(mod.weight.detach().reshape(1, 1, -1)),
                _rep(mod.bias.detach().reshape(1, 1, -1)),
                float(mod.eps))

    layers = []
    for blk in blocks:
        attn, mlp = blk.attn, blk.mlp
        Wc = attn.c_attn.weight.detach()          # [1024, 3072] (in, q|k|v)
        bc = attn.c_attn.bias.detach()
        layers.append({
            "ln1": _norm(blk.ln_1),
            "ln2": _norm(blk.ln_2),
            "wt_q": _shard(Wc[:, :embed], 1),
            "wt_k": _shard(Wc[:, embed:2 * embed], 1),
            "wt_v": _shard(Wc[:, 2 * embed:], 1),
            "bt_q": _shard(bc[:embed].reshape(1, 1, -1), 2),
            "bt_k": _shard(bc[embed:2 * embed].reshape(1, 1, -1), 2),
            "bt_v": _shard(bc[2 * embed:].reshape(1, 1, -1), 2),
            "wt_ao": _rep(attn.c_proj.weight.detach()),
            "b_ao": _rep(attn.c_proj.bias.detach().reshape(1, 1, -1)),
            "wt_fc": _shard(mlp.c_fc.weight.detach(), 1),
            "b_fc": _shard(mlp.c_fc.bias.detach().reshape(1, 1, -1), 2),
            "wt_mo": _rep(mlp.c_proj.weight.detach()),
            "b_mo": _rep(mlp.c_proj.bias.detach().reshape(1, 1, -1)),
        })

    ln_f_w, ln_f_b, ln_f_eps = _norm(ln_f)

    def _to_heads(t, T):
        hl = int(t.shape[-1]) // head_dim
        t = ttnn.reshape(t, [1, T, hl, head_dim])
        return ttnn.permute(t, [0, 2, 1, 3])

    def _block(x, L, T):
        h = ttnn.layer_norm(x, weight=L["ln1"][0], bias=L["ln1"][1], epsilon=L["ln1"][2])
        q = _to_heads(ttnn.add(_mm(h, L["wt_q"]), L["bt_q"]), T)
        k = _to_heads(ttnn.add(_mm(h, L["wt_k"]), L["bt_k"]), T)
        v = _to_heads(ttnn.add(_mm(h, L["wt_v"]), L["bt_v"]), T)
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scaling, compute_kernel_config=kcfg)
        hl = int(ctx.shape[1])
        ctx = ttnn.reshape(ttnn.permute(ctx, [0, 2, 1, 3]), [1, T, hl * head_dim])
        ctx = ttnn.all_gather(ctx, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        x = ttnn.add(x, ttnn.add(_mm(ctx, L["wt_ao"]), L["b_ao"]))

        h = ttnn.layer_norm(x, weight=L["ln2"][0], bias=L["ln2"][1], epsilon=L["ln2"][2])
        ff = ttnn.add(_mm(h, L["wt_fc"]), L["b_fc"])
        ff = ttnn.gelu(ff, variant=ttnn.GeluVariant.Tanh)
        ff = ttnn.all_gather(ff, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        x = ttnn.add(x, ttnn.add(_mm(ff, L["wt_mo"]), L["b_mo"]))
        return x

    def forward(inputs_embeds, *_, **__):
        x = inputs_embeds
        T = int(x.shape[-2])
        for L in layers:
            x = _block(x, L, T)
        return ttnn.layer_norm(x, weight=ln_f_w, bias=ln_f_b, epsilon=ln_f_eps)

    return forward


def gpt_gpt(device, torch_module=None):
    return build(device, torch_module)
