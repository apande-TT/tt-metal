# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t2_inference_model` for coqui/XTTS-v2.

HF submodule: ``gpt.gpt_inference`` — the XTTS ``GPT2InferenceModel`` wrapper.
Its forward embeds the token ids, runs a 30-layer GPT2 transformer stack, and
projects to logits with a two-LayerNorm lm_head:

    emb    = concat(cached_prefix_emb, embeddings(gen_ids) + pos_embedding)
    hidden = transformer(inputs_embeds=emb)          # 30 x GPT2Block, then ln_f
    logits = lm_head(hidden) = linear(final_norm(hidden))   # [b, T, vocab]

The token->vector embedding is a table gather (not tensor-parallel compute); the
PCC harness therefore feeds the module's own ``inputs_embeds`` (``emb``) as this
stub's primary input, and this stub natively reproduces the TP-relevant compute:
the 30 transformer blocks + ``ln_f`` + the ``final_norm``/linear lm_head. The
golden remains the FULL module forward over the token ids.

TP=8 scheme (genuine tensor-parallel, math unchanged)
-----------------------------------------------------
Every one of the 30 GPT2 blocks is sharded exactly like the graduated
``g_p_t2_block`` stub:
  * LayerNorms (ln_1, ln_2, ln_f, final_norm) reduce over the full hidden dim ->
    REPLICATED; the block input arrives replicated so every chip normalizes the
    identical row.
  * Attention is HEAD-parallel (16 heads / 8 chips = 2 heads/chip): fused
    ``c_attn`` split into head-major q|k|v column shards
    (``ShardTensorToMesh(dim=1)``), per-chip on-device causal flash-attention,
    ``all_gather(dim=2)`` to reassemble heads, REPLICATED ``c_proj``.
  * MLP is column-then-gather: ``c_fc`` COLUMN-parallel + tanh-GELU per chip,
    ``all_gather(dim=2)``, REPLICATED ``c_proj``.
The lm_head linear (1024->1026) is small and REPLICATED. Only the placement of
each block's large projections changes; the gathered logits equal the
single-device golden.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    transformer = torch_module.transformer
    blocks = transformer.h
    ln_f = transformer.ln_f
    final_norm = torch_module.final_norm          # == lm_head[0]
    lm_linear = torch_module.lm_head[1]           # nn.Linear(1024 -> 1026)

    attn0 = blocks[0].attn
    n_heads = int(attn0.num_heads)
    head_dim = int(attn0.head_dim)
    embed = n_heads * head_dim
    scaling = float(getattr(attn0, "scaling", head_dim ** -0.5))

    # 30 stacked layers compound bf16 rounding; run every matmul at full HiFi4
    # fidelity with fp32 accumulation so the gathered logits stay within PCC.
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

    # Per-block sharded weights (mirrors the graduated g_p_t2_block scheme).
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
    fn_w, fn_b, fn_eps = _norm(final_norm)
    # nn.Linear: y = x @ W^T + b; store the transpose for a plain matmul.
    lm_w = _rep(lm_linear.weight.detach().t())    # [1024, 1026]
    lm_b = _rep(lm_linear.bias.detach().reshape(1, 1, -1))

    def _to_heads(t, T):
        hl = int(t.shape[-1]) // head_dim
        t = ttnn.reshape(t, [1, T, hl, head_dim])
        return ttnn.permute(t, [0, 2, 1, 3])

    def _block(x, L, T):
        # --- attention ---
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

        # --- mlp ---
        h = ttnn.layer_norm(x, weight=L["ln2"][0], bias=L["ln2"][1], epsilon=L["ln2"][2])
        ff = ttnn.add(_mm(h, L["wt_fc"]), L["b_fc"])
        ff = ttnn.gelu(ff, variant=ttnn.GeluVariant.Tanh)
        ff = ttnn.all_gather(ff, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        x = ttnn.add(x, ttnn.add(_mm(ff, L["wt_mo"]), L["b_mo"]))
        return x

    def forward(emb, *_, **__):
        T = int(emb.shape[-2])
        x = emb
        for L in layers:
            x = _block(x, L, T)
        x = ttnn.layer_norm(x, weight=ln_f_w, bias=ln_f_b, epsilon=ln_f_eps)
        x = ttnn.layer_norm(x, weight=fn_w, bias=fn_b, epsilon=fn_eps)
        return ttnn.add(_mm(x, lm_w), lm_b)    # [1, T, vocab]

    return forward
