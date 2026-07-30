# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t` for coqui/XTTS-v2.

HF submodule: ``gpt`` -- the XTTS ``GPT`` module. Called here in the
``return_latent=True`` path (``cond_latents`` supplied), whose forward is:

    emb    = concat(cond_latents, text_emb, mel_emb)          # [1, L, 1024]
    hidden = gpt(inputs_embeds=emb).last_hidden_state         # ln_f(30xblock)
    enc    = final_norm(hidden[:, offset:])                   # drop cond rows
    return enc[:, -mel_len:][:, :-5]                          # mel latent

The token->vector front-half (max-length padding, set_mel_padding, start/stop
prepend, embedding-table gather, the cond|text|mel concat) is pure integer glue
with data-dependent control flow -- NOT tensor-parallel compute. So, mirroring
the graduated ``gpt_gpt_inference`` stub, the PCC harness feeds the module's OWN
assembled ``inputs_embeds`` (``emb``) as this stub's primary input, and this
stub natively reproduces the TP-relevant compute: the 30 GPT2 blocks + ``ln_f``
+ ``final_norm``, then the mel-latent slice. The golden stays the FULL module
forward over the captured token ids.

Because ``ln_f`` and ``final_norm`` are per-position LayerNorms, dropping the
``offset`` conditioning rows before ``final_norm`` is unnecessary: the kept
output is ``final_norm(hidden)[:, L-mel_len : L-sub]`` (L = full seq len,
sub = 5), taken from the tail, so neither ``offset`` nor ``text_len`` is needed.

TP=8 scheme (genuine tensor-parallel, math unchanged)
-----------------------------------------------------
Every one of the 30 GPT2 blocks is sharded exactly like the graduated
``g_p_t2_block`` / ``gpt_gpt`` stubs:
  * LayerNorms (ln_1, ln_2, ln_f, final_norm) reduce over the full hidden dim ->
    REPLICATED; the block input arrives replicated so every chip normalizes the
    identical row.
  * Attention is HEAD-parallel (16 heads / 8 chips = 2 heads/chip): fused
    ``c_attn`` split into head-major q|k|v column shards
    (``ShardTensorToMesh(dim=1)``), per-chip on-device causal flash-attention,
    ``all_gather(dim=2)`` to reassemble heads, REPLICATED ``c_proj``.
  * MLP is column-then-gather: ``c_fc`` COLUMN-parallel + tanh-GELU per chip,
    ``all_gather(dim=2)``, REPLICATED ``c_proj``.
30 stacked layers compound bf16 rounding, so every matmul runs at HiFi4 fidelity
with fp32 accumulation. The gathered, sliced latent equals the single-device
golden.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    transformer = torch_module.gpt
    blocks = transformer.h
    ln_f = transformer.ln_f
    final_norm = torch_module.final_norm

    # Slice geometry published by the PCC harness (see test_g_p_t.py).
    mel_len = int(getattr(torch_module, "_tt_mel_len"))
    sub = int(getattr(torch_module, "_tt_sub", 5))

    attn0 = blocks[0].attn
    n_heads = int(attn0.num_heads)
    head_dim = int(attn0.head_dim)
    embed = n_heads * head_dim
    scaling = float(getattr(attn0, "scaling", head_dim ** -0.5))

    # 30 stacked layers compound bf16 rounding; run every matmul at full HiFi4
    # fidelity with fp32 accumulation so the gathered latent stays within PCC.
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

    # Chip count drives the PER-CHIP head slice: ShardTensorToMesh(dim=1) hands each chip
    # `embed // n_dev` columns of each projection, i.e. `heads_pc` whole heads.
    try:
        n_dev = max(1, len(device.get_device_ids()))
    except Exception:  # noqa: BLE001 - single (non-mesh) device
        n_dev = 1
    shard_w = embed // n_dev
    heads_pc = shard_w // head_dim

    def _fused_qkv(Wc, bc):
        """Interleave q|k|v PER CHIP so one sharded matmul emits the fused
        [q_heads | k_heads | v_heads] layout ``nlp_create_qkv_heads`` consumes.

        ShardTensorToMesh(dim=1) cuts dim 1 into n_dev equal chunks, so chunk i must
        already hold chip i's q, then k, then v columns — hence the per-chip regroup
        here rather than a plain cat of the three whole projections."""
        cols, bias = [], []
        for i in range(n_dev):
            s = slice(i * shard_w, (i + 1) * shard_w)
            for o in (0, embed, 2 * embed):
                cols.append(Wc[:, o:o + embed][:, s])
                bias.append(bc[o:o + embed][s])
        return (_shard(torch.cat(cols, dim=1), 1),
                _shard(torch.cat(bias).reshape(1, 1, -1), 2))

    # Per-block sharded weights (mirrors the graduated g_p_t2_block scheme).
    layers = []
    for blk in blocks:
        attn, mlp = blk.attn, blk.mlp
        Wc = attn.c_attn.weight.detach()          # [1024, 3072] (in, q|k|v)
        bc = attn.c_attn.bias.detach()
        wt_qkv, bt_qkv = _fused_qkv(Wc, bc)
        layers.append({
            "ln1": _norm(blk.ln_1),
            "ln2": _norm(blk.ln_2),
            "wt_qkv": wt_qkv,
            "bt_qkv": bt_qkv,
            "wt_ao": _rep(attn.c_proj.weight.detach()),
            "b_ao": _rep(attn.c_proj.bias.detach().reshape(1, 1, -1)),
            "wt_fc": _shard(mlp.c_fc.weight.detach(), 1),
            "b_fc": _shard(mlp.c_fc.bias.detach().reshape(1, 1, -1), 2),
            "wt_mo": _rep(mlp.c_proj.weight.detach()),
            "b_mo": _rep(mlp.c_proj.bias.detach().reshape(1, 1, -1)),
        })

    ln_f_w, ln_f_b, ln_f_eps = _norm(ln_f)
    fn_w, fn_b, fn_eps = _norm(final_norm)

    def _block(x, L, T):
        # --- attention ---
        # ONE fused sharded qkv matmul + the multicore head shuffle. The hand-rolled
        # reshape([1,T,h,d]) + permute split used to be three separate matmuls whose
        # reshape SPLITS the last tile dim -- ttnn services that with an untilize +
        # SINGLE-CORE retilize (the profile's grid=tiny TilizeWithValPadding, ~62 us
        # per call, 4 per block). nlp_create_qkv_heads / nlp_concat_heads do the same
        # shuffle as one multicore device op, so no layout round-trip happens at all.
        h = ttnn.layer_norm(x, weight=L["ln1"][0], bias=L["ln1"][1], epsilon=L["ln1"][2])
        qkv = ttnn.add(_mm(h, L["wt_qkv"]), L["bt_qkv"])       # [1, T, 3*shard_w]
        qkv = ttnn.reshape(qkv, [1, 1, T, 3 * shard_w])        # leading-dim view, no repack
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=heads_pc, num_kv_heads=heads_pc, transpose_k_heads=False)
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scaling, compute_kernel_config=kcfg)
        ctx = ttnn.reshape(ttnn.experimental.nlp_concat_heads(ctx), [1, T, shard_w])
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
        # last_hidden_state = ln_f(blocks(emb)); then the module's final_norm.
        x = ttnn.layer_norm(x, weight=ln_f_w, bias=ln_f_b, epsilon=ln_f_eps)
        x = ttnn.layer_norm(x, weight=fn_w, bias=fn_b, epsilon=fn_eps)
        # mel latent = final_norm(hidden)[:, -mel_len:][:, :-sub] (LayerNorm is
        # per-position, so slicing the tail is exact). Full hidden dim = 1024.
        hidden_dim = int(x.shape[-1])
        return ttnn.slice(x, [0, T - mel_len, 0], [1, T - sub, hidden_dim])

    return forward


def g_p_t(device, torch_module=None):
    return build(device, torch_module)
