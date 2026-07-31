# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `gpt_gpt_inference` for coqui/XTTS-v2.

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
    per-chip head merge, then a ROW-parallel ``c_proj`` + ``all_reduce``.
  * MLP is column-then-row: ``c_fc`` COLUMN-parallel + tanh-GELU per chip, then a
    ROW-parallel ``c_proj`` + ``all_reduce``.
  Both output projections contract over exactly the dim their producer is already
  split on, so each chip multiplies its own slice by its own weight ROWS and the
  partials are summed. That is the same math as gathering the input to full width
  and running a replicated projection, but the collective carries the 1024-wide
  OUTPUT instead of the 4096-wide MLP input, and each chip does 1/8 of the matmul
  rather than all of it.
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

    # The CCL sum needs no multiplier precision (it is an add into an fp32 accumulator),
    # so it runs at LoFi while every matmul stays at HiFi4.
    _ccl_kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # NORM fidelity. LayerNorm was running on ttnn's default compute config; the catalogued
    # policy for norms is HiFi2 (never LoFi -- it compounds to a PCC failure over depth) with
    # fp32_dest_acc_en MANDATORY, because the variance reduction loses too much precision in
    # an fp16 destination register. HiFi2 halves the math cost of the reduction relative to
    # HiFi4 while keeping the accumulator wide.
    _ln_kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _mm(a, b):
        return ttnn.matmul(a, b, compute_kernel_config=kcfg)

    def _mm_l1(a, b):
        """Row-parallel partial product, emitted straight into L1.

        Its only consumer is the collective in _reduce, which is dispatch/latency-bound.
        Producing the partial in L1 (and reducing back into L1) means no DRAM round-trip
        anywhere around the collective, instead of write-DRAM / read-DRAM / write-DRAM /
        read-DRAM for the matmul->reduce->add chain."""
        try:
            return ttnn.matmul(a, b, compute_kernel_config=kcfg,
                               memory_config=ttnn.L1_MEMORY_CONFIG)
        except Exception:  # noqa: BLE001 - shape does not fit L1
            return ttnn.matmul(a, b, compute_kernel_config=kcfg)

    def _stage(t, mapper):
        """Upload one weight to the mesh, tilizing ROW VECTORS on the host.

        A bias / norm scale has logical height 1, so a DEVICE tilize has to val-pad it
        1 -> 32 rows, which ttnn runs on a SINGLE core: ~88 us per call for a few KB.
        There are 8 such vectors per block, so the 30 blocks burned tens of ms of device
        time uploading ~1 MB. Tilizing them host-side is free by comparison. Real
        matrices are already tile-shaped (no val padding) and keep the multicore device
        path, where host tilizing megabytes would be the worse trade."""
        t = t.contiguous().to(torch.bfloat16)
        if t.dim() >= 2 and int(t.shape[-2]) == 1:
            return ttnn.to_device(
                ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mapper),
                device)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, mesh_mapper=mapper)

    def _rep(t):
        return _stage(t, ttnn.ReplicateTensorToMesh(device))

    def _shard(t, dim):
        return _stage(t, ttnn.ShardTensorToMesh(device, dim=dim))

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
            # Both output projections are ROW-parallel: their contraction dim is exactly
            # the dim the producer is already split on, so each chip owns the matching
            # weight ROWS and the partial products are SUMMED (see _block).
            "wt_ao": _shard(attn.c_proj.weight.detach(), 0),
            "b_ao": _rep(attn.c_proj.bias.detach().reshape(1, 1, -1)),
            "wt_fc": _shard(mlp.c_fc.weight.detach(), 1),
            "b_fc": _shard(mlp.c_fc.bias.detach().reshape(1, 1, -1), 2),
            "wt_mo": _shard(mlp.c_proj.weight.detach(), 0),
            "b_mo": _rep(mlp.c_proj.bias.detach().reshape(1, 1, -1)),
        })

    ln_f_w, ln_f_b, ln_f_eps = _norm(ln_f)
    fn_w, fn_b, fn_eps = _norm(final_norm)
    # nn.Linear: y = x @ W^T + b; store the transpose for a plain matmul.
    lm_w = _rep(lm_linear.weight.detach().t())    # [1024, 1026]
    lm_b = _rep(lm_linear.bias.detach().reshape(1, 1, -1))

    def _reduce(partial):
        """Finish a ROW-parallel projection: sum the per-chip partial products.

        The old scheme all_gathered the projection INPUT to full width and then ran a
        REPLICATED projection, which (a) made every chip redo the whole 4096->1024 MLP
        matmul and (b) moved 4096 columns over the fabric. Row-parallel moves the SAME
        1024-wide output instead -- 4x fewer bytes for the MLP -- and each chip does 1/8
        of the matmul. Mathematically identical: sum_i (x_i . W_i) == x . W for a weight
        split along the contraction dim, and the bias is added once, after the sum."""
        if n_dev <= 1:
            return partial
        # Spelled out as the reduce_scatter + all_gather pair that all_reduce runs anyway,
        # for two reasons: (a) only reduce_scatter takes a compute_kernel_config, so the
        # summation fidelity is reachable at all, and (b) all_reduce only lets us place its
        # final output, while this places BOTH halves in L1. The collective is DISPATCH-bound
        # (profile tags both halves grid=tiny), so keeping the whole chain out of DRAM is the
        # lever. The reduce is a plain sum of 8 partials into an fp32 accumulator, so it does
        # not need HiFi4 multiplier precision.
        # NOTE: do NOT reach for num_links>1 here -- on this 1x8 linear fabric that HANGS the
        # collective (silent timeout, no exception, wedged device).
        try:
            scattered = ttnn.reduce_scatter(
                partial, dim=2, num_links=1, topology=ttnn.Topology.Linear,
                memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_ccl_kcfg)
            return ttnn.all_gather(scattered, dim=2, num_links=1,
                                   topology=ttnn.Topology.Linear,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)
        except Exception:  # noqa: BLE001 - fall back to the fused op
            return ttnn.all_reduce(partial, num_links=1, topology=ttnn.Topology.Linear,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)

    def _mlp(x, L):
        h = ttnn.layer_norm(x, weight=L["ln2"][0], bias=L["ln2"][1],
                            epsilon=L["ln2"][2], compute_kernel_config=_ln_kcfg)
        ff = ttnn.add(_mm(h, L["wt_fc"]), L["b_fc"])
        ff = ttnn.gelu(ff, variant=ttnn.GeluVariant.Tanh)
        return ttnn.add(x, ttnn.add(_reduce(_mm_l1(ff, L["wt_mo"])), L["b_mo"]))

    def _head(x):
        x = ttnn.layer_norm(x, weight=ln_f_w, bias=ln_f_b, epsilon=ln_f_eps,
                            compute_kernel_config=_ln_kcfg)
        x = ttnn.layer_norm(x, weight=fn_w, bias=fn_b, epsilon=fn_eps,
                            compute_kernel_config=_ln_kcfg)
        return ttnn.add(_mm(x, lm_w), lm_b)

    def _block(x, L, T, kv_sink=None):
        # --- attention ---
        # ONE fused sharded qkv matmul + the multicore head shuffle. The hand-rolled
        # reshape([1,T,h,d]) + permute split used to be three separate matmuls whose
        # reshape SPLITS the last tile dim -- ttnn services that with an untilize +
        # SINGLE-CORE retilize (the profile's grid=tiny TilizeWithValPadding, ~62 us
        # per call, 4 per block). nlp_create_qkv_heads / nlp_concat_heads do the same
        # shuffle as one multicore device op, so no layout round-trip happens at all.
        h = ttnn.layer_norm(x, weight=L["ln1"][0], bias=L["ln1"][1],
                            epsilon=L["ln1"][2], compute_kernel_config=_ln_kcfg)
        qkv = ttnn.add(_mm(h, L["wt_qkv"]), L["bt_qkv"])       # [1, T, 3*shard_w]
        qkv = ttnn.reshape(qkv, [1, 1, T, 3 * shard_w])        # leading-dim view, no repack
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=heads_pc, num_kv_heads=heads_pc, transpose_k_heads=False)
        if kv_sink is not None:
            # K/V for the WHOLE padded prefix, in exactly the [1,h,C,d] layout the decode
            # attention wants -- so the cache is just what prefill already computed.
            kv_sink.append((k, v))
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scaling, compute_kernel_config=kcfg)
        ctx = ttnn.reshape(ttnn.experimental.nlp_concat_heads(ctx), [1, T, shard_w])
        x = ttnn.add(x, ttnn.add(_reduce(_mm_l1(ctx, L["wt_ao"])), L["b_ao"]))
        return _mlp(x, L)

    def forward(emb, *_, **__):
        T = int(emb.shape[-2])
        x = emb
        for L in layers:
            x = _block(x, L, T)
        return _head(x)                        # [1, T, vocab]

    # ================================================================= #
    # KV-CACHE DECODE. Without it the AR loop is repeat_prefill: every token re-runs the
    # FULL padded prefix through all 30 blocks, so each step pays T rows of matmul, T rows
    # of LayerNorm and a T-row collective to learn ONE new row. With the cache a step
    # computes seq_len=1 and attends to cached history, which is where the per-token time
    # actually goes. Attached as attributes on `forward` so the stub's build(device, module)
    # -> forward(emb) contract is unchanged for the PCC harness and the full-sequence path.
    # ================================================================= #
    _kv = {}

    def _shard_kv_row(t):
        """paged_update_cache REQUIRES a sharded input (it asserts input_tensor.is_sharded()).

        The row is [1,1,heads_pc,head_dim] -- heads_pc < 32, so it is a single tile row once
        padded -- which means one L1 shard of [TILE, head_dim] on one core satisfies the op
        with no real data movement. The SHAPE must be left alone: the op cross-checks the
        input's last dim against the cache's head_dim, so folding heads into the last dim
        (the obvious way to make it one row) is rejected."""
        core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(core, [32, int(t.shape[-1])], ttnn.ShardOrientation.ROW_MAJOR))
        return ttnn.to_memory_config(t, cfg)

    def prefill_cache(emb):
        """Run the prefix once, KEEPING every layer's K/V. Returns the prefix logits."""
        T = int(emb.shape[-2])
        kv = []
        x = emb
        for L in layers:
            x = _block(x, L, T, kv_sink=kv)
        _kv["kv"], _kv["T"] = kv, T
        return _head(x)

    def decode_one(row_emb, pos):
        """ONE cached AR step. row_emb: [1,1,dim] (the new token only); pos: a DEVICE tensor
        holding its absolute position, so the step is trace-capturable (a Python int would
        bake a stale constant into the trace). Returns [1,1,vocab]."""
        x = row_emb
        for L, (kc, vc) in zip(layers, _kv["kv"]):
            h = ttnn.layer_norm(x, weight=L["ln1"][0], bias=L["ln1"][1],
                            epsilon=L["ln1"][2], compute_kernel_config=_ln_kcfg)
            qkv = ttnn.add(_mm(h, L["wt_qkv"]), L["bt_qkv"])            # [1,1,3*shard_w]
            qkv = ttnn.reshape(qkv, [1, 1, 1, 3 * shard_w])
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=heads_pc, num_kv_heads=heads_pc, transpose_k_heads=False)
            # nlp_create_qkv_heads gives [1,h,S,d]; the cache-update and decode-attention
            # ops both want the batch-major [1,S,h,d], and the update wants it SHARDED.
            ttnn.experimental.paged_update_cache(
                kc, _shard_kv_row(ttnn.permute(k, [0, 2, 1, 3])), update_idxs_tensor=pos)
            ttnn.experimental.paged_update_cache(
                vc, _shard_kv_row(ttnn.permute(v, [0, 2, 1, 3])), update_idxs_tensor=pos)
            ctx = ttnn.transformer.scaled_dot_product_attention_decode(
                ttnn.permute(q, [0, 2, 1, 3]), kc, vc, is_causal=True,
                cur_pos_tensor=pos, scale=scaling, compute_kernel_config=kcfg)
            ctx = ttnn.reshape(
                ttnn.experimental.nlp_concat_heads(ttnn.permute(ctx, [0, 2, 1, 3])),
                [1, 1, shard_w])
            x = ttnn.add(x, ttnn.add(_reduce(_mm_l1(ctx, L["wt_ao"])), L["b_ao"]))
            x = _mlp(x, L)
        return _head(x)

    forward.prefill_cache = prefill_cache
    forward.decode_one = decode_one
    forward.cache_len = lambda: int(_kv.get("T", 0))
    return forward
