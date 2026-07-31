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

    def _add_l1(a, b):
        """Residual-stream add, landing in L1.

        Its consumer is the next LayerNorm, which is DISPATCH/latency-bound at one-to-two
        tile rows -- so the lever is removing the DRAM write + read between them, not giving
        the norm more cores (a width shard was measured and LOST: the reshard cost more than
        the norm saved, because the qkv matmul downstream reshards anyway). The residual
        stream is [1,T,1024] bf16, one live tensor at a time, so it cannot crowd L1."""
        try:
            return ttnn.add(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)
        except Exception:  # noqa: BLE001 - will not fit L1
            return ttnn.add(a, b)

    # knob:grid (second attempt). Forcing the MAXIMUM core grid regressed decode 4.6%
    # (64 cores each owning one output tile, mcast setup dominating). So try a RIGHT-SIZED
    # explicit program config instead: one output tile per core, K split into blocks so the
    # in0 stream is bounded, derived from the actual tile counts at the call site rather
    # than from a fixed guess. out_subblock_h*w must stay <= 4 because fp32_dest_acc_en is
    # on. Falls back to the default routing for shapes this cannot tile evenly.
    _grid = device.compute_with_storage_grid_size()

    def _pcfg(a, b):
        mt = (int(a.shape[-2]) + 31) // 32
        kt = int(a.shape[-1]) // 32
        nt = int(b.shape[-1]) // 32
        if kt == 0 or nt == 0 or int(a.shape[-1]) % 32 or int(b.shape[-1]) % 32:
            return None
        ncores = min(_grid.x * _grid.y, nt)
        gx = min(_grid.x, ncores)
        gy = max(1, ncores // gx)
        if gx * gy == 0 or nt % (gx * gy):
            return None
        ibw = 4 if kt % 4 == 0 else 1
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
            per_core_M=mt, per_core_N=nt // (gx * gy),
            fuse_batch=True, fused_activation=None, mcast_in0=True)

    def _mm(a, b):
        return ttnn.matmul(a, b, compute_kernel_config=kcfg)

    def _lin(a, b, bias):
        """Projection WITH its bias folded into the matmul.

        A decode step is dispatch-bound -- op COUNT is the cost -- and a separate
        ttnn.add for the bias is a whole extra dispatch to add one broadcast row to a
        result the matmul kernel already has in its accumulator. ttnn.linear does it in
        the same op. This is only legal where the bias belongs to THIS chip's shard: the
        qkv and c_fc projections are COLUMN-parallel (each chip owns the matching bias
        columns), whereas the row-parallel output projections must add their replicated
        bias ONCE, after the collective -- never folded into the per-chip partial."""
        pc = _pcfg(a, b)
        if pc is None:
            return ttnn.linear(a, b, bias=bias, compute_kernel_config=kcfg)
        return ttnn.linear(a, b, bias=bias, compute_kernel_config=kcfg,
                           program_config=pc)

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
        # DECODE takes the FUSED op, and for the opposite reason to the prefill path above.
        # At one tile row the collective is pure latency: the reduce_scatter + all_gather
        # pair is TWO dispatches per projection, i.e. 4 per block and 120 per token across
        # 30 blocks, to move a single 1024-wide row. ttnn.all_reduce does the same math in
        # ONE dispatch, halving that. The two reasons the pair is preferred at prefill both
        # evaporate here: the summation fidelity a compute_kernel_config would set is
        # irrelevant to summing 8 partials into an fp32 accumulator, and placing BOTH halves
        # in L1 is worth nothing when there is no intermediate big enough to care about.
        if int(partial.shape[-2]) <= 1:
            return ttnn.all_reduce(partial, num_links=1, topology=ttnn.Topology.Linear,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)
        try:
            # knob:shard -- give the collective a WIDTH-SHARDED input so its workers read
            # their own L1 shard instead of an interleaved buffer: one tile column per core,
            # which spreads it over the grid without splitting a tile.
            #
            # Measured both ways: on the MULTI-row prefill / latent forwards this is worth
            # ~1.2 ms of device time, but at DECODE (one tile row) it COST 2.6% per token --
            # there is nothing to spread across cores, so the reshard is pure added latency
            # on an already latency-bound op. That is the same shape rule the catalogued
            # reduction lever states for LayerNorm, and it holds for the collective too.
            if int(partial.shape[-2]) > 1:
                W = int(partial.shape[-1])
                ncores = max(1, min(32, W // 32))
                cfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(7, ncores - 1),
                                                                 (ncores - 1) // 8))}),
                        [32 * max(1, (int(partial.shape[-2]) + 31) // 32), W // ncores],
                        ttnn.ShardOrientation.ROW_MAJOR))
                partial = ttnn.to_memory_config(partial, cfg)
            scattered = ttnn.reduce_scatter(
                partial, dim=2, num_links=1, topology=ttnn.Topology.Linear,
                memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_ccl_kcfg)
            # num_workers_per_link is the OCCUPANCY knob for a CCL (worker cores on the SAME
            # ethernet link -- a different thing from num_links, which HANGS this 1x8 linear
            # fabric). At decode the gathered tensor is ONE tile row: there is not enough
            # payload for a second worker to overlap, so the extra core only adds setup to a
            # collective that is already latency-bound. Left at the default.
            return ttnn.all_gather(scattered, dim=2, num_links=1,
                                   topology=ttnn.Topology.Linear,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)
        except Exception:  # noqa: BLE001 - fall back to the fused op
            return ttnn.all_reduce(partial, num_links=1, topology=ttnn.Topology.Linear,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)

    def _mlp(x, L):
        h = ttnn.layer_norm(x, weight=L["ln2"][0], bias=L["ln2"][1],
                            epsilon=L["ln2"][2], compute_kernel_config=_ln_kcfg)
        ff = _lin(h, L["wt_fc"], L["b_fc"])
        ff = ttnn.gelu(ff, variant=ttnn.GeluVariant.Tanh)
        return _add_l1(x, ttnn.add(_reduce(_mm_l1(ff, L["wt_mo"])), L["b_mo"]))

    def _head(x):
        x = ttnn.layer_norm(x, weight=ln_f_w, bias=ln_f_b, epsilon=ln_f_eps,
                            compute_kernel_config=_ln_kcfg)
        x = ttnn.layer_norm(x, weight=fn_w, bias=fn_b, epsilon=fn_eps,
                            compute_kernel_config=_ln_kcfg)
        return _lin(x, lm_w, lm_b)

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
        qkv = _lin(h, L["wt_qkv"], L["bt_qkv"])                # [1, T, 3*shard_w]
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
        x = _add_l1(x, ttnn.add(_reduce(_mm_l1(ctx, L["wt_ao"])), L["b_ao"]))
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

    # nlp_create_qkv_heads_decode asserts its input is WIDTH_SHARDED (not height —
    # it splits the fused row BY HEAD across cores and hands each core its own head).
    # So give it exactly that: one head per core, head_dim wide, over the
    # 3*heads_pc heads of the fused q|k|v row. Laying it out this way also lands Q,
    # K and V on disjoint cores, which is what the FUSED K/V cache write requires.
    _qkv_cores = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3 * heads_pc - 1, 0))})
    _qkv_shard = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(_qkv_cores, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR))

    def _shard_row(t):
        return ttnn.to_memory_config(t, _qkv_shard)

    def _shard_kv_row(t, cx=0):
        """paged_update_cache REQUIRES a sharded input (it asserts input_tensor.is_sharded()).

        The row is [1,1,heads_pc,head_dim] -- heads_pc < 32, so it is a single tile row once
        padded -- which means one L1 shard of [TILE, head_dim] on one core satisfies the op
        with no real data movement. The SHAPE must be left alone: the op cross-checks the
        input's last dim against the cache's head_dim, so folding heads into the last dim
        (the obvious way to make it one row) is rejected.

        ``cx`` picks WHICH core: the FUSED K+V cache write asserts its two inputs occupy
        non-overlapping core ranges (it runs both writes concurrently), so K and V have to
        land on different cores or the op rejects them outright."""
        core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(cx, 0), ttnn.CoreCoord(cx, 0))})
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
            qkv = _lin(h, L["wt_qkv"], L["bt_qkv"])                     # [1,1,3*shard_w]
            # DECODE-SPECIFIC head shuffle. The generic nlp_create_qkv_heads emits
            # [1,h,S,d], but every consumer here (cache write, decode attention) wants
            # batch-major [1,S,h,d] and SHARDED -- which cost three ttnn.permute calls
            # plus two to_memory_config reshards per block. With heads_pc=2 and S=1 those
            # permutes move a 2x64 payload but pay a full untilize + VAL-PADDED retilize
            # each (both 2 and 1 pad to a 32-row tile), which is where the profile's
            # grid=tiny TilizeWithValPadding/UntilizeWithUnpadding time comes from.
            # nlp_create_qkv_heads_decode emits that exact layout, already sharded.
            qkv = _shard_row(ttnn.reshape(qkv, [1, 1, 1, 3 * shard_w]))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=heads_pc, num_kv_heads=heads_pc,
                overlap_qk_coregrid=False)
            # K and V must occupy non-overlapping core ranges for the FUSED write; the
            # decode head op already places them apart, so one dispatch does both.
            ttnn.experimental.paged_fused_update_cache(kc, k, vc, v, update_idxs_tensor=pos)
            ctx = ttnn.transformer.scaled_dot_product_attention_decode(
                q, kc, vc, is_causal=True,
                cur_pos_tensor=pos, scale=scaling, compute_kernel_config=kcfg)
            # The ctx side stays on the generic permute + concat. Its decode-specific
            # twin (nlp_concat_heads_decode) asserts a SHARDED input, and sdpa_decode
            # refuses to produce one for this head configuration ("Sharded output not
            # supported for GQA"), so chaining them would need a reshard that costs
            # more than the two ops it removes.
            ctx = ttnn.reshape(
                ttnn.experimental.nlp_concat_heads(ttnn.permute(ctx, [0, 2, 1, 3])),
                [1, 1, shard_w])
            x = _add_l1(x, ttnn.add(_reduce(_mm_l1(ctx, L["wt_ao"])), L["b_ao"]))
            x = _mlp(x, L)
        return _head(x)

    forward.prefill_cache = prefill_cache
    forward.decode_one = decode_one
    forward.cache_len = lambda: int(_kv.get("T", 0))
    return forward
