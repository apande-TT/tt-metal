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

    # DECODE-ATTENTION fidelity. scaled_dot_product_attention_decode (flash-DECODE) returns
    # GARBAGE when fp32_dest_acc_en is on: measured against a torch reference on this box,
    # relative error is ~6.0 with fp32_dest_acc_en=True at ANY math fidelity, and ~0.02 with
    # it off (HiFi4 0.020 / HiFi2 0.018 / LoFi 0.074). The full-sequence flash SDPA is fine
    # with the same config (0.024), so this is specific to the decode kernel -- and it fails
    # SILENTLY: greedy decode still produces plausible-looking, stable token ids, so only a
    # per-step comparison against the reference catches it. Keep HiFi4 for the multiplier and
    # leave the destination register in bf16 for THIS op only.
    _sdpa_dec_kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
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

    def _wshard(rows, width):
        """WIDTH-SHARDED L1: one tile column per core, never splitting a tile."""
        nc = max(1, min(32, width // 32))
        return nc, ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(min(7, nc - 1), (nc - 1) // 8))}),
                [rows, width // nc],
                ttnn.ShardOrientation.ROW_MAJOR))

    def _add_l1(a, b):
        """Residual-stream add, landing WIDTH-SHARDED in L1.

        The residual add's consumer is the next LayerNorm, and an INTERLEAVED LayerNorm
        parallelises over tile ROWS -- so at decode, where the stream is ONE tile row, it
        runs on ONE core and costs 41 us to normalise 64 KB. Over 30 blocks x 2 norms that
        is ~2.5 ms of the ~8.7 ms token, the largest single reachable cost in the decode
        step. The sharded LayerNorm parallelises over the WIDTH instead (32 tile columns ->
        32 cores, cross-core reduction), which is the only form that scales at one row.
        A width shard was tried here before and lost -- but that attempt paid an explicit
        to_memory_config in front of the norm, and the reshard cost more than the norm
        saved. Placing the shard as the ADD's own output makes it free, so the earlier
        result does not carry over. The residual stream is [B,T,1024] bf16, one live
        tensor at a time, so it cannot crowd L1."""
        rows = _mtiles(a) * 32
        w = int(a.shape[-1])
        if w // 32 >= 8:
            try:
                return ttnn.add(a, b, memory_config=_wshard(rows, w)[1])
            except Exception:  # noqa: BLE001 - shard spec rejected for this shape
                pass
        try:
            return ttnn.add(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)
        except Exception:  # noqa: BLE001 - will not fit L1
            return ttnn.add(a, b)

    def _ln(x, weight, bias, eps, keep_sharded=False):
        """LayerNorm that follows its input's placement.

        On a width-sharded input take the sharded program config -- block_h is the
        stream's tile rows, block_w the tile columns each core owns (1 by construction of
        _wshard), and the grid is exactly the shard's core range. On anything else fall
        back to the interleaved form."""
        mc = x.memory_config()
        if mc.is_sharded():
            rows = _mtiles(x) * 32
            w = int(x.shape[-1])
            nc = max(1, min(32, w // 32))
            gx, gy = min(8, nc), max(1, (nc + 7) // 8)
            try:
                y = ttnn.layer_norm(
                    x, weight=weight, bias=bias, epsilon=eps,
                    compute_kernel_config=_ln_kcfg, memory_config=mc,
                    program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                        subblock_w=1, block_h=max(1, rows // 32),
                        block_w=(w // 32) // nc, inplace=False))
                # Hand the norm's consumer an INTERLEAVED tensor. A MultiCast1D matmul
                # with a sharded in0 demands that the shard's core count equal the program
                # config's grid, and that grid is sized from N -- 12 cores for the fused
                # qkv, 16 for c_fc -- which a K-side width shard cannot match (K=32 tiles
                # does not divide by 12). One reshard here is far cheaper than losing both
                # the tuned matmul configs and the fused GELU, which a sharded activation
                # also forbids.
                # A norm whose consumer is ANOTHER norm should stay sharded: the reshard
                # exists only for the matmuls, and the two-LayerNorm lm_head would
                # otherwise hand the second norm an interleaved tensor and drop it back
                # to the 1-core kernel.
                return y if keep_sharded else ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
            except Exception:  # noqa: BLE001 - fall back to the interleaved norm
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=eps,
                               compute_kernel_config=_ln_kcfg)

    # knob:grid (second attempt). Forcing the MAXIMUM core grid regressed decode 4.6%
    # (64 cores each owning one output tile, mcast setup dominating). So try a RIGHT-SIZED
    # explicit program config instead: one output tile per core, K split into blocks so the
    # in0 stream is bounded, derived from the actual tile counts at the call site rather
    # than from a fixed guess. out_subblock_h*w must stay <= 4 because fp32_dest_acc_en is
    # on. Falls back to the default routing for shapes this cannot tile evenly.
    _grid = device.compute_with_storage_grid_size()

    def _mtiles(t):
        """M tiles as the matmul sees them.

        With ``fuse_batch=True`` every LEADING dim folds into M, so a batched ``[B, T, K]``
        activation contributes B slices -- and each slice's rows are PADDED to a whole tile.
        The count is therefore ``prod(leading) * ceil(T/32)``:
          * reading ``shape[-2]`` alone under-sizes ``per_core_M`` B-fold (the tuned config
            would compute only the first stream's rows);
          * ``ceil(prod(leading)*T / 32)`` under-counts whenever T is NOT tile-aligned, and
            the config then asks for more blocks than there are cores
            (TT_FATAL num_blocks_total <= num_cores).
        ``ttnn.Shape`` has no python slicing, so walk it by index."""
        n = 1
        for i in range(len(t.shape) - 2):
            n *= int(t.shape[i])
        return n * ((int(t.shape[-2]) + 31) // 32)

    # The MLP's tanh-GELU is a standalone unary pass over the [rows, 4096] hidden -- the
    # widest activation in the block -- purely to apply a scalar function to a tensor the
    # c_fc matmul had in its accumulator one op earlier. It rides the matmul's packer
    # instead: the fused form costs the SFPU LUT and nothing else, while the standalone op
    # costs a dispatch plus a full DRAM round-trip of the hidden.
    _GELU_TANH = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)

    def _pcfg(a, b, act=None):
        mt = _mtiles(a)
        kt = int(a.shape[-1]) // 32
        nt = int(b.shape[-1]) // 32
        if kt == 0 or nt == 0 or int(a.shape[-1]) % 32 or int(b.shape[-1]) % 32:
            return None
        # Pick the core count as the largest DIVISOR of N_tiles that factors into the real
        # grid, so every core gets a whole number of output tiles. A naive
        # gx=min(grid.x, nt) silently bailed out on the fused qkv projection (N=12 tiles
        # does not divide an 8x1 grid), which left the biggest projection in the block on
        # ttnn's default routing while the MLP got the tuned config.
        fac = None
        for d in range(min(_grid.x * _grid.y, nt), 0, -1):
            if nt % d:
                continue
            for gx in range(min(d, _grid.x), 0, -1):
                if d % gx == 0 and d // gx <= _grid.y:
                    fac = (gx, d // gx)
                    break
            if fac:
                break
        if not fac:
            return None
        gx, gy = fac
        ibw = 4 if kt % 4 == 0 else 1
        if a.memory_config().is_sharded():
            return None
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
            per_core_M=mt, per_core_N=nt // (gx * gy),
            fuse_batch=True, fused_activation=act, mcast_in0=True)

    def _mm(a, b):
        return ttnn.matmul(a, b, compute_kernel_config=kcfg)

    def _lin(a, b, bias, act=None):
        """Projection WITH its bias (and optionally its activation) folded into the matmul.

        A decode step is dispatch-bound -- op COUNT is the cost -- and a separate
        ttnn.add for the bias is a whole extra dispatch to add one broadcast row to a
        result the matmul kernel already has in its accumulator. ttnn.linear does it in
        the same op. This is only legal where the bias belongs to THIS chip's shard: the
        qkv and c_fc projections are COLUMN-parallel (each chip owns the matching bias
        columns), whereas the row-parallel output projections must add their replicated
        bias ONCE, after the collective -- never folded into the per-chip partial."""
        pc = _pcfg(a, b, act)
        if pc is None:
            if act is None:
                return ttnn.linear(a, b, bias=bias, compute_kernel_config=kcfg)
            return ttnn.linear(a, b, bias=bias, compute_kernel_config=kcfg,
                               activation=act)
        # A program_config carries its OWN fused_activation, and ttnn rejects the
        # `activation` kwarg alongside one -- so the activation went into the pc above.
        return ttnn.linear(a, b, bias=bias, compute_kernel_config=kcfg,
                           program_config=pc)

    def _mm_l1(a, b):
        """Row-parallel partial product, emitted straight into L1, on a right-sized grid.

        These are the two c_proj projections per block, and they were the last matmuls
        still on ttnn default routing after the tuned program config landed on qkv/c_fc.

        Its only consumer is the collective in _reduce, which is dispatch/latency-bound.
        Producing the partial in L1 (and reducing back into L1) means no DRAM round-trip
        anywhere around the collective, instead of write-DRAM / read-DRAM / write-DRAM /
        read-DRAM for the matmul->reduce->add chain."""
        pc = _pcfg(a, b)
        try:
            return ttnn.matmul(a, b, compute_kernel_config=kcfg,
                               memory_config=ttnn.L1_MEMORY_CONFIG,
                               program_config=pc)
        except Exception:  # noqa: BLE001 - output too large for L1; fall back to DRAM
            return ttnn.matmul(a, b, compute_kernel_config=kcfg, program_config=pc)

    def _lin_l1(a, b, bias):
        """Row-parallel partial product with its SHARE of the replicated bias folded in.

        structural -- the row-parallel output projections used to add their bias as a
        separate ttnn.add AFTER the collective, on the stated rule that a replicated bias
        must be added exactly once and so can never go into a per-chip partial. That rule
        has an exact escape: the collective SUMS the n_dev partials, so staging bias/n_dev
        and folding THAT into each chip reproduces the bias precisely once after the sum
        (dividing by a power of two is exact in binary floating point, so it is not even a
        rounding trade). The bias then rides the matmul kernel's own accumulator instead of
        costing a dispatch -- one op fewer per output projection, 2 per block, 60 per
        30-block forward, on a profile whose largest single bucket is per-op dispatch."""
        pc = _pcfg(a, b)
        try:
            return ttnn.linear(a, b, bias=bias, compute_kernel_config=kcfg,
                               memory_config=ttnn.L1_MEMORY_CONFIG, program_config=pc)
        except Exception:  # noqa: BLE001 - output too large for L1; fall back to DRAM
            return ttnn.linear(a, b, bias=bias, compute_kernel_config=kcfg,
                               program_config=pc)

    def _stage(t, mapper, split=1):
        """Upload one weight to the mesh, tilizing ROW VECTORS on the host.

        A bias / norm scale has logical height 1, so a DEVICE tilize has to val-pad it
        1 -> 32 rows, which ttnn runs on a SINGLE core: ~88 us per call for a few KB.
        There are 8 such vectors per block, so the 30 blocks burned tens of ms of device
        time uploading ~1 MB. Tilizing them host-side is free by comparison. Real
        matrices are already tile-shaped (no val padding) and keep the multicore device
        path, where host tilizing megabytes would be the worse trade."""
        t = t.contiguous().to(torch.bfloat16)
        # knob:grid. from_torch(..., device=...) uploads ROW_MAJOR and tilizes ON DEVICE,
        # and a tilize parallelises over tile ROWS -- so its core count is the weight's
        # row count / 32 and NO program config can widen it. wt_ao is [128, 1024]: four
        # tile rows, four cores, 18 us to move 256 KB (14 GB/s). Anything under half the
        # grid is better tilized on the host, where it costs wall that is already
        # dominated by weight loading; the tall matrices keep the device path, where
        # host-tilizing megabytes would be the worse trade. The <=1-row case (biases and
        # norm scales) is the same rule at its extreme -- a device tilize would have to
        # val-pad 1 -> 32 rows on a SINGLE core.
        # `split` is n_dev when the mapper shards the ROW axis: the device tilize sees the
        # PER-CHIP tensor, so that is the row count that decides its core count.
        rows = (1 if t.dim() < 2 else int(t.shape[-2])) // max(1, split)
        if rows // 32 < _grid.x * _grid.y // 2:
            return ttnn.to_device(
                ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=mapper),
                device)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, mesh_mapper=mapper)

    def _rep(t):
        return _stage(t, ttnn.ReplicateTensorToMesh(device))

    def _shard(t, dim):
        # A shard on the ROW axis divides the per-chip height, and it is the per-chip
        # height that sets the device tilize's core count.
        rowdim = max(0, len(t.shape) - 2)
        return _stage(t, ttnn.ShardTensorToMesh(device, dim=dim),
                      split=n_dev if dim == rowdim else 1)

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
        Wo = Wc.new_zeros(int(Wc.shape[0]), 3 * shard_w * n_dev)
        bo = bc.new_zeros(3 * shard_w * n_dev)
        for i in range(n_dev):
            s = slice(i * shard_w, (i + 1) * shard_w)
            for j, o in enumerate((0, embed, 2 * embed)):
                # chip i's j-th projection lands in chunk (3i + j) -- writing to explicit
                # destination offsets keeps the per-chip grouping visible.
                d = slice((i * 3 + j) * shard_w, (i * 3 + j + 1) * shard_w)
                Wo[:, d] = Wc[:, o:o + embed][:, s]
                bo[d] = bc[o:o + embed][s]
        return _shard(Wo, 1), _shard(bo.reshape(1, 1, -1), 2)

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
            "b_ao": _rep(attn.c_proj.bias.detach().reshape(1, 1, -1) / n_dev),
            "wt_fc": _shard(mlp.c_fc.weight.detach(), 1),
            "b_fc": _shard(mlp.c_fc.bias.detach().reshape(1, 1, -1), 2),
            "wt_mo": _shard(mlp.c_proj.weight.detach(), 0),
            "b_mo": _rep(mlp.c_proj.bias.detach().reshape(1, 1, -1) / n_dev),
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
        # ONE tile row of work == the decode step, at ANY decode batch: B streams are B
        # ROWS of the same tile (B <= 32), so the test is the TILE count, not shape[-2] --
        # which for a batched [B,T,W] prefill is T and would send it down the decode path.
        rows = _mtiles(partial) * 32
        if rows <= 32:
            # The reasoning above weighed DISPATCH count, and under trace that is nearly
            # free: the per-token wall now tracks the SUM of device times, and this pair is
            # 200 us of the ~250 us each layer spends. Written as the explicit pair the two
            # halves become tunable -- the sum runs at LoFi (it is an add into an fp32
            # accumulator, no multiplier precision to lose) and each half gets a second
            # worker core on the same ethernet link. num_links stays 1: raising the LINK
            # count on this 1x8 linear fabric hangs the collective outright.
            try:
                sc = ttnn.reduce_scatter(
                    partial, dim=2, num_links=1, topology=ttnn.Topology.Linear,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=_ccl_kcfg, num_workers_per_link=2)
                return ttnn.all_gather(sc, dim=2, num_links=1,
                                       topology=ttnn.Topology.Linear,
                                       memory_config=ttnn.L1_MEMORY_CONFIG,
                                       num_workers_per_link=2)
            except Exception:  # noqa: BLE001 - fall back to the fused op
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
            if rows > 32:
                W = int(partial.shape[-1])
                ncores = max(1, min(32, W // 32))
                cfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(7, ncores - 1),
                                                                 (ncores - 1) // 8))}),
                        # shard height is the tensor's TOTAL padded rows (B*T for a batched
                        # prefill), not shape[-2]: a width shard splits only the last dim.
                        [32 * max(1, (rows + 31) // 32), W // ncores],
                        ttnn.ShardOrientation.ROW_MAJOR))
                partial = ttnn.to_memory_config(partial, cfg)
            # num_workers_per_link is the OCCUPANCY knob for a CCL (worker cores on the SAME
            # ethernet link -- a different thing from num_links, which HANGS this 1x8 linear
            # fabric). It is ROW-COUNT-AWARE, and this branch is the multi-row one: at decode
            # the tensor is ONE tile row, there is no payload for a second worker to overlap,
            # and the extra core is pure setup on an already latency-bound collective -- but
            # decode never gets here (it returned the fused all_reduce above). At T rows there
            # IS payload to overlap, so both halves take the second worker.
            # knob:shard, GATHER half. The scatter half above already READS a width-sharded
            # input, but it still WROTE one interleaved L1 buffer -- and that buffer is the
            # all_gather's input, which is why the gather still profiles cores=1 while the
            # scatter does not. Placing the scatter's OUTPUT width-sharded hands the gather a
            # per-core shard for free: it is the reduce_scatter's own output placement, not
            # an added reshard, so it costs no extra op. The scatter narrows the last dim by
            # n_dev, so the sharded width here is W/n_dev, not W.
            scfg = ttnn.L1_MEMORY_CONFIG
            if rows > 32:
                sw = int(partial.shape[-1]) // n_dev
                sc = max(1, min(32, sw // 32))
                scfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(7, sc - 1),
                                                                 (sc - 1) // 8))}),
                        [32 * max(1, (rows + 31) // 32), sw // sc],
                        ttnn.ShardOrientation.ROW_MAJOR))
            scattered = ttnn.reduce_scatter(
                partial, dim=2, num_links=1, topology=ttnn.Topology.Linear,
                memory_config=scfg, compute_kernel_config=_ccl_kcfg,
                num_workers_per_link=2)
            return ttnn.all_gather(scattered, dim=2, num_links=1,
                                   topology=ttnn.Topology.Linear,
                                   memory_config=ttnn.L1_MEMORY_CONFIG,
                                   num_workers_per_link=2)
        except Exception:  # noqa: BLE001 - fall back to the fused op
            return ttnn.all_reduce(partial, num_links=1, topology=ttnn.Topology.Linear,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)

    def _mlp(x, L):
        h = _ln(x, L["ln2"][0], L["ln2"][1], L["ln2"][2])
        ff = _lin(h, L["wt_fc"], L["b_fc"], act=_GELU_TANH)
        return _add_l1(x, _reduce(_lin_l1(ff, L["wt_mo"], L["b_mo"])))

    def _head(x):
        # Both lm_head norms stay sharded: the first because its consumer is the second
        # norm, the second because the lm_head projection is the ONE matmul here with no
        # tuned program config to satisfy (its N is 1026, which tiles to 33 -- prime-ish
        # and not worth a config), so it can take the sharded activation on ttnn's default
        # routing and the reshard disappears entirely.
        x = _ln(x, ln_f_w, ln_f_b, ln_f_eps, keep_sharded=True)
        x = _ln(x, fn_w, fn_b, fn_eps, keep_sharded=True)
        return _lin(x, lm_w, lm_b)

    def _block(x, L, T, kv_sink=None):
        # --- attention ---
        # ONE fused sharded qkv matmul + the multicore head shuffle. The hand-rolled
        # reshape([1,T,h,d]) + permute split used to be three separate matmuls whose
        # reshape SPLITS the last tile dim -- ttnn services that with an untilize +
        # SINGLE-CORE retilize (the profile's grid=tiny TilizeWithValPadding, ~62 us
        # per call, 4 per block). nlp_create_qkv_heads / nlp_concat_heads do the same
        # shuffle as one multicore device op, so no layout round-trip happens at all.
        h = _ln(x, L["ln1"][0], L["ln1"][1], L["ln1"][2])
        # B = DECODE BATCH: the prefill/full-sequence activation is [B, T, dim], i.e. B
        # INDEPENDENT streams stacked on the leading axis. Batch is a separate axis from
        # the TP-sharded weight axis -- nothing about the sharding or the collectives
        # changes, the rows just multiply.
        B = int(x.shape[0])
        qkv = _lin(h, L["wt_qkv"], L["bt_qkv"])                # [B, T, 3*shard_w]
        qkv = ttnn.reshape(qkv, [B, 1, T, 3 * shard_w])        # leading-dim view, no repack
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=heads_pc, num_kv_heads=heads_pc, transpose_k_heads=False)
        if kv_sink is not None:
            # K/V for the WHOLE padded prefix, in exactly the [B,h,C,d] layout the decode
            # attention wants -- so the cache is just what prefill already computed, and it
            # already holds B independent sequences, one per cache slot.
            kv_sink.append((k, v))
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scaling, compute_kernel_config=kcfg)
        ctx = ttnn.reshape(ttnn.experimental.nlp_concat_heads(ctx), [B, T, shard_w])
        x = _add_l1(x, _reduce(_lin_l1(ctx, L["wt_ao"], L["b_ao"])))
        return _mlp(x, L)

    def forward(emb, *_, **__):
        """Full-sequence (prefill) forward. emb ``[B, T, dim]`` -> logits ``[B, T, vocab]``;
        B independent streams in ONE program (B=1 is just the degenerate case)."""
        T = int(emb.shape[-2])
        x = emb
        for L in layers:
            x = _block(x, L, T)
        return _head(x)                        # [B, T, vocab]

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
        """Run the prefix once, KEEPING every layer's K/V. Returns the prefix logits.

        emb is ``[B, C, dim]``, so each layer's kept K/V is ``[B, heads, C, head_dim]`` --
        B independent sequences, one cache slot per batch row, seeded for free by the
        prefill's own head split. Returns ``[B, C, vocab]``."""
        T = int(emb.shape[-2])
        kv = []
        x = emb
        for L in layers:
            x = _block(x, L, T, kv_sink=kv)
        _kv["kv"], _kv["T"] = kv, T
        return _head(x)

    def decode_one(row_emb, pos):
        """ONE cached AR step for ALL B decode streams at once.

        row_emb: ``[1, B, dim]`` -- the B streams' new tokens as B ROWS of one tile
        (B <= 32), which is the layout every decode op here wants ("users" on dim 2 of the
        4-D qkv view). pos: a DEVICE tensor of B positions, so the step is
        trace-capturable (a Python int would bake a stale constant into the trace) AND
        each stream indexes its OWN cache slot. Returns ``[1, B, vocab]``.

        ONE program serves all B streams: the projections carry B rows, the fused cache
        write takes B update indices, and decode-SDPA reads B independent cache slots.
        There is no python loop over streams anywhere in here."""
        B = int(row_emb.shape[-2])
        x = row_emb
        for L, (kc, vc) in zip(layers, _kv["kv"]):
            h = _ln(x, L["ln1"][0], L["ln1"][1], L["ln1"][2])
            qkv = _lin(h, L["wt_qkv"], L["bt_qkv"])                     # [1,B,3*shard_w]
            # DECODE-SPECIFIC head shuffle. The generic nlp_create_qkv_heads emits
            # [1,h,S,d], but every consumer here (cache write, decode attention) wants
            # batch-major [1,S,h,d] and SHARDED -- which cost three ttnn.permute calls
            # plus two to_memory_config reshards per block. With heads_pc=2 and S=1 those
            # permutes move a 2x64 payload but pay a full untilize + VAL-PADDED retilize
            # each (both 2 and 1 pad to a 32-row tile), which is where the profile's
            # grid=tiny TilizeWithValPadding/UntilizeWithUnpadding time comes from.
            # nlp_create_qkv_heads_decode emits that exact layout, already sharded.
            # nlp_create_qkv_heads_decode calls dim 2 "users": [1, 1, B, 3*shard_w] with
            # dims 0/1 both 1 (it TT_FATALs otherwise). The WIDTH shard spec needs no
            # change for any B <= 32 -- the B user rows pad into the same single tile row.
            qkv = _shard_row(ttnn.reshape(qkv, [1, 1, B, 3 * shard_w]))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=heads_pc, num_kv_heads=heads_pc,
                overlap_qk_coregrid=False)                              # [1,B,h,d]
            # K and V must occupy non-overlapping core ranges for the FUSED write; the
            # decode head op already places them apart, so one dispatch does both.
            # k/v are [1,B,h,d] (batch on dim 1) and the cache is [B,h,C,d] (batch on
            # dim 0): the op cross-checks input.padded_shape[1] == cache.padded_shape[0]
            # == len(update_idxs), so ONE dispatch writes all B streams' slots, each at
            # its own position.
            ttnn.experimental.paged_fused_update_cache(kc, k, vc, v, update_idxs_tensor=pos)
            ctx = ttnn.transformer.scaled_dot_product_attention_decode(
                q, kc, vc, is_causal=True,
                cur_pos_tensor=pos, scale=scaling, compute_kernel_config=_sdpa_dec_kcfg)
            # The ctx side stays on the generic permute + concat. Its decode-specific
            # twin (nlp_concat_heads_decode) asserts a SHARDED input, and sdpa_decode
            # refuses to produce one for this head configuration ("Sharded output not
            # supported for GQA"), so chaining them would need a reshard that costs
            # more than the two ops it removes.
            ctx = ttnn.reshape(
                ttnn.experimental.nlp_concat_heads(ttnn.permute(ctx, [0, 2, 1, 3])),
                [1, B, shard_w])
            x = _add_l1(x, _reduce(_lin_l1(ctx, L["wt_ao"], L["b_ao"])))
            x = _mlp(x, L)
        return _head(x)

    forward.prefill_cache = prefill_cache
    forward.decode_one = decode_one
    forward.cache_len = lambda: int(_kv.get("T", 0))
    return forward
