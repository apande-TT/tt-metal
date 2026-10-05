# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mistral_decoder_layer` -- one text-backbone `MistralDecoderLayer`
(`model.layers.0`).

    h = x + self_attn(input_layernorm(x));   out = h + mlp(post_attention_layernorm(h))

GQA 32 query heads over 8 KV heads, head_dim **128** (explicitly 128, not 3072/32=96), hidden_dim
9216, `norm_eps` 1e-5, no biases. RoPE is `rotate_half` against the `(cos, sin)` the caller passes;
the harness marshals them onto the device, because the runtime native probe counts every torch call
the forward makes and graduates only at zero -- a `ttnn.from_torch` in here alone costs three.

Causality comes from SDPA's `is_causal`, so the additive `attention_mask` argument is accepted and
ignored. That matches the reference for an unpadded batch; a mask carrying real PADDING would need
to be fed through as an `attn_mask` instead.

TWO phases, one set of weights. With no `kv_cache` this is exactly the graduated prefill body.
Given one it ALSO seeds it from its own post-RoPE k/v, and `decode=True` then runs the cached
single-token path, which reads the resident history instead of recomputing it. The no-cache path is
untouched, so the per-component PCC test still measures the same arithmetic it graduated on."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import (
    cpp_ctx_merge,
    cpp_down_dec,
    cpp_kv_join,
    cpp_pv_dec,
    cpp_qkv_rope,
    cpp_qkv_rope_dec,
    cpp_rope_dec,
    cpp_scores_dec,
    cpp_sqmean,
    cpp_swiglu_mm,
    cpp_tail_rows,
    ttl_kv,
    ttl_swiglu,
)

# SDPA takes bfloat16 and nothing wider (`sdpa_device_operation.cpp:43`), and the KV cache is read
# by the same op family, so q/k/v and the cache are bf16 while the residual stream stays float32.
_SDPA_DTYPE = ttnn.bfloat16
# The resident KV cache: every decode step streams all of it through both attention bmms.
_CACHE_DTYPE = ttnn.bfloat8_b

# `ttnn.linear` on its DEFAULTS leaves `fp32_dest_acc_en` off, which rounds the matmul
# accumulator to bfloat16 at every step even though the activations are float32. Every linear
# here passes this instead; `mistral_model.py` carries the identical config for the identical
# arithmetic, and without it the two ports of the same layer do not agree with each other.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# Tall (prefill) linears are compute-bound, so they run at LoFi rather than the HiFi4 the
# rest of this file uses; the one-token decode linears are weight-bandwidth-bound and keep HiFi4.
_TALL_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
)
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_L1_BUDGET = 1_100_000


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _mcast_cfg(x, w, rows, out_dtype):
    """A full-grid 2D-multicast program config for a tall `[rows, K] x [K, N]` linear, or None.

    M goes over the grid rows and N over the grid columns. Per-core M/N are searched a few tiles
    above the minimum (a slightly larger block often divides into better subblocks), and when the
    whole per-core output does not fit L1 it is split into out-blocks. Ranked by per-core work,
    then the tiles each core re-reads across out-blocks, then a K-block of at least 4, then subblock
    area (16-bit DEST allows 8 tiles).
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    xs, ws = size(x.dtype), size(w.dtype)
    # 16-bit DEST: no separate float32 accumulation buffer beside the output block.
    os_ = size(out_dtype)
    best = None
    for pm in range(-(-mt // gy), -(-mt // gy) + 5):
        if -(-mt // pm) > gy:
            continue
        for pn in range(-(-nt // gx), -(-nt // gx) + 5):
            if -(-nt // pn) > gx:
                continue
            for bh in _divisors(pm):
                for bw in _divisors(pn):
                    kb = next(
                        (
                            c
                            for c in (8, 4, 2, 1)
                            if kt % c == 0 and bh * bw * os_ + 2 * c * (bh * xs + bw * ws) <= _L1_BUDGET
                        ),
                        None,
                    )
                    if kb is None:
                        continue
                    sub = max(
                        (
                            (h, s)
                            for h in range(1, 9)
                            for s in range(1, 9)
                            if h * s <= 8 and bh % h == 0 and bw % s == 0
                        ),
                        key=lambda hs: (hs[0] * hs[1], hs[1]),
                    )
                    reads = kt * (pm * (pn // bw) + pn * (pm // bh))
                    score = (pm * pn, reads, -min(kb, 4), -sub[0] * sub[1], -kb)
                    if best is None or score < best[0]:
                        best = (score, pm, pn, bh, bw, kb, sub)
    if best is None:
        return None
    _, pm, pn, bh, bw, kb, sub = best
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        out_block_h=bh,
        out_block_w=bw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
    )


def _row_cfg(x, w, out_dtype):
    """A 1D in0-multicast config for a ONE-tile-row (decode) linear, or None.

    Every core owns a slice of N and streams only its own weight columns while the single
    activation row is multicast; the widest K block that fits L1 keeps the multicast rounds few.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    kt, nt = int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    per_n = next(p for p in range(-(-nt // (gx * gy)), nt + 1) if nt % p == 0)
    if per_n > 2:  # wide N (gate/up): ttnn's own choice measured faster
        return None
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    fixed = per_n * (size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096))
    kb = next(
        (
            c
            for c in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
            if kt % c == 0 and fixed + 2 * c * (size(x.dtype) + per_n * size(w.dtype)) <= _L1_BUDGET
        ),
        None,
    )
    if kb is None:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=1,
        out_subblock_w=max(s for s in range(1, 5) if per_n % s == 0),
        per_core_M=1,
        per_core_N=per_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _short_cfg(x, w, rows, out_dtype, fp32_dest):
    """A 1D in0-multicast config for a SHORT (4-7 tile row) linear -- the 160-row shared prefix.

    ttnn's default there streams K two tiles at a time with 1x1 subblocks (144 multicast rounds at
    K=9216) and reached ~22% of DRAM bandwidth. As in `_row_cfg`, every core owns a slice of N and
    streams only its own weight columns while the few activation rows are multicast, here in
    16-tile K blocks (18 rounds at K=9216). The caller's compute config is kept (HiFi4 + float32
    DEST for the prefix every sample attends to), so a float32 DEST caps the subblock at 4 tiles.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    per_n = next(p for p in range(-(-nt // (gx * gy)), nt + 1) if nt % p == 0)
    if per_n > 2:  # wide N (the composed gate/up): ttnn's own choice measured faster (105 vs 118 us)
        return None
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    fixed = mt * per_n * (size(out_dtype) + (4096 if fp32_dest else 0))
    kb = next(
        (
            c
            for c in (16, 8, 6, 4, 3, 2, 1)
            if kt % c == 0 and fixed + 2 * c * (mt * size(x.dtype) + per_n * size(w.dtype)) <= _L1_BUDGET // 2
        ),
        None,
    )
    if kb is None:
        return None
    cap = 4 if fp32_dest else 8
    sub = max(
        ((h, s) for h in _divisors(mt) for s in _divisors(per_n) if h * s <= cap and (h == 1 or s == per_n)),
        key=lambda hs: (hs[0] * hs[1], hs[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        per_core_M=mt,
        per_core_N=per_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _lin(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, S, K]` activation against a 2-D weight runs as B separate `S x K x N` matmuls that
    each re-read the whole weight from DRAM; `[1, 1, B*S, K]` is one matmul that reads it once.
    Tall results (>= 8 tile rows) also get a hand-sized full-grid program config, short ones
    (4-7 tile rows) a 1D in0-multicast one.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    rows = lead * shape[-2]
    if rows >= 256 and rows % 32 == 0 and "program_config" not in kwargs:
        cfg = _mcast_cfg(x, w, rows, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
        kwargs["compute_kernel_config"] = _TALL_COMPUTE
    elif rows == 32 and "program_config" not in kwargs:
        cfg = _row_cfg(x, w, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
    elif 128 <= rows < 256 and rows % 32 == 0 and "program_config" not in kwargs:
        fp32_dest = bool(getattr(kwargs.get("compute_kernel_config"), "fp32_dest_acc_en", False))
        cfg = _short_cfg(x, w, rows, kwargs.get("dtype") or x.dtype, fp32_dest)
        if cfg is not None:
            kwargs["program_config"] = cfg
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, rows, shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


class _Like:
    """What `_short_cfg` reads of an activation (its device and dtype), before the tensor exists."""

    def __init__(self, device, dtype):
        self._device, self.dtype = device, dtype

    def device(self):
        return self._device


def _down_k_split(device, x_dtype, rows, k, w, out_dtype, fp32_dest):
    """`_short_cfg`'s 1D config for a SHORT down projection and the WIDTH-SHARDED L1 layout of its
    activation in exactly that config's K blocks, or None."""
    cfg = _short_cfg(_Like(device, x_dtype), w, rows, out_dtype, fp32_dest)
    kb = getattr(cfg, "in0_block_w", None)
    kt = k // 32
    grid = device.compute_with_storage_grid_size()
    if not (kb and kt % kb == 0 and kt // kb <= int(grid.x) * int(grid.y)):
        return None
    mem = ttnn.create_sharded_memory_config(
        shape=(rows, kb * 32),
        core_grid=ttnn.num_cores_to_corerangeset(kt // kb, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return cfg, mem


def _down_short(x, w, **kwargs):
    """The down projection; a SHORT one (4-7 tile rows, the 160-row prefix) reads its activation
    WIDTH-SHARDED in L1 in exactly the K blocks `_short_cfg`'s 1D multicast streams, so each block is
    multicast from the core holding it instead of ONE sender reading the whole 2.9 MB activation.
    Same config, same K order: the arithmetic is unchanged. An activation its producer already wrote
    in that layout is read as it is."""
    shape = [int(d) for d in x.shape]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    if 128 <= rows < 256 and rows % 32 == 0 and "program_config" not in kwargs:
        fp32_dest = bool(getattr(kwargs.get("compute_kernel_config"), "fp32_dest_acc_en", False))
        split = _down_k_split(x.device(), x.dtype, rows, shape[-1], w, kwargs.get("dtype") or x.dtype, fp32_dest)
        if split is not None:
            cfg, mem = split
            ready = (
                x.is_sharded()
                and len(shape) == 4
                and shape[:2] == [1, 1]
                and list(x.memory_config().shard_spec.shape) == list(mem.shard_spec.shape)
                and x.memory_config().shard_spec.grid == mem.shard_spec.grid
            )
            xs = x if ready else ttnn.to_memory_config(ttnn.reshape(x, [1, 1, rows, shape[-1]]), mem)
            kwargs["program_config"] = cfg
            y = ttnn.linear(xs, w, **kwargs)
            if not ready:
                ttnn.deallocate(xs)
            return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])
    return _lin(x, w, **kwargs)


def _last_of_each(x, every):
    """Row `every - 1` of each group of `every` rows of `x [1, H, R, D]`, as `[1, H, R // every, D]`.

    The rows are `every` apart and off-tile, so they are picked ROW_MAJOR (the stack's own
    `_last_tail_rows` picks the same rows of the residual)."""
    _, heads, rows, width = (int(d) for d in x.shape)
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    groups = ttnn.reshape(rm, [heads, rows // every, every, width])
    picked = ttnn.slice(groups, [0, 0, every - 1, 0], [heads, rows // every, every, width])
    ttnn.deallocate(rm)
    return ttnn.to_layout(ttnn.reshape(picked, [1, heads, rows // every, width]), ttnn.TILE_LAYOUT)


def _kept_rows_lin(x, w, full_rows, **kwargs):
    """A linear over a few KEPT rows of a tall (`full_rows`) activation, computed exactly as the tall
    linear computes them: the 2D config's own K block and the tall compute config, here as a 1D
    in0 multicast over the kept tile rows. Same K blocks, same accumulation: same bits per row."""
    out_dtype = kwargs.get("dtype") or x.dtype
    cfg2d = _mcast_cfg(x, w, full_rows, out_dtype)
    rows = 1
    for d in [int(s) for s in x.shape][:-1]:
        rows *= d
    if cfg2d is None or rows % 32:
        return _lin(x, w, **kwargs)
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    nt = int(w.shape[-1]) // 32
    per_n = next(p for p in range(-(-nt // (gx * gy)), nt + 1) if nt % p == 0)
    kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=cfg2d.in0_block_w,
        out_subblock_h=1,
        out_subblock_w=max(s for s in range(1, 5) if per_n % s == 0),
        per_core_M=rows // 32,
        per_core_N=per_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    kwargs["compute_kernel_config"] = _TALL_COMPUTE
    return ttnn.linear(x, w, **kwargs)


def _merge_heads(a, w, out_dtype, compute):
    """`nlp_concat_heads` of the attention output `a` `[1, H, S, D]`, laid out for its o_proj `w`.

    Interleaved, the merge deals out one 32-row tile per core, so the 160-row prefix ran on 5
    cores. A SHORT one (4-7 tile rows) is height-sharded instead, as many heads per core as fill one
    of `_short_cfg`'s K blocks, and every core merges its heads into its own column block of a
    WIDTH-sharded result -- the layout `_down_short` feeds the 1D multicast, so o_proj multicasts
    each K block from the core holding it. Same config, same K order: the arithmetic is unchanged.
    """
    b, n_heads, rows, head_dim = (int(d) for d in a.shape)
    if b == 1 and 128 <= rows < 256 and rows % 32 == 0 and head_dim % 32 == 0:
        fp32_dest = bool(getattr(compute, "fp32_dest_acc_en", False))
        cfg = _short_cfg(a, w, rows, out_dtype or a.dtype, fp32_dest)
        kb = getattr(cfg, "in0_block_w", None)
        per_core = kb * 32 // head_dim if kb and (kb * 32) % head_dim == 0 else 0
        grid = a.device().compute_with_storage_grid_size()
        if per_core and n_heads % per_core == 0 and n_heads // per_core <= int(grid.x) * int(grid.y):
            cores = ttnn.num_cores_to_corerangeset(n_heads // per_core, grid, row_wise=True)
            heads_cfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [per_core * rows, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
            )
            merged_cfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [rows, per_core * head_dim], ttnn.ShardOrientation.ROW_MAJOR),
            )
            heads = ttnn.to_memory_config(a, heads_cfg)
            ttnn.deallocate(a)
            merged = ttnn.experimental.nlp_concat_heads(heads, memory_config=merged_cfg)
            ttnn.deallocate(heads)
            return merged
    grid = a.device().compute_with_storage_grid_size()
    if b == 1 and rows >= 256 and rows % 32 == 0 and n_heads <= int(grid.x) * int(grid.y):
        # A TALL one (the 640-row tail) feeds a 2D multicast, which wants it interleaved; still,
        # one head per core merges on 32 cores where the interleaved op deals out 20 tile rows.
        cores = ttnn.num_cores_to_corerangeset(n_heads, grid, row_wise=True)
        heads_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [rows, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        merged_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [rows, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        heads = ttnn.to_memory_config(a, heads_cfg)
        ttnn.deallocate(a)
        merged = ttnn.experimental.nlp_concat_heads(heads, memory_config=merged_cfg)
        ttnn.deallocate(heads)
        out = ttnn.to_memory_config(merged, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return out
    return ttnn.experimental.nlp_concat_heads(a, memory_config=ttnn.L1_MEMORY_CONFIG)


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if layout == ttnn.TILE_LAYOUT and t.numel() <= 262144:
        # A small tensor (a gamma, a bias, a layer scale) tilizes on the HOST: tilized on the device it is
        # a whole single-core op (~80 us for a 1 x 1024 float32 row) at load time.
        mapper = ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None
        return ttnn.to_device(ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper), device)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _norm_weight(norm, device):
    """Gamma as `[1, 1, 1, dim]` float32 TILE -- the form `_rms_norm`'s final multiply takes."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1), device, dtype=ttnn.float32)


_STATS_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
)


def _sq_mean(x):
    """`mean(x^2, -1)` of a float32 `x`, as a square then a row-mean reduce at every height.

    A tall `x` used to take `rms_norm_pre_all_gather` (one read of `x`, float32 row sums); measured
    on the 640-row tail it ran ~71 us a norm (65 us on its 20 cores, one per tile row, plus the
    slice and scale) where the square into L1 and the one-core-per-tile-row mean take ~45 us."""
    shape = [int(d) for d in x.shape]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    # x^2 and its row means live in L1 -- the mean is a one-core-per-tile-row reduce, which reads x^2
    # back faster from L1 -- and x^2 is freed before any other op runs (~72 KB a core at 640 rows).
    l1 = ttnn.L1_MEMORY_CONFIG
    # A one-tile-row (decode-step) mean takes the FPU reduce path (x^2 truncated to TF32 on the way in)
    # instead of the accurate SFPU one; the prefill's taller norms keep the accurate path.
    if rows > 32 and cpp_sqmean.supports(x):
        # cpp: the square and the row mean as ONE generic_op on two cores a tile row (tt/cpp_sqmean) -- the same
        # float32 SFPU steps in the same order, so every mean is the stock one, bit for bit.
        return cpp_sqmean.sq_mean(x, memory_config=l1)
    return ttnn.mean(
        ttnn.square(x, memory_config=l1),
        dim=-1,
        keepdim=True,
        memory_config=l1,
        fast_and_approximate_mode=rows <= 32,
    )


def _rms_norm(x, gamma, eps, dtype=None, memory_config=None):
    """`x * rsqrt(mean(x^2) + eps) * gamma`, spelled out, entirely in float32.

    NOT `ttnn.rms_norm`. On the real layer-0 input the stock op lands at 9.65e-4 relative error
    against the reference and these four ops at 6.6e-8. The error is RELATIVE, so it rescales the
    whole branch that follows, and this stack runs two per layer over 26 layers; left in, the
    prefill's last hidden state came back at PCC 0.9971 and the 21-level acoustic quantiser
    downstream turned that into ~26% wrong audio codes.

    A zero pad row stays zero: `mean(x^2)` is 0 and `0 * rsqrt(eps)` is 0, so an off-tile
    sequence padded up to a tile multiple neither NaNs nor leaks into a real row.
    """
    scale = ttnn.add(_sq_mean(x), eps, activations=[ttnn.UnaryOpType.RSQRT])
    if gamma is None:  # folded into the consuming weights
        return ttnn.multiply(x, scale, dtype=dtype or ttnn.float32, memory_config=memory_config)
    return ttnn.multiply(ttnn.multiply(x, scale), gamma, dtype=dtype or ttnn.float32, memory_config=memory_config)


def _view4(x, dim):
    """`[..., seq, dim]` -> `([lead, 1, seq, dim], lead, seq, rank)`.

    THE LEADING BOUND IS READ OFF THE TENSOR. This used to be a literal
    `ttnn.reshape(x, [1, 1, seq, dim])`, which is right at the batch of 1 the per-component PCC
    harness feeds and wrong for every batched caller: at B=32 the reshape either raises on volume
    or -- worse, once a leading 1 is folded in elsewhere -- keeps only row 0 and silently drops
    samples 1..31. Everything downstream of here is per-row, so collapsing every leading axis into
    one `lead` is exact for `[B, S, D]`, `[B, 1, S, D]` and the decode stream's `[1, 1, B, D]`.
    """
    shape = [int(s) for s in x.shape]
    seq = shape[-2]
    lead = 1
    for size in shape[:-2]:
        lead *= size
    return ttnn.reshape(x, [lead, 1, seq, dim]), lead, seq, len(shape)


def _restore(x, lead, seq, rank, dim):
    """Put a `[lead, 1, seq, dim]` result back into the RANK the caller handed in."""
    return ttnn.reshape(x, [lead, seq, dim] if rank == 3 else [lead, 1, seq, dim])


def _broadcast4(t, seq, width):
    """A `(cos, sin)` table as `[lead, 1, seq, width]`, its leading bound read off the tensor."""
    volume = 1
    for size in t.shape:
        volume *= int(size)
    return ttnn.reshape(t, [volume // (seq * width), 1, seq, width])


def _rope(x, cos, sin, half):
    """`x * cos + rotate_half(x) * sin` -- the convention `apply_rotary_pos_emb` uses.

    `rotate_half` is `cat(-x[..., half:], x[..., :half])`; both halves are multiples of the tile
    width, so the two slices are tile-aligned.
    """
    ends = list(x.shape)
    lower = ttnn.slice(x, [0, 0, 0, 0], [ends[0], ends[1], ends[2], half])
    upper = ttnn.slice(x, [0, 0, 0, half], ends)
    rotated = ttnn.concat([ttnn.neg(upper), lower], dim=-1)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


def _rope_signed(x, cos, sin_signed, half):
    """`_rope` with the rotate-half sign already on the table: `x * cos + cat(x2, x1) * sin_signed`,
    where `sin_signed = cat(-sin1, sin2)` -- no negation of half of `x`."""
    ends = list(x.shape)
    lower = ttnn.slice(x, [0, 0, 0, 0], [ends[0], ends[1], ends[2], half])
    upper = ttnn.slice(x, [0, 0, 0, half], ends)
    l1 = ttnn.L1_MEMORY_CONFIG  # every term is read once, by the next op
    return ttnn.add(
        ttnn.multiply(x, cos, memory_config=l1),
        ttnn.multiply(ttnn.concat([upper, lower], dim=-1, memory_config=l1), sin_signed, memory_config=l1),
        memory_config=l1,
    )


def _rope_rows_cfg(shape, device, short=False):
    """The HEIGHT-sharded L1 config for a tall `[..., S, D]` RoPE input: the most cores that split its
    tile rows evenly (the fused RoPE then reads its rows locally and parallelises over the shard
    grid), or None for a short one. `short` takes a short one too -- the head split slices it there
    directly, so no reshard op is paid for it (the 160-row prefix's q: 80 cores, 2 rows each, where
    the interleaved op deals 1-2 rows to 110)."""
    if shape[-2] < 256 and not short:
        return None
    rows = 1
    for d in shape[:-1]:
        rows *= d
    tr = rows // 32
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    cores = next((c for c in range(gx * gy, 0, -1) if tr % c == 0), None)
    if not cores or cores < gx:
        return None
    return ttnn.create_sharded_memory_config(
        shape=(rows // cores, shape[-1]),
        core_grid=ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(gx, gy), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _rope_rows_sharded(x):
    """A tall `x` in `_rope_rows_cfg`'s layout (as is when the head split already sliced it there)."""
    if x.is_sharded():
        return x
    mem = _rope_rows_cfg([int(d) for d in x.shape], x.device())
    return x if mem is None else ttnn.to_memory_config(x, mem)


def _rope_prefill(q, k, cos, sin, half):
    """Prefill RoPE on q and k as ONE fused kernel each when the table is shared by the batch.

    `ttnn.experimental.rotary_embedding` computes the same `x * cos + rotate_half(x) * sin` in a
    single pass, where `_rope` spends six ops (two slices, a neg, a concat, two multiplies and an
    add) and a DRAM round-trip per op over `[B, H, S, head_dim]`. It wants a `[1, 1, S, head_dim]`
    table in the input's dtype; a per-row table (explicit position ids) keeps the spelled-out path.
    """
    if int(cos.shape[0]) != 1 or q.dtype != ttnn.bfloat16 or k.dtype != ttnn.bfloat16:
        l1 = ttnn.L1_MEMORY_CONFIG
        q = ttnn.to_memory_config(q, l1) if q.is_sharded() else q
        k = ttnn.to_memory_config(k, l1) if k.is_sharded() else k
        return _rope(q, cos, sin, half), _rope(k, cos, sin, half)
    if cos.dtype != ttnn.bfloat16:
        cos, sin = ttnn.typecast(cos, ttnn.bfloat16), ttnn.typecast(sin, ttnn.bfloat16)
    # SDPA is q's only reader and streams it chunk by chunk, so q lands in L1, and k does too
    # (interleaved, whatever layout the head split sliced it in).
    return (
        ttnn.experimental.rotary_embedding(_rope_rows_sharded(q), cos, sin, memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.experimental.rotary_embedding(_rope_rows_sharded(k), cos, sin, memory_config=ttnn.L1_MEMORY_CONFIG),
    )


def _bmm(a, b, transpose_b=False, memory_config=None):
    """Head-batched decode attention `a @ b` spread over the full grid.

    Without a program config the `[B, n_kv, groups, C]` products land on a handful of cores (probs @ V
    on four) and run the B * n_kv small matmuls back to back. The reuse config makes every
    (batch, kv-head) output block its own work unit, so they fan out across the grid.
    """
    # Ceil, not floor: the grouped query is `[B, n_kv, groups, head_dim]` with groups=4 rows, one padded tile.
    m, k, n = (-(-int(d) // 32) for d in (a.shape[-2], a.shape[-1], b.shape[-2 if transpose_b else -1]))
    grid = a.device().compute_with_storage_grid_size()
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=max(s for s in range(1, 5) if n % s == 0),
        per_core_M=m,
        per_core_N=n,
    )
    return ttnn.matmul(
        a, b, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=_COMPUTE, memory_config=memory_config
    )


def _sdpa_cfg(q):
    """Prefill SDPA on the full grid with the widest q/k chunk (<= 128) that divides the sequence.

    With no program config SDPA takes small default chunks, so each (batch, head) row is many
    tiny work units that re-stream K/V per chunk.
    """
    grid = q.device().compute_with_storage_grid_size()
    seq = int(q.shape[-2])
    chunk = next(c for c in (128, 64, 32) if seq % c == 0)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )


def _compact_sdpa_cfg(q):
    """The compact-tail SDPA config: the widest q chunk that still spreads over half the grid.

    Every Q chunk streams its head's WHOLE K/V (the 160-row prefix plus all 640 tail rows) once, so
    the op is K/V-read bound and the chunk count sets the traffic: 32 heads x 640 rows is 160 chunks
    at q_chunk 128 (and 320 at 64, which measured slower) but 64 at 320, still >= half of 110 cores.
    The k chunk is 4 tiles, the key axis padded up to it (SDPA fills the padded mask columns with
    -inf): a one-tile chunk rescales the whole running output after every key tile, and a 5-tile
    chunk (160, the widest that divides 800 keys) scrambled the masked output (PCC 0).
    """
    grid = q.device().compute_with_storage_grid_size()
    cores = int(grid.x) * int(grid.y)
    rows, seq = int(q.shape[0]) * int(q.shape[1]), int(q.shape[-2])
    q_chunk = max(c for c in range(32, seq + 1, 32) if seq % c == 0 and (c == 32 or 2 * rows * (seq // c) >= cores))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=q_chunk,
        k_chunk_size=128,
    )


def _per_sample(t, batch, real, padded):
    """A compact `[1, H, batch * real, D]` k/v as the cache's per-sample `[batch, H, padded, D]`.

    Row `b * real + i` is sample b's i-th tail position; the rows past `real` are zeros, which
    the decode slot mask keeps closed until a step writes them.
    """
    heads, width = int(t.shape[1]), int(t.shape[-1])
    rows = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [heads, batch, real, width])
    rows = ttnn.permute(rows, (1, 0, 2, 3))
    if padded != real:
        rows = ttnn.pad(rows, [(0, 0), (0, 0), (0, padded - real), (0, 0)], 0.0)
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


class _Split:
    """A compact prefill's k (or v) as the two pieces `_seed_cache` fills in place: the batch-1
    prefix `[1, H, P, D]` every sample shares, and the tail `[H * batch, 1, real, D]` in the cache
    dtype, one tile row per (head, sample). `slots` is the tail's padded width, so `filled` stays
    P + slots, as it was for the joined seed."""

    def __init__(self, prefix, tail, slots):
        self.prefix, self.tail, self.slots = prefix, tail, int(slots)


def _tail_rows(t, batch, real):
    """A compact `[1, H, batch * real, D]` k/v as `[H * batch, 1, real, D]` TILE in the cache dtype.

    Row `h * batch + b` is sample b's tail for head h, padded with zeros to its own tile row (the
    decode slot mask keeps those slots closed until a step writes them). Head-major, as the compact
    rows already are, so there is no permute: the tail fill's page table routes each row.
    """
    heads, width = int(t.shape[1]), int(t.shape[-1])
    rows = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [heads * batch, 1, real, width])
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT, dtype=_CACHE_DTYPE)


def _split_heads(qkv, n_heads, n_kv_heads, rope=False):
    """`[B, 1, S, (n_heads + 2 n_kv) * D]` -> q `[B, n_heads, S, D]`, k / v `[B, n_kv, S, D]`.

    The fused split parallelises over tile rows only (5 cores for the 160-row prefix, 20 for the
    640-row tail). Asked for every section as a Q head (`num_kv_heads=0`) it runs head-parallel
    over the whole grid, and three head-range slices then hand back exactly the same q, k and v.
    With `rope` (the fused RoPE reads q and k next) a tall q and k are sliced straight into
    `_rope_rows_cfg`'s height-sharded layout, so no separate reshard runs before the RoPE.
    """
    heads = ttnn.experimental.nlp_create_qkv_heads(
        qkv,
        num_heads=n_heads + 2 * n_kv_heads,
        num_kv_heads=0,
        transpose_k_heads=False,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )[0]
    b, _, s, d = (int(x) for x in heads.shape)
    # q and k are read once, by the RoPE ops right after, so they land in L1.
    l1 = ttnn.L1_MEMORY_CONFIG
    q_mem = _rope_rows_cfg([b, n_heads, s, d], heads.device(), short=True) if rope else None
    k_mem = _rope_rows_cfg([b, n_kv_heads, s, d], heads.device(), short=True) if rope else None
    q = ttnn.slice(heads, [0, 0, 0, 0], [b, n_heads, s, d], memory_config=q_mem or l1)
    k = ttnn.slice(heads, [0, n_heads, 0, 0], [b, n_heads + n_kv_heads, s, d], memory_config=k_mem or l1)
    # The tall tail's v is read at once too; the short prefix's is stashed across the stack.
    v = ttnn.slice(
        heads,
        [0, n_heads + n_kv_heads, 0, 0],
        [b, n_heads + 2 * n_kv_heads, s, d],
        memory_config=l1 if s >= 256 else None,
    )
    ttnn.deallocate(heads)
    return q, k, v


def _qkv_heads_rope(qkv, n_heads, n_kv_heads, position_embeddings, seq, head_dim, half):
    """The prefill head split + RoPE: `(q, k, v)` from the fused qkv projection.

    cpp: when the fused RoPE's form applies (a batch-1 bf16 table, bf16 qkv), ONE generic_op (tt/cpp_qkv_rope)
    reads q / k / v straight out of the fused rows and runs the STOCK RoPE compute kernel on them -- the head split,
    its three slices and the two rotary ops gone, the same bits. Else the stock split + `_rope_prefill`."""
    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = _broadcast4(cos, seq, head_dim)
        sin = _broadcast4(sin, seq, head_dim)
        if int(cos.shape[0]) == 1 and qkv.dtype == ttnn.bfloat16:
            cb, sb = cos, sin
            if cb.dtype != ttnn.bfloat16:
                cb, sb = ttnn.typecast(cb, ttnn.bfloat16), ttnn.typecast(sb, ttnn.bfloat16)
            if cpp_qkv_rope.supports(qkv, n_heads, n_kv_heads, cb, sb):
                return cpp_qkv_rope.apply(qkv, n_heads, n_kv_heads, cb, sb)
    q, k, v = _split_heads(qkv, n_heads, n_kv_heads, rope=position_embeddings is not None)
    if position_embeddings is not None:
        q, k = _rope_prefill(q, k, cos, sin, half)
    return q, k, v


def _prefill_sdpa(q, k, v, kv_cache, keep=None):
    """Prefill SDPA, `(attn, k, v)` -- the k/v to seed the cache with, or None to seed nothing.

    `keep` is the stack's LAST block reading only some rows: "none" (the prefix chain, read only for
    its stashed k/v -- no SDPA at all, attn None) or `(every, mask)` (the compact tail, read only at
    each sample's last row: SDPA runs on those q rows alone, against `mask`'s matching rows).

    With no `kv_cache["prefix_phase"]` this is the plain causal SDPA. A text stack running a SHARED
    prompt prefix once calls every layer twice: "stash" (the batch-1 prefix) keeps its k/v in the
    cache dict and seeds nothing, and "extend" (the per-row tail at the full batch) puts that k/v in
    front of its own for every row and attends through `kv_cache["prefix_mask"]` -- prefix columns
    open, the tail's own columns causal -- so the cache is seeded with the whole prompt.
    """
    phase = kv_cache.get("prefix_phase") if kv_cache is not None else None
    compact = kv_cache.get("prefix_compact") if kv_cache is not None else None
    if phase == "extend" and compact is not None:
        # COMPACT tail: `[1, H, batch * real, D]`, only the positions the rows disagree on. All
        # of it is one sequence behind the batch-1 prefix k/v, and the staged mask keeps each row
        # to the shared prefix plus its own sample's earlier tail positions.
        batch, real, padded = compact
        pk, pv = kv_cache.pop("prefix_kv")
        mask = kv_cache["prefix_mask"]
        if isinstance(keep, tuple):
            # Same k chunks, so each kept row's scores, softmax and output are bit for bit the ones
            # the whole tail's SDPA gives it.
            q, mask = _last_of_each(q, keep[0]), keep[1]
        # The joined k/v is SDPA's alone (the cache is built from pk/k below) and every q chunk
        # streams it whole, so it sits in L1 (~1.6 MB each); the mask SDPA insists on reading from DRAM.
        if cpp_kv_join.supports(pk, k, pv, v):
            # cpp: both joined operands written by ONE generic_op (tt/cpp_kv_join), tile for tile.
            jk, jv = cpp_kv_join.join(pk, k, pv, v, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            jk = ttnn.concat([pk, k], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            jv = ttnn.concat([pv, v], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        a = ttnn.transformer.scaled_dot_product_attention(
            q,
            jk,
            jv,
            is_causal=False,
            attn_mask=mask,
            scale=1.0,
            program_config=_compact_sdpa_cfg(q),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if kv_cache.get("tail_fill") is not None and kv_cache.get("fill_tables") is not None:
            # SEEDED IN PLACE, NEVER JOINED: `_seed_cache` fills the prefix and the tail into the
            # resident cache separately, so the per-sample [batch, H, P + T, D] k/v is not built.
            if cpp_tail_rows.supports([k, v], batch, real, _CACHE_DTYPE):
                # structural: both tails regrouped TILE -> TILE in ONE generic_op (tt/cpp_tail_rows) instead of
                # an untilize, a ROW_MAJOR view and a tilize-with-padding for each of k and v.
                kt, vt = cpp_tail_rows.apply([k, v], batch, real, _CACHE_DTYPE)
            else:
                kt, vt = _tail_rows(k, batch, real), _tail_rows(v, batch, real)
            return a, _Split(pk, kt, padded), _Split(pv, vt, padded)
        rep = ttnn.Shape([batch, 1, 1, 1])
        k = ttnn.concat([ttnn.repeat(pk, rep), _per_sample(k, batch, real, padded)], dim=2)
        v = ttnn.concat([ttnn.repeat(pv, rep), _per_sample(v, batch, real, padded)], dim=2)
        ttnn.deallocate(pk)
        ttnn.deallocate(pv)
        return a, k, v
    if phase == "extend":
        pk, pv = kv_cache.pop("prefix_kv")
        rep = ttnn.Shape([int(q.shape[0]), 1, 1, 1])
        k = ttnn.concat([ttnn.repeat(pk, rep), k], dim=2)
        v = ttnn.concat([ttnn.repeat(pv, rep), v], dim=2)
        ttnn.deallocate(pk)
        ttnn.deallocate(pv)
        a = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=kv_cache["prefix_mask"], scale=1.0, program_config=_sdpa_cfg(q)
        )
        return a, k, v
    if phase == "stash" and keep == "none":
        kv_cache["prefix_kv"] = (k, v)
        return None, None, None
    # The attention output's only reader is the head merge right after: L1.
    a = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=1.0, program_config=_sdpa_cfg(q), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    if phase == "stash":
        kv_cache["prefix_kv"] = (k, v)
        return a, None, None
    return a, k, v


def _decode_shard(device, rows, width):
    """HEIGHT-sharded over the batch, one user per core -- the decode op set's layout.

    `nlp_create_qkv_heads_decode`, decode-mode `rotary_embedding_hf` and
    `nlp_concat_heads_decode` are a matched set: each wants one 32-row tile per user, and the RoPE
    op rejects a merely-interleaved tensor outright, so this is part of the contract.
    """
    grid = device.compute_with_storage_grid_size()
    cols = min(int(grid.x), int(rows))
    while rows % cols:
        cols -= 1
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, int(width)),
        core_grid=ttnn.CoreGrid(y=rows // cols, x=cols),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# THE ZERO TAIL IS A PERSISTENT BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds the tensor on the host and enqueues a WRITE to get it onto the device, and a
# write is exactly what a captured trace cannot replay: capturing a prefill that seeded its cache
# this way died on `TT_FATAL: Writes are not supported during trace capture`. The tail is the same
# shape of the same zeros on every call, so it is created once per (device, shape) and reused --
# and because the pad is now shared it is NEVER deallocated by the caller, which used to free it
# after the concat. `flow_matching_audio_transformer` hoists its tile pad for the same reason.
_ZERO_TAIL = {}


def _zero_tail(device, b, h, rows, width):
    key = (id(device), b, h, rows, width)
    buf = _ZERO_TAIL.get(key)
    if buf is None:
        buf = ttnn.zeros([b, h, rows, width], dtype=_CACHE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        _ZERO_TAIL[key] = buf
    return buf


def _seed_cache(kv, k, v):
    """Hand the prefill's POST-RoPE k/v to the cache, widened out to `kv["capacity"]`.

    No second source of truth: the cache's first slots ARE the prefill's own k/v -- a zero tail
    concatenated on the sequence axis the first time, the same buffer filled in place after -- so
    the resident history cannot disagree with the prefill that produced it. The tail slots are
    never READ before they are written -- a decode step writes slot `position` and then attends to
    `[0, position]` -- so they only have to exist (and hold finite values).
    """
    if isinstance(k, _Split):
        return _seed_split(kv, k, v)
    _drop_shared_prefix(kv)  # this prefill writes every row: a shared prefix from an earlier one is stale
    capacity = int(kv.get("capacity") or 0)
    # The text stack hands back the previous prefill's buffers (same batch and capacity) with the
    # staged fill tables; this prefill's slots are then written into them IN PLACE by one
    # `paged_fill_cache` (block b = row b's C slots) instead of rebuilding all C around a zero tail.
    resident = kv.pop("resident", None) or (None, None)
    tables = kv.get("fill_tables")
    for (key, tensor), home in zip((("k", k), ("v", v)), resident):
        if tensor.dtype != _CACHE_DTYPE:
            tensor = ttnn.typecast(tensor, _CACHE_DTYPE)
        shape = [int(s) for s in tensor.shape]
        full = shape[:2] + [capacity, shape[-1]]
        if home is not None and [int(s) for s in home.shape] != full:
            ttnn.deallocate(home)
            home = None
        if tables is not None and capacity > shape[-2]:
            # The first prefill zero-fills the buffer; every prefill (the first too) seeds it by
            # the SAME fill, so the correctness gate covers the path the traced prefill replays.
            if home is None:
                home = ttnn.zeros(full, dtype=_CACHE_DTYPE, layout=ttnn.TILE_LAYOUT, device=tensor.device())
            ttnn.experimental.paged_fill_cache(home, tensor, tables[0], batch_idx_tensor=tables[1])
            if tensor is not k and tensor is not v:
                ttnn.deallocate(tensor)
            tensor = home
        else:
            if home is not None:
                ttnn.deallocate(home)
            if capacity > shape[-2]:
                pad = _zero_tail(tensor.device(), shape[0], shape[1], capacity - shape[-2], shape[-1])
                tensor = ttnn.concat([tensor, pad], dim=2)
            elif capacity and capacity < shape[-2]:
                raise ValueError(f"kv capacity {capacity} is shorter than the prefill's {shape[-2]}")
        stale = kv.get(key)
        if stale is not None:
            try:
                ttnn.deallocate(stale)
            except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
                pass
        kv[key] = tensor
    kv["filled"] = int(k.shape[-2])


def _seed_split(kv, k, v):
    """`_seed_cache` for a compact prefill: the prefix and the tail go into the resident cache by
    two in-place fills, and are never joined.

    The prefix is cast to the cache dtype at batch 1, repeated there (half the bytes of a bf16
    repeat) and filled into every sample's first P slots by the row-block fill tables. The tail is
    already one tile row per (head, sample), and viewing the cache as `[B * H * C/32, 1, 32, D]` --
    a zero-copy view, one block per tile row of one (sample, head) -- makes it a paged cache whose
    page table (`kv["tail_fill"]`, staged by the text stack) sends row (h, b) to sample b, head h,
    slot P. The permute to sample-major, the concat behind the repeated prefix and the full-width
    typecast the joined seed paid for are all gone.
    """
    capacity = int(kv["capacity"])
    resident = kv.pop("resident", None) or (None, None)
    rows_fill, tail_fill = kv["fill_tables"], kv["tail_fill"]
    tile = ttnn.TILE_SIZE
    _drop_shared_prefix(kv)
    shared = {}
    for (key, part), home in zip((("k", k), ("v", v)), resident):
        heads, prefix_rows, width = (int(part.prefix.shape[i]) for i in (1, 2, 3))
        batch = int(part.tail.shape[0]) // heads
        full = [batch, heads, capacity, width]
        if home is not None and [int(s) for s in home.shape] != full:
            ttnn.deallocate(home)
            home = None
        if home is None:
            home = ttnn.zeros(full, dtype=_CACHE_DTYPE, layout=ttnn.TILE_LAYOUT, device=part.tail.device())
        if _SHARED_PREFIX:
            # structural: the prefix every user shares stays ONE copy (DRAM, the cache's dtype) -- the C++
            # decode scores / P@V read their first rows from it (cpp_scores_dec / cpp_pv_dec `prefix`), so it is
            # never written into all `batch` users' rows; any other cache read fills it in first.
            narrow = ttnn.typecast(part.prefix, _CACHE_DTYPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            narrow = ttnn.typecast(part.prefix, _CACHE_DTYPE)
        ttnn.deallocate(part.prefix)
        if _SHARED_PREFIX and cpp_scores_dec.prefix_ok(home, narrow):
            shared[key] = narrow
        elif ttl_kv.supports(home, narrow):
            # The tt-lang seed: each prefix tile row read once and written into every sample's rows.
            ttl_kv.fill_prefix(home, narrow)
            ttnn.deallocate(narrow)
        else:
            shared = ttnn.repeat(narrow, ttnn.Shape([batch, 1, 1, 1]))
            ttnn.deallocate(narrow)
            ttnn.experimental.paged_fill_cache(home, shared, rows_fill[0], batch_idx_tensor=rows_fill[1])
            ttnn.deallocate(shared)
        tile_rows = ttnn.reshape(home, [batch * heads * capacity // tile, 1, tile, width])
        ttnn.experimental.paged_fill_cache(tile_rows, part.tail, tail_fill[0], batch_idx_tensor=tail_fill[1])
        ttnn.deallocate(part.tail)
        stale = kv.get(key)
        if stale is not None and stale is not home:
            try:
                ttnn.deallocate(stale)
            except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
                pass
        kv[key] = home
    if shared:
        kv["prefix_shared"] = shared
        if len(shared) != 2:  # only one of k / v qualified: fill it in like the other
            _materialize_prefix(kv)
    kv["filled"] = prefix_rows + k.slots


_SHARED_PREFIX = os.environ.get("VOXTRAL_SHARED_PREFIX", "1") == "1"


def _drop_shared_prefix(kv):
    shared = kv.pop("prefix_shared", None)
    for t in (shared or {}).values():
        try:
            ttnn.deallocate(t)
        except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
            pass


def _materialize_prefix(kv):
    """Write the shared prefix (kv["prefix_shared"]) into every user's first cache rows, for a read of the cache
    that is not the C++ decode scores / P@V's -- the fill `_seed_split` skipped."""
    shared = kv.pop("prefix_shared", None)
    if not shared:
        return
    rows_fill = kv.get("fill_tables")
    for key, narrow in shared.items():
        home = kv[key]
        if ttl_kv.supports(home, narrow):
            ttl_kv.fill_prefix(home, narrow)
        else:
            rep = ttnn.repeat(narrow, ttnn.Shape([int(home.shape[0]), 1, 1, 1]))
            ttnn.experimental.paged_fill_cache(home, rep, rows_fill[0], batch_idx_tensor=rows_fill[1])
            ttnn.deallocate(rep)
        ttnn.deallocate(narrow)


# The additive mask for one decode position, `[1, 1, 1, C]`: 0 through `position`, a large
# negative beyond it. The row comes off the `[C, C]` float32 table the text stack uploaded ONCE at
# build time and handed to every block in its `kv` dict -- built here it would be a torch call
# inside the forward, and a host write inside a captured trace.
_MASK_NEG = -1e9


def _decode_mask(kv_cache, position, cap):
    table = kv_cache.get("mask")
    if table is None:
        raise RuntimeError(
            "the decode step needs kv_cache['mask'] -- the [capacity, capacity] additive table "
            "the text stack stages at build time; without it the zero tail of the cache is "
            "attended as if it were real keys"
        )
    shared = kv_cache.get("mask_row")
    if shared is not None and shared[0] == int(position) and int(shared[1].shape[-1]) == cap:
        return shared[1]
    row = ttnn.slice(table, [int(position), 0], [int(position) + 1, cap])
    return ttnn.to_layout(ttnn.reshape(row, [1, 1, 1, cap]), ttnn.TILE_LAYOUT)


def _softmax(x, dim=-1):
    """`exp(x - max) / sum(exp(x - max))`, spelled out in three ops.

    NOT `ttnn.softmax`. Measured on this build against a float64 reference, the stock op's rows do
    not sum to 1: mean 0.9943, worst 0.9611, for ~1.8e-2 relative error -- and NO flag changes it
    (`numeric_stable=True` and `compute_kernel_config` all return the identical tensor). These
    three ops sit at 5.5e-8, and renormalising the stock op's output only reaches 2.1e-2, so its
    per-element values are wrong too, not just its sum.

    A softmax that does not sum to 1 ATTENUATES the attention output it weights. That reads as a
    NORM RATIO below 1 at a PCC of 0.9999, so a PCC-only gate cannot see it, and it is invisible
    in any layer whose residual is already large. The acoustic stubs beside this file spell the
    same three ops out for the same reason.
    """
    # The exp rides on the subtract as a post-activation: one pass over x instead of two.
    e = ttnn.subtract(x, ttnn.max(x, dim=dim, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    return ttnn.divide(e, ttnn.sum(e, dim=dim, keepdim=True))


def _swiglu_pairs(gate, up, tile=32):
    """`[K, N]` gate and up weights as ONE `[K, 2N]` weight of interleaved column-tile pairs
    `[gate_t0, up_t0, gate_t1, up_t1, ...]` -- the layout `minimal_matmul(fuse_swiglu=True)` reads
    to emit `silu(gate) * up` straight from the matmul."""
    rows, n = int(gate.shape[0]), int(gate.shape[-1])
    pairs = torch.stack([gate.reshape(rows, n // tile, tile), up.reshape(rows, n // tile, tile)], dim=2)
    return pairs.reshape(rows, 2 * n).contiguous()


def _ttl_swiglu_weights(gate, up, device):
    if not ttl_swiglu.enabled():
        return None
    return tuple(_from_torch(w.contiguous(), device, dtype=ttnn.bfloat16) for w in (gate, up))


def _fused_swiglu(h, w_gu, w_ttl=None, memory_config=None, w_cpp=None):
    """Prefill `silu(h @ Wg) * (h @ Wu)` as ONE matmul: no gate/up tensors are written and there
    is no separate multiply pass over them. `memory_config` is where the matmul writes its result
    (the down projection's K-block layout for the short prefix)."""
    if w_ttl is not None and ttl_swiglu.supports(h, w_ttl[0]):
        return ttl_swiglu.apply(h, *w_ttl)
    grid = h.device().compute_with_storage_grid_size()
    rows = 1
    for d in list(h.shape)[:-1]:
        rows *= int(d)
    # Half of each core's M share per block (vs the default 8 rows) halves the weight re-reads and
    # 18-tile N blocks split each core's 54 fused columns (27 output tiles x 2) into exactly 3 blocks
    # (12 left a half-empty 5th block); 4-tile K blocks keep the L1 footprint near 1 MB.
    share = -(-(rows // 32) // int(grid.y))
    # The fewest grid rows that keep that share: rows past ceil(M / share) would only compute padding
    # (32 M tiles over 10 rows is 4 each, and rows 8-9 held the pad).
    grid = ttnn.CoreCoord(int(grid.x), -(-(rows // 32) // share))
    # A share of <= 8 tiles fits one M block, so each core streams its weight columns ONCE.
    m_blk = share if share <= 8 else -(-share // 2)
    # At <= 2 M tiles a core the bf4_b weight's blocks are small enough for 8-tile K blocks
    # (half the K steps of 4) within the ~1 MB of L1 that still traces.
    wide = m_blk <= 2
    w_c = w_cpp if w_cpp is not None else w_gu
    if cpp_swiglu_mm.supports(h, w_c, 18, int(grid.x)):
        # cpp: the same arithmetic tile for tile (the same K blocks, in the same order), the weight streamed by every core
        # instead of minimal_matmul's grid.x column readers; handed to `memory_config` by one copy.
        gated = cpp_swiglu_mm.apply(h, w_c, 8 if wide else 4, 18, int(grid.x))
        if memory_config is None or memory_config == gated.memory_config():
            return gated
        out = ttnn.to_memory_config(gated, memory_config)
        ttnn.deallocate(gated)
        return out
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=m_blk,
        K_block_size=8 if wide else 4,
        N_block_size=18,
        subblock_h=1,
        subblock_w=6,
        compute_with_storage_grid_size=grid,
    )
    return ttnn.experimental.minimal_matmul(
        h,
        w_gu,
        fuse_swiglu=True,
        config=cfg,
        dtype=ttnn.bfloat16,
        compute_kernel_config=_TALL_COMPUTE,
        memory_config=memory_config,
    )


def build(device, torch_module):
    layer = torch_module
    attn = layer.self_attn
    mlp = layer.mlp

    dim = int(attn.q_proj.in_features)
    n_heads = int(attn.config.num_attention_heads)
    n_kv_heads = int(attn.config.num_key_value_heads)
    head_dim = int(attn.head_dim)
    half = head_dim // 2
    scale = float(attn.scaling)

    # Each RMSNorm's gamma is FOLDED into the input rows of the weights it feeds (`x*s*g @ W` ==
    # `x*s @ diag(g) W`), so the norm is one scaling pass over the residual, not two. The product
    # is formed in float32 and rounded to bfloat16 once.
    g_in_t = layer.input_layernorm.weight.detach().float().reshape(-1, 1)
    g_post_t = layer.post_attention_layernorm.weight.detach().float().reshape(-1, 1)

    def _folded(linear, g):
        return _from_torch((linear.weight.detach().float().transpose(0, 1) * g).contiguous(), device)

    wqkv = _from_torch(
        torch.cat(
            [
                # The attention scale folded into Wq (RoPE is linear, so it commutes).
                attn.q_proj.weight.detach().float().transpose(0, 1) * float(attn.scaling),
                attn.k_proj.weight.detach().float().transpose(0, 1),
                attn.v_proj.weight.detach().float().transpose(0, 1),
            ],
            dim=-1,
        )
        .mul(g_in_t)
        .contiguous(),
        device,
        dtype=ttnn.bfloat8_b,
    )
    wo = _from_torch(attn.o_proj.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    # Decode-only now (prefill runs the fused `w_gu` below): a one-token step streams these from
    # DRAM every step, so bf8_b halves what it reads.
    w_gate, w_up = (
        _from_torch((p.weight.detach().float().transpose(0, 1) * g_post_t).contiguous(), device, dtype=ttnn.bfloat8_b)
        for p in (mlp.gate_proj, mlp.up_proj)
    )
    # Prefill's fused SwiGLU weight, bf4_b against the bf16 norm output (minimal_matmul takes mixed
    # dtypes); decode keeps the separate gate/up above for its float32 activation.
    gu_pairs = _swiglu_pairs(
        mlp.gate_proj.weight.detach().float().transpose(0, 1) * g_post_t,
        mlp.up_proj.weight.detach().float().transpose(0, 1) * g_post_t,
    )
    w_gu = _from_torch(gu_pairs, device, dtype=ttnn.bfloat4_b)
    # The short prefix's C++ SwiGLU reads a copy with its tile columns re-laid bank-contiguous per core; the same
    # conversion of the same pairs, so every tile is minimal_matmul's.
    w_gu_cpp = cpp_swiglu_mm.relayout(
        gu_pairs,
        device,
        lambda t: _from_torch(t, device, dtype=ttnn.bfloat4_b),
        18,
        int(device.compute_with_storage_grid_size().x),
        8,
    )
    del gu_pairs
    w_ttl = _ttl_swiglu_weights(
        mlp.gate_proj.weight.detach().float().transpose(0, 1) * g_post_t,
        mlp.up_proj.weight.detach().float().transpose(0, 1) * g_post_t,
        device,
    )
    # bf8_b halves the weight both the prefill (LoFi) and decode down projections unpack.
    w_down = _from_torch(mlp.down_proj.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    w_down_cpp = cpp_down_dec.shard(mlp.down_proj.weight.detach().transpose(0, 1).contiguous(), device)
    g_in = None
    g_post = None
    eps_in = float(layer.input_layernorm.variance_epsilon)
    eps_post = float(layer.post_attention_layernorm.variance_epsilon)

    use_cpp_scores = cpp_scores_dec.claim()
    use_cpp_pv = cpp_pv_dec.claim()

    def _decode_attn(xn, position_embeddings, kv_cache, position):
        """ONE token per user, attending to the RESIDENT cache instead of recomputing the prefix.

        The stream arrives folded as `[1, 1, B, dim]`: `[B, 1, dim]` pads its middle dim out to a
        whole tile, so every op would touch 32x the data the step carries.
        """
        # THE USER COUNT IS THE VOLUME OVER THE WIDTH, not any single leading dim. A decode step
        # reaches here as `[1, 1, B, dim]` (the folded stream) or as `[B, 1, 1, dim]`, and reading
        # `shape[-2]` returns B for the first and 1 for the second -- which would quietly process
        # one user and drop the other 31.
        held = [int(size) for size in xn.shape]
        batch = 1
        for size in held[:-1]:
            batch *= size
        # FLOAT32 ALL THE WAY THROUGH. `_SDPA_DTYPE` is bfloat16 because SDPA takes nothing wider,
        # and this path no longer calls SDPA -- so the narrowing bought nothing and cost a bfloat16
        # ulp on every q, k and v. The three ops that forced it are gone with it:
        # `nlp_create_qkv_heads_decode` (its head split is three slices and a reshape, and it
        # hands back a HEIGHT-SHARDED tensor this path would only have to interleave again) and
        # decode-mode `rotary_embedding_hf` (which typecasts its input to bfloat16 outright --
        # `models/tt_transformers/tt/attention.py:664`). `_rope` is the SAME rotate-half the
        # prefill branch below runs, in float32, and every user in a step shares one position so
        # `cos`/`sin` are `[1, 1, 1, head_dim]` and broadcast.
        groups = n_heads // n_kv_heads
        flat = ttnn.reshape(xn, [1, 1, batch, dim])
        fused = _lin(
            flat, wqkv, dtype=ttnn.float32, compute_kernel_config=_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        # ONE data-movement op splits the fused qkv by user and head, float32 preserved, into the
        # decode shard layout (one user per core): q `[1, B, n_heads, head_dim]` and k / v in the
        # cache's own `[1, B, n_kv, head_dim]` -- the n_kv heads share ONE tile per user there, where
        # `[B, n_kv, 1, head_dim]` would pad every head to its own tile. It replaces an untilize and
        # three slice / reshape / tilize chains. v is already the cache update's input layout.
        shard = _decode_shard(fused.device(), batch, head_dim)
        rope_full = kv_cache.get("rope_full") if kv_cache is not None else None
        rope_signed = kv_cache.get("rope_signed") if kv_cache is not None else None
        if (
            position_embeddings is not None
            and rope_full is not None
            and rope_full[0] == int(position)
            and rope_signed is not None
            and rope_signed[0] == int(position)
            and cpp_qkv_rope_dec.supports(fused, n_heads, n_kv_heads, rope_full[1], rope_full[2], shard)
        ):
            # cpp: the head split, q's and k's moves and their RoPE in ONE generic_op (tt/cpp_qkv_rope_dec): q
            # interleaved and RoPE'd, k RoPE'd and v straight into the cache update's shard layout -- the same bits.
            q, k, v = cpp_qkv_rope_dec.apply(fused, n_heads, n_kv_heads, rope_full[1], rope_full[2], shard)
            ttnn.deallocate(fused)
            q_rows = -(-groups // 32) * 32
        else:
            q_s, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                fused,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                memory_config=shard,
            )
            ttnn.deallocate(fused)

            # The grouped query is `groups` rows of a 32-row tile. Zero-padding it to a LOGICAL full tile
            # costs nothing physically and lets every softmax reduction below skip its FillPad pass.
            q_rows = -(-groups // 32) * 32
            # q is RoPE'd with the heads as ROWS, `[B, 1, n_heads, head_dim]` (one tile per user), and
            # only then regrouped by kv head: the grouped `[B, n_kv, groups, head_dim]` form pads each
            # group of `groups` rows out to a 32-row tile, 8x the tiles for the six RoPE ops. The RoPE
            # ops read q and k interleaved.
            q = ttnn.reshape(ttnn.to_memory_config(q_s, ttnn.L1_MEMORY_CONFIG), [batch, 1, n_heads, head_dim])
            ttnn.deallocate(q_s)
            k_s = k
            k = ttnn.to_memory_config(k_s, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(k_s)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                signed = kv_cache.get("rope_signed") if kv_cache is not None else None
                if signed is not None and signed[0] == int(position):
                    full = kv_cache.get("rope_full")
                    if (
                        full is not None
                        and full[0] == int(position)
                        and cpp_rope_dec.supports([q, k], full[1], full[2])
                    ):
                        # cpp: q's and k's RoPE in ONE generic_op (tt/cpp_rope_dec) -- binary_ng's float32 SFPU
                        # multiplies and add in its order, the rotate-half a tile swap: the same bits, 1 op not 12.
                        q, k = cpp_rope_dec.apply([q, k], full[1], full[2])
                    else:
                        q = _rope_signed(q, cos, signed[1], half)
                        k = _rope_signed(k, cos, signed[1], half)
                else:
                    q = _rope(q, cos, sin, half)
                    k = _rope(k, cos, sin, half)
        # The C++ scores can read the RoPE'd q in place, gathering each kv head's `groups` rows into a
        # zeroed tile itself (the same grouped layout), when it also reads the cache in place.
        cap_kv = int(kv_cache["k"].shape[-2])
        span_kv = min(cap_kv, int(kv_cache.get("span") or cap_kv))
        q_in_place = (
            span_kv < cap_kv
            and use_cpp_scores
            and use_cpp_pv
            and cpp_scores_dec.supports_grouped(q, kv_cache["k"], span_kv, groups)
        )
        if not q_in_place:
            # The tilize writes zeros into the tile rows past `groups`; relabelling them logical is a
            # zero-cost view, where `ttnn.pad` re-filled the same zeros in a FillPad pass.
            full = ttnn.Shape([batch, n_kv_heads, q_rows, head_dim])
            q = ttnn.tilize_with_val_padding(
                ttnn.reshape(ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT), [batch, n_kv_heads, groups, head_dim]),
                full,
                0.0,
            )
            if q_rows != groups:
                q = ttnn.reshape(q, full, full)
        # A split prefill leaves dead slots before the tail: position p lives at slot p + offset.
        idxs = [int(position) + int(kv_cache.get("slot_offset", 0))] * batch
        # `paged_update_cache` wants the decode layout `[1, B, n_kv, head_dim]` AND it wants that
        # tensor HEIGHT-SHARDED, one user per core -- it is part of the decode op set even though
        # the rest of that set is gone from this path ("Expect input_tensor to be sharded"). The
        # k/v split above already produced that layout.
        for slot, tensor in (("k", k), ("v", v)):
            ttnn.experimental.paged_update_cache(
                kv_cache[slot],
                ttnn.to_memory_config(
                    tensor,
                    _decode_shard(tensor.device(), batch, head_dim),
                ),
                update_idxs=idxs,
            )
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # FLASH-DECODE ATTENUATES, so this path spells the attention out instead.
        # `scaled_dot_product_attention_decode` came back with its output norm SHORT of the
        # reference's at every layer -- measured against torch on this model, one decode step, the
        # reference fed the same cache: norm ratio 0.9478 / 0.9781 / 0.9888 / 0.9867 / 0.9895 /
        # 0.9942 at PCC 0.9999, i.e. almost pure attenuation, worst where the softmax is flattest.
        # It is a denominator that carries mass the numerator does not. That is invisible wherever
        # the residual is large, and this model puts its first three layers at hidden norms of
        # 1.1 / 2.6 / 4.0 before layer 3 jumps to 287, so the whole error lands there: the cached
        # decode step measured PCC 0.93 against the reference while THIS STACK'S OWN PREFILL path,
        # same weights and same token, measured 0.9999.
        #
        # The query heads are GROUPED BY KV HEAD rather than repeat_interleaved: reading q as
        # `[B, n_kv, groups, head_dim]` makes the batch dims line up with the cache's
        # `[B, n_kv, C, head_dim]`, so the whole thing is two batched matmuls and no cache
        # tensor is ever materialised `n_heads` times. `head // groups` IS the reference's
        # `repeat_kv` mapping, so the grouping is the same one HF uses.
        cap = int(kv_cache["k"].shape[-2])
        # The bmm reads the cache transposed in place; an explicit transpose re-wrote the whole
        # [B, n_kv, C, head_dim] K cache every step.
        # Scale the [B, n_kv, 32, head_dim] query, not the [B, n_kv, 32, C] scores: one pass over
        # a tensor C/head_dim times smaller.
        # The [B, n_kv, 32, C] float32 scores (~20 MB at B=32, C=608) and everything derived from
        # them live in L1: in DRAM the mask add and the exp pass each moved ~40 MB at ~265 GB/s,
        # and the two reductions and P@V read them back again, every layer of every step. About
        # 180 KB a core per tensor across the grid; at most two are alive at once.
        l1 = ttnn.L1_MEMORY_CONFIG
        # Only the first `span` slots (the text stack's tile-rounded run up to this step's slot) are
        # read: every slot past it is masked, so its exp is exactly 0 and it adds exactly nothing
        # to the max, the row sum or P@V. The cut k/v sit in L1 while they are small.
        span = min(cap, int(kv_cache.get("span") or cap))
        keys, values = kv_cache["k"], kv_cache["v"]
        # The C++ scores and P@V read the live span of the cache IN PLACE (their reads stride over the
        # cache rows), so no cut of k / v is copied out each step.
        in_place = q_in_place or (
            span < cap and use_cpp_scores and use_cpp_pv and cpp_scores_dec.supports(q, keys, span)
        )
        # The prompt prefix every user shares may sit in ONE copy (`_seed_split`): only the C++ in-place scores and
        # P@V read it from there; any other read of the cache has it written into every user's rows first.
        shared_prefix = kv_cache.get("prefix_shared")
        if shared_prefix is not None and not (
            in_place
            and use_cpp_pv
            and cpp_scores_dec.prefix_ok(keys, shared_prefix.get("k"))
            and cpp_scores_dec.prefix_ok(values, shared_prefix.get("v"))
            and int(shared_prefix["k"].shape[-2]) <= span
        ):
            _materialize_prefix(kv_cache)
            shared_prefix = None
            keys, values = kv_cache["k"], kv_cache["v"]
        cut_kv = span < cap and not in_place
        if cut_kv:
            ends = [int(s) for s in keys.shape]
            ends[-2] = span
            cut = l1 if span <= 320 else None
            keys = ttnn.slice(keys, [0, 0, 0, 0], ends, memory_config=cut)
            values = ttnn.slice(values, [0, 0, 0, 0], ends, memory_config=cut)
        # structural: with q grouped in place and the context merge in C++, kv head h's `groups` real score rows
        # land at rows h * groups .. of ONE 32-row tile row a sample (cpp_scores_dec packed=True) -- the mask add,
        # max, exp and sum below touch n_kv-times fewer tiles; every row's values are the same (all per-row ops).
        packed = (
            q_in_place
            and use_cpp_pv
            and groups * n_kv_heads == n_heads == 32
            and cpp_ctx_merge.enabled()
            and cpp_scores_dec.packed_enabled()
        )
        if in_place:
            # the C++ decode scores, keys (and q, when grouped there) read in place
            scores = cpp_scores_dec.apply(
                q,
                keys,
                span,
                groups=groups if q_in_place else None,
                packed=packed,
                prefix=shared_prefix["k"] if shared_prefix is not None else None,
            )
        elif use_cpp_scores and cpp_scores_dec.supports(q, keys):
            scores = cpp_scores_dec.apply(q, keys)  # the C++ decode scores
        else:
            scores = _bmm(q, keys, transpose_b=True, memory_config=l1)
        ttnn.deallocate(q)
        # The cache tail beyond `position` is zeros, and a zero key scores ZERO -- which is a
        # perfectly ordinary logit, not a small one. It has to be masked explicitly.
        masked = ttnn.add(scores, _decode_mask(kv_cache, position, span), memory_config=l1)
        ttnn.deallocate(scores)
        # Normalise AFTER P@V: dividing the [B, n_kv, 32, head_dim] context by the row sums is the
        # same arithmetic as dividing the C/head_dim-times larger [B, n_kv, 32, C] weights first.
        e = ttnn.subtract(
            masked,
            ttnn.max(masked, dim=-1, keepdim=True, memory_config=l1),
            activations=[ttnn.UnaryOpType.EXP],
            memory_config=l1,
        )
        ttnn.deallocate(masked)
        if use_cpp_pv and cpp_pv_dec.supports(e, values, groups=groups if packed else None):
            pv = cpp_pv_dec.apply(  # the C++ decode P@V
                e,
                values,
                groups=groups if packed else None,
                prefix=shared_prefix["v"] if shared_prefix is not None else None,
            )
        else:
            if shared_prefix is not None:
                _materialize_prefix(kv_cache)
                shared_prefix = None
                values = kv_cache["v"]
            pv = _bmm(e, values, memory_config=l1)
        ssum = ttnn.sum(e, dim=-1, keepdim=True, memory_config=l1)
        if packed and not cpp_ctx_merge.supports(pv, ssum, groups):
            raise RuntimeError("packed decode scores need the C++ context merge")
        if groups * n_kv_heads == n_heads and cpp_ctx_merge.supports(pv, ssum, groups):
            # structural: the normalise and the head merge in ONE generic_op (tt/cpp_ctx_merge) -- each merged
            # tile's batch rows gathered from the P@V tiles and divided by their row sums; the divide, the
            # untilize, the row-major relayout and the tilize are gone. The same bits.
            merged = cpp_ctx_merge.apply(pv, ssum, groups)
            ttnn.deallocate(e)
            if cut_kv:
                ttnn.deallocate(keys)
                ttnn.deallocate(values)
        else:
            ctx = ttnn.divide(pv, ssum)
            ttnn.deallocate(e)
            if cut_kv:
                ttnn.deallocate(keys)
                ttnn.deallocate(values)
            # Relabelling the context to its `groups` real rows is a zero-cost tile view, so the untilize
            # drops the pad rows itself instead of writing all 32 and slicing them off.
            valid = ttnn.reshape(
                ctx,
                ttnn.Shape([batch, n_kv_heads, groups, head_dim]),
                ttnn.Shape([batch, n_kv_heads, q_rows, head_dim]),
            )
            merged = ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(valid, ttnn.ROW_MAJOR_LAYOUT), [1, 1, batch, n_heads * head_dim]),
                ttnn.TILE_LAYOUT,
            )
            ttnn.deallocate(ctx)
        out = _lin(
            ttnn.reshape(merged, [1, 1, batch, n_heads * head_dim]),
            wo,
            dtype=xn.dtype,
            compute_kernel_config=_COMPUTE,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # read once, by the residual add
        )
        ttnn.deallocate(merged)
        # Back in the caller's own shape, so the residual add downstream lines up whichever form
        # of the one-token stream came in.
        return ttnn.reshape(out, held[:-1] + [dim])

    def _prefill_attn(xn, position_embeddings, kv_cache, seq, trim=None):
        # The fused qkv's only reader is the head split right after: L1.
        qkv = _lin(xn, wqkv, dtype=_SDPA_DTYPE, compute_kernel_config=_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, v = _qkv_heads_rope(qkv, n_heads, n_kv_heads, position_embeddings, seq, head_dim, half)
        ttnn.deallocate(qkv)
        # The stack's last block reads only some rows (`trim`): their attention is all that runs.
        keep = None
        phase = kv_cache.get("prefix_phase") if kv_cache is not None else None
        if trim is not None and getattr(trim, "keeps_none", False) and phase == "stash":
            keep = "none"
        elif (
            trim is not None
            and getattr(trim, "every", None)
            and getattr(trim, "mask", None) is not None
            and phase == "extend"
            and kv_cache.get("prefix_compact") is not None
        ):
            keep = (trim.every, trim.mask)
        a, k, v = _prefill_sdpa(q, k, v, kv_cache, keep=keep)
        if kv_cache is not None and k is not None:
            _seed_cache(kv_cache, k, v)
        if a is None:
            return None
        if isinstance(keep, tuple):
            merged = ttnn.experimental.nlp_concat_heads(a, memory_config=ttnn.L1_MEMORY_CONFIG)
            out = _kept_rows_lin(
                merged,
                wo,
                int(xn.shape[0]) * int(xn.shape[1]) * int(xn.shape[2]),
                dtype=xn.dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(merged)
            return out
        merged = _merge_heads(a, wo, xn.dtype, _COMPUTE)
        out = _lin(
            merged,
            # The tall tail read a bfloat4_b copy of Wo here (2026-10-02, 0.19 ms). Dropped 2026-10-05: the
            # coarser weight shifted the first frames' codes and clipped demo row 15's opening words
            # ("A cartographer" -> "redrew the coastline"), unseen by the corpus-level gates.
            wo,
            dtype=xn.dtype,
            compute_kernel_config=_COMPUTE,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # read once, by the residual add
        )
        ttnn.deallocate(merged)
        return out

    def mistral_decoder_layer(
        hidden_states,
        position_embeddings=None,
        kv_cache=None,
        position=None,
        decode=False,
        trim=None,
        **kwargs,
    ):
        h, lead, seq, rank = _view4(hidden_states, dim)

        # A decode step's norm output (the qkv matmul's only input) is small: L1.
        xn = _rms_norm(
            h,
            g_in,
            eps_in,
            dtype=None if decode else ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if decode:
            attn_out = _decode_attn(xn, position_embeddings, kv_cache, position)
        else:
            attn_out = _prefill_attn(xn, position_embeddings, kv_cache, seq, trim=trim)
        ttnn.deallocate(xn)
        if attn_out is None:
            return None  # the last block of a chain read only for its k/v
        # The float32 residual stream lives in L1 between blocks -- the 160-row prefix's, the 640-row
        # tail's (~72 KB a core) and a decode step's: both adds write it there and both norms read
        # it from there instead of re-streaming it from DRAM.
        resid = ttnn.L1_MEMORY_CONFIG
        if trim is not None and not decode and int(attn_out.shape[-2]) != int(h.shape[-2]):
            # The attention ran on the kept rows only: the residual is trimmed to them first.
            kept = trim(h)
            h = ttnn.add(kept, attn_out, memory_config=resid)
            ttnn.deallocate(kept)
            ttnn.deallocate(attn_out)
            trim, decode, kept_rows = None, True, True
        else:
            h = ttnn.add(h, attn_out, memory_config=resid)
            ttnn.deallocate(attn_out)
            kept_rows = False
        if trim is not None:
            # The stack reads only some of this block's rows (its last block): `trim` keeps those as
            # `[1, 1, rows, dim]` (or None when none are read), and the FFN runs on them as a decode
            # step's would.
            kept = trim(h)
            ttnn.deallocate(h)
            if kept is None:
                return None
            h, decode = kept, True

        # Prefill's norm output is read only by the fused SwiGLU, once per N block: it lands in L1,
        # bfloat8_b -- the LoFi matmul reads no more of its mantissa, and half the bytes cross the grid.
        hn = _rms_norm(
            h,
            g_post,
            eps_post,
            dtype=None if decode else ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if decode:
            l1 = ttnn.L1_MEMORY_CONFIG  # gate / up / their product are each read once, right after
            gated = ttnn.multiply(
                _lin(hn, w_gate, dtype=ttnn.bfloat16, compute_kernel_config=_COMPUTE, memory_config=l1),
                _lin(hn, w_up, dtype=ttnn.bfloat16, compute_kernel_config=_COMPUTE, memory_config=l1),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=l1,
            )
        else:
            # The short prefix's SwiGLU writes straight into the down projection's K-block shards.
            rows = 1
            for d in list(hn.shape)[:-1]:
                rows *= int(d)
            split = None
            if 128 <= rows < 256 and rows % 32 == 0 and len(list(hn.shape)) == 4:
                split = _down_k_split(
                    h.device(),
                    ttnn.bfloat16,
                    rows,
                    int(w_down.shape[-2]),
                    w_down,
                    ttnn.bfloat16,
                    bool(getattr(_COMPUTE, "fp32_dest_acc_en", False)),
                )
            gated = _fused_swiglu(hn, w_gu, w_ttl, memory_config=split[1] if split else None, w_cpp=w_gu_cpp)
        ttnn.deallocate(hn)
        # Prefill hands the down projection over in bf16, as the composed kinds' mlp and every wo
        # already do; the residual it is added into stays float32.
        down_dtype = h.dtype if decode else ttnn.bfloat16
        if decode and down_dtype == ttnn.float32 and cpp_down_dec.serves(gated, w_down_cpp):
            down = cpp_down_dec.apply(gated, w_down_cpp)  # the C++ decode down projection, every layer
        else:
            down = _down_short(
                gated, w_down, dtype=down_dtype, compute_kernel_config=_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        h = ttnn.add(h, down, memory_config=resid)
        ttnn.deallocate(gated)

        return h if trim is not None or kept_rows else _restore(h, lead, seq, rank, dim)

    return mistral_decoder_layer
