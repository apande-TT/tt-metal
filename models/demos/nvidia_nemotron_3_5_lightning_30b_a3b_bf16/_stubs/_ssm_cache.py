# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent decode state for the Mamba2 mixers and the attention layer.

The stubs are full-sequence bodies. With a cache attached they also run in two
extra modes, selected by the pipeline through `stub._cache_mode`:

  "fill"   : the normal full-prompt forward, which additionally writes the
             state left after the last prompt position.
  "decode" : a single-token step that reads that state and advances it. Its
             activations arrive as (1, B, .) -- the B tokens share one tile
             row instead of each padding a (B, 1, .) row to 32 -- and are
             split to per-sample (B, H, 1, d) only for the recurrent math.

State tensors are allocated by the first fill and then only ever updated in
place (ttnn.copy), so a captured decode trace keeps pointing at live buffers.

Mamba2 state per layer: the last K-1 PRE-conv rows (the depthwise conv's
window) and the SSD state S[h] (N x P), updated as
    S <- exp(dt*A) * S + B^T (dt*x),     y = C S + D x.
Attention state: bf16 K/V caches over `KV_CAPACITY` positions, written with
ttnn's in-place cache ops (fill_cache / paged_update_cache), an int32 write
position, and `rel = arange - pos`, which masks the future.
"""
from __future__ import annotations

import torch

import ttnn

# Cached positions per attention layer: prompt plus generated tokens.
KV_CAPACITY = 128


def persist(state, name, new):
    """Keep `new` as state[name]: the first call adopts it, later calls copy
    into the existing buffer so trace-captured addresses stay valid."""
    old = state.get(name)
    if old is None:
        state[name] = new
    else:
        ttnn.copy(new, old)
        ttnn.deallocate(new)


def upload(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, **kw)


def heads_matmul(a, b, ckc, cb_budget=256 * 1024, **kw):
    """Per-head batched matmul (..., M, K) @ (..., K, N) with the batch spread
    over the grid: one whole (M, N) output block per core (BMM "reuse", no
    multicast). ttnn's own choice for these parallelises only the Mt x Nt
    output tiles of ONE head -- a handful of cores looping over B*H heads."""
    g = a.device().compute_with_storage_grid_size()
    mt, kt, nt = [-(-int(v) // 32) for v in (a.padded_shape[-2], a.padded_shape[-1], b.padded_shape[-1])]
    tb = max(4096 if t.dtype == ttnn.float32 else 2048 for t in (a, b))
    kw_blk = max(d for d in range(1, kt + 1) if kt % d == 0 and (d == 1 or 2 * d * (mt + nt) * tb <= cb_budget))
    sub = max(
        ((h, w) for h in range(1, mt + 1) for w in range(1, nt + 1) if mt % h == 0 and nt % w == 0 and h * w <= 4),
        key=lambda s: (s[0] * s[1], s[1]),
    )
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=g,
        in0_block_w=kw_blk,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        per_core_M=mt,
        per_core_N=nt,
    )
    return ttnn.matmul(a, b, compute_kernel_config=ckc, program_config=pc, **kw)


def to_heads(t, B, S, n, d):
    """(B, S, n*d) tile -> (B, n, S, d) tile."""
    rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    rm = ttnn.reshape(rm, [B, S, n, d])
    rm = ttnn.permute(rm, (0, 2, 1, 3))
    return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)


def from_heads(t, B, S, n, d):
    """(B, n, S, d) tile -> (B, S, n*d) tile."""
    rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    rm = ttnn.permute(rm, (0, 2, 1, 3))
    rm = ttnn.reshape(rm, [B, S, n * d])
    return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)


# --------------------------------------------------------------------------- #
#  Mamba2
# --------------------------------------------------------------------------- #
def mamba_fill(state, device, hbc_pre, cumA, B_h, x_disc, K, ckc):
    """Write the state after the last prompt position.

    hbc_pre (B,S,conv_dim) pre-conv channels; cumA (B,H,S,1) inclusive cumsum of
    A*dt; B_h (B,H,S,N); x_disc (B,H,S,P) = x*dt.
    """
    B, S, C = [int(v) for v in hbc_pre.shape]
    for s in range(1, K):
        if S - s >= 0:
            row = ttnn.slice(hbc_pre, [0, S - s, 0], [B, S - s + 1, C])
            row = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT), [1, B, C]), ttnn.TILE_LAYOUT)
        else:
            row = upload(device, torch.zeros(1, B, C))
        persist(state, f"prev{s}", row)

    H = int(cumA.shape[1])
    last = ttnn.slice(cumA, [0, 0, S - 1, 0], [B, H, S, 1])  # (B,H,1,1)
    w = ttnn.exp(ttnn.subtract(last, cumA))  # decay from s to the end, (B,H,S,1); s<=S-1 always
    Bw = ttnn.multiply(B_h, w)  # (B,H,S,N)
    persist(state, "ssm", heads_matmul(ttnn.transpose(Bw, -2, -1), x_disc, ckc))  # (B,H,N,P)


def mamba_conv_step(state, hbc_t, taps, bias, K):
    """Causal depthwise conv for one token, then advance the window.
    taps[s] multiplies x[t-s]. hbc_t (1,B,conv_dim) -> silu(conv), same shape."""
    acc = ttnn.multiply(hbc_t, taps[0])
    for s in range(1, K):
        acc = ttnn.addcmul(acc, state[f"prev{s}"], taps[s])  # acc + prev_s * tap_s in one op
    if bias is not None:
        acc = ttnn.add(acc, bias)
    for s in range(K - 1, 1, -1):
        ttnn.copy(state[f"prev{s - 1}"], state[f"prev{s}"])
    ttnn.copy(hbc_t, state["prev1"])
    return ttnn.silu(acc)


def group_rms(device, y, w, gs, eps, cache, ckc):
    """Grouped RMSNorm of one tile row of tokens, (1, M<=32, I), over consecutive
    gs-wide groups, times w -- as two tiny matmuls against group-indicator
    constants instead of a ROW_MAJOR reshape round trip to (M*ng, gs)."""
    I = int(y.shape[-1])
    ng = I // gs
    key = ("grp", I, gs)
    if key not in cache:
        g = torch.zeros(I, ng)
        for j in range(ng):
            g[j * gs : (j + 1) * gs, j] = 1.0
        cache[key] = (upload(device, g / gs), upload(device, g.t().contiguous()))
    g_mean, g_expand = cache[key]
    ms = ttnn.matmul(ttnn.multiply(y, y), g_mean, compute_kernel_config=ckc)  # (1, M, ng) mean of squares
    r = ttnn.rsqrt(ttnn.add(ms, eps))
    return ttnn.multiply(ttnn.multiply(y, ttnn.matmul(r, g_expand, compute_kernel_config=ckc)), w)


def softplus(x):
    """log(1 + exp(x)) in two ops. Decode dt pre-activations sit far below
    fp32 exp overflow (~88), so the stable relu + log1p(exp(-|x|)) form's
    extra four ops buy nothing here."""
    return ttnn.log1p(ttnn.exp(x))


def mamba_ssm_step(state, x_h, B_h, C_h, dt_h, A, D, ckc):
    """One SSD recurrence step. x_h (B,H,1,P), B_h/C_h (B,H,1,N), dt_h (B,H,1,1).
    Returns y (B,H,1,P).

    The (B,H,N,P) fp32 state is the traffic that matters, so it is touched as
    few times as possible: one pass for the decay, and one addcmul that forms
    the B^T (x dt) outer product by broadcasting and writes the sum straight
    back into the state buffer."""
    decay = ttnn.exp(ttnn.multiply(dt_h, A))  # (B,H,1,1)
    xdt = ttnn.multiply(x_h, dt_h)  # (B,H,1,P)
    S_dec = ttnn.multiply(state["ssm"], decay)
    ttnn.addcmul(S_dec, ttnn.transpose(B_h, -2, -1), xdt, output_tensor=state["ssm"])  # (B,H,N,1)*(B,H,1,P)
    ttnn.deallocate(S_dec)
    # y = C S as broadcast-multiply + reduce over N: a 1024-way batch of
    # (1 x N) @ (N x P) matmuls pads M to a full tile and runs far off bandwidth
    y = ttnn.sum(ttnn.multiply(state["ssm"], ttnn.transpose(C_h, -2, -1)), dim=-2, keepdim=True)  # (B,H,1,P)
    return ttnn.add(y, ttnn.multiply(x_h, D))


# --------------------------------------------------------------------------- #
#  attention
# --------------------------------------------------------------------------- #
def attn_fill(state, device, Kh, Vh):
    """Seed the bf16 K/V caches (B,H,KV_CAPACITY,D) from the prompt's (B,H,S,D)
    keys/values with ttnn.fill_cache, and set the write position to S."""
    B, H, S, D = [int(v) for v in Kh.shape]
    assert S <= KV_CAPACITY, f"prompt of {S} exceeds KV_CAPACITY={KV_CAPACITY}"
    for name, src in (("k", Kh), ("v", Vh)):
        if name not in state:
            state[name] = upload(device, torch.zeros(B, H, KV_CAPACITY, D), dtype=ttnn.bfloat16)
        src16 = ttnn.typecast(src, ttnn.bfloat16)
        for b in range(B):
            ttnn.fill_cache(state[name], ttnn.slice(src16, [b, 0, 0, 0], [b + 1, H, S, D]), batch_idx=b)
        ttnn.deallocate(src16)
    persist(state, "pos", upload(device, torch.full((B,), S, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))
    ar = torch.arange(KV_CAPACITY, dtype=torch.float32) - S
    persist(state, "rel_row", upload(device, ar.reshape(1, 1, 1, KV_CAPACITY)))


def _decode_rows(device, t_h):
    """(B,H,1,D) heads -> (1,B,H,D) bf16 height-sharded one sample per core,
    the input layout paged_update_cache expects."""
    B, H, _, D = [int(v) for v in t_h.shape]
    rows = ttnn.typecast(ttnn.permute(t_h, (2, 0, 1, 3)), ttnn.bfloat16)  # (1,B,H,D)
    mem = ttnn.create_sharded_memory_config(
        shape=(-(-H // 32) * 32, D),
        core_grid=ttnn.num_cores_to_corerangeset(B, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(rows, mem)


def attn_step(state, device, q_h, k_h, v_h, scaling, ckc):
    """One cached attention step. q_h/k_h/v_h (B,H,1,D). Writes this token's K/V
    in place at `pos` (paged_update_cache), attends over positions <= pos, then
    advances pos. Returns (B,H,1,D)."""
    for name, t in (("k", k_h), ("v", v_h)):
        rows = _decode_rows(device, t)
        ttnn.experimental.paged_update_cache(state[name], rows, update_idxs_tensor=state["pos"])
        ttnn.deallocate(rows)

    # q K^T and P V as broadcast multiply + reduce (one dtype per product: the
    # bf16 cache): as matmuls they are a B*H-way batch of single-row products
    future = ttnn.multiply(ttnn.typecast(ttnn.gtz(state["rel_row"]), ttnn.float32), -1e9)  # (1,1,1,Cap)
    qk = ttnn.multiply(state["k"], ttnn.typecast(q_h, ttnn.bfloat16))  # (B,H,Cap,D)
    scores = ttnn.typecast(ttnn.sum(qk, dim=-1, keepdim=True), ttnn.float32)  # (B,H,Cap,1)
    scores = ttnn.add(ttnn.multiply(ttnn.transpose(scores, -2, -1), scaling), future)  # (B,H,1,Cap)
    probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=ckc, numeric_stable=True)
    probs = ttnn.typecast(ttnn.transpose(probs, -2, -1), ttnn.bfloat16)  # (B,H,Cap,1)
    out = ttnn.typecast(ttnn.sum(ttnn.multiply(state["v"], probs), dim=-2, keepdim=True), ttnn.float32)  # (B,H,1,D)

    ttnn.plus_one(state["pos"])
    ttnn.copy(ttnn.subtract(state["rel_row"], 1.0), state["rel_row"])
    return out
