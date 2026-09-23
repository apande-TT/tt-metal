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
Attention state: K/V caches over `KV_CAPACITY` positions plus a position
offset `rel = arange - pos` that masks the future and selects the write slot.
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
    persist(state, "ssm", ttnn.matmul(ttnn.transpose(Bw, -2, -1), x_disc, compute_kernel_config=ckc))  # (B,H,N,P)


def mamba_conv_step(state, hbc_t, taps, bias, K):
    """Causal depthwise conv for one token, then advance the window.
    taps[s] multiplies x[t-s]. hbc_t (1,B,conv_dim) -> silu(conv), same shape."""
    acc = ttnn.multiply(hbc_t, taps[0])
    for s in range(1, K):
        acc = ttnn.add(acc, ttnn.multiply(state[f"prev{s}"], taps[s]))
    if bias is not None:
        acc = ttnn.add(acc, bias)
    for s in range(K - 1, 1, -1):
        ttnn.copy(state[f"prev{s - 1}"], state[f"prev{s}"])
    ttnn.copy(hbc_t, state["prev1"])
    return ttnn.silu(acc)


def mamba_ssm_step(state, x_h, B_h, C_h, dt_h, A, D, ckc):
    """One SSD recurrence step. x_h (B,H,1,P), B_h/C_h (B,H,1,N), dt_h (B,H,1,1).
    Returns y (B,H,1,P)."""
    decay = ttnn.exp(ttnn.multiply(dt_h, A))  # (B,H,1,1)
    xdt = ttnn.multiply(x_h, dt_h)
    upd = ttnn.matmul(ttnn.transpose(B_h, -2, -1), xdt, compute_kernel_config=ckc)  # (B,H,N,P)
    S_new = ttnn.add(ttnn.multiply(state["ssm"], decay), upd)
    ttnn.deallocate(upd)
    y = ttnn.matmul(C_h, S_new, compute_kernel_config=ckc)  # (B,H,1,P)
    ttnn.copy(S_new, state["ssm"])
    ttnn.deallocate(S_new)
    return ttnn.add(y, ttnn.multiply(x_h, D))


# --------------------------------------------------------------------------- #
#  attention
# --------------------------------------------------------------------------- #
def attn_fill(state, device, Kh, Vh):
    """Seed the K/V caches from the prompt's (B,H,S,D) keys/values and set the
    write position to S."""
    B, H, S, D = [int(v) for v in Kh.shape]
    pad = KV_CAPACITY - S
    assert pad >= 0, f"prompt of {S} exceeds KV_CAPACITY={KV_CAPACITY}"
    zeros = upload(device, torch.zeros(B, H, pad, D))
    persist(state, "k", ttnn.concat([Kh, zeros], dim=2))
    persist(state, "v", ttnn.concat([Vh, zeros], dim=2))
    ar = torch.arange(KV_CAPACITY, dtype=torch.float32) - S
    persist(state, "rel_row", upload(device, ar.reshape(1, 1, 1, KV_CAPACITY)))
    persist(state, "rel_col", upload(device, ar.reshape(1, 1, KV_CAPACITY, 1)))


def attn_step(state, q_h, k_h, v_h, scaling, ckc):
    """One cached attention step. q_h/k_h/v_h (B,H,1,D). Returns (B,H,1,D)."""
    slot = ttnn.typecast(ttnn.eqz(state["rel_col"]), ttnn.float32)  # (1,1,Cap,1) one-hot at pos
    keep = ttnn.rsub(slot, 1.0)
    k_new = ttnn.add(ttnn.multiply(state["k"], keep), ttnn.multiply(slot, k_h))
    v_new = ttnn.add(ttnn.multiply(state["v"], keep), ttnn.multiply(slot, v_h))
    ttnn.copy(k_new, state["k"])
    ttnn.copy(v_new, state["v"])
    ttnn.deallocate(k_new)
    ttnn.deallocate(v_new)

    future = ttnn.multiply(ttnn.typecast(ttnn.gtz(state["rel_row"]), ttnn.float32), -1e9)  # (1,1,1,Cap)
    scores = ttnn.matmul(q_h, ttnn.transpose(state["k"], -2, -1), compute_kernel_config=ckc)  # (B,H,1,Cap)
    scores = ttnn.add(ttnn.multiply(scores, scaling), future)
    probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=ckc, numeric_stable=True)
    out = ttnn.matmul(probs, state["v"], compute_kernel_config=ckc)  # (B,H,1,D)

    ttnn.copy(ttnn.subtract(state["rel_row"], 1.0), state["rel_row"])
    ttnn.copy(ttnn.subtract(state["rel_col"], 1.0), state["rel_col"])
    return out
