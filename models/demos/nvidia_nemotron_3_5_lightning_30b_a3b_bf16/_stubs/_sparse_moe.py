# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sparse (routed-pairs-only) MoE mixture for prefill.

Each local expert takes its top-C tokens by routing weight (unrouted slots
carry weight 0 and contribute nothing), gathers their rows, runs its own
up/relu2/down, and a one-hot matmul adds every row back to its token. The
dense form evaluates all Eloc experts on all T tokens; this does Eloc*C rows,
C ~ 4x the mean expert load.
"""
from __future__ import annotations

import math

import torch

import ttnn

TILE = 32


def capacity(num_tokens, top_k, num_experts):
    """Tokens per local expert: 4x the mean load, tile-aligned, capped at T."""
    mean = num_tokens * top_k / num_experts
    return min(num_tokens, TILE * max(1, math.ceil(4 * mean / TILE)))


def _tile_bytes(dtype):
    return {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(dtype, 4096)


def bmm_config(device, M, K, N, in0_dtype, in1_dtype, cb_budget=512 * 1024):
    """Hand-shaped 2-D multicast config for the batched per-expert matmuls.

    Left to itself ttnn picks in0_block_w=1 here, so every K tile is its own
    multicast round trip. Take the widest K block whose double-buffered in0/in1
    CBs fit cb_budget, and the largest subblock the fp32 dest (4 tiles) holds.
    Rows of the grid cover M, columns cover N (same core footprint ttnn picks).
    """
    g = device.compute_with_storage_grid_size()
    mt, kt, nt = M // TILE, K // TILE, N // TILE
    pm, pn = math.ceil(mt / g.y), math.ceil(nt / g.x)
    b0, b1 = _tile_bytes(in0_dtype), _tile_bytes(in1_dtype)
    kw = max(d for d in range(1, kt + 1) if kt % d == 0 and 2 * d * (pm * b0 + pn * b1) <= cb_budget)
    sub = max(
        ((h, w) for h in range(1, pm + 1) for w in range(1, pn + 1) if pm % h == 0 and pn % w == 0 and h * w <= 4),
        key=lambda s: (s[0] * s[1], s[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=g,
        in0_block_w=kw,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def routed_mix(device, hs, W, up_b, down_cat, C, ckc, arange_cache):
    """hs (T, H); W (T, Eloc) routing weights for this chip's experts;
    up_b (Eloc, H, I); down_cat (Eloc*I, H). Returns this chip's partial
    mixture (T, H) fp32."""
    T, H = int(hs.shape[-2]), int(hs.shape[-1])
    Eloc = int(W.shape[-1])
    I = int(up_b.shape[-1])
    # per-expert top-C tokens: values = routing weights, indices = token ids
    vals, idx = ttnn.topk(ttnn.typecast(ttnn.transpose(W, -2, -1), ttnn.bfloat16), C, dim=-1)  # (Eloc, C)
    idx_rm = ttnn.reshape(ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, Eloc * C])
    table = ttnn.to_layout(ttnn.typecast(hs, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)  # (T, H)
    xe = ttnn.reshape(ttnn.embedding(idx_rm, table, layout=ttnn.TILE_LAYOUT), [Eloc, C, H])
    ttnn.deallocate(table)
    act = ttnn.matmul(
        xe,
        up_b,
        compute_kernel_config=ckc,
        dtype=ttnn.bfloat8_b,
        program_config=bmm_config(device, C, H, I, xe.dtype, up_b.dtype),
    )  # (Eloc, C, I)
    ttnn.deallocate(xe)
    w3 = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT), [Eloc, C, 1]), ttnn.TILE_LAYOUT)
    act = ttnn.multiply(
        ttnn.relu(act), w3, dtype=ttnn.bfloat8_b, input_tensor_a_activations=[ttnn.UnaryOpType.SQUARE]
    )  # relu2 * routing weight
    ye = ttnn.matmul(
        act,
        ttnn.reshape(down_cat, [Eloc, I, H]),
        compute_kernel_config=ckc,
        dtype=ttnn.bfloat8_b,
        program_config=bmm_config(device, C, I, H, act.dtype, down_cat.dtype),
    )
    ttnn.deallocate(act)
    # combine: out[t] = sum over (e, c) with idx[e, c] == t of ye[e, c]
    ar = arange_cache.get(T)
    if ar is None:
        kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
        ar = arange_cache[T] = ttnn.from_torch(
            torch.arange(T, dtype=torch.float32).reshape(T, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            **kw,
        )
    idx_f = ttnn.reshape(ttnn.typecast(idx, ttnn.float32), [1, Eloc * C])
    onehot = ttnn.eq(ar, idx_f, dtype=ttnn.bfloat8_b)  # (T, Eloc*C); 0/1 is exact in bf8_b
    out = ttnn.matmul(onehot, ttnn.reshape(ye, [Eloc * C, H]), compute_kernel_config=ckc, dtype=ttnn.float32)
    ttnn.deallocate(onehot)
    ttnn.deallocate(ye)
    return out
