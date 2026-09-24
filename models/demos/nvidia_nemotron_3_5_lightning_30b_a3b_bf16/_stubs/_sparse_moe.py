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
    act = ttnn.matmul(xe, up_b, compute_kernel_config=ckc, dtype=ttnn.bfloat8_b)  # (Eloc, C, I)
    ttnn.deallocate(xe)
    w3 = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT), [Eloc, C, 1]), ttnn.TILE_LAYOUT)
    act = ttnn.multiply(
        ttnn.relu(act), w3, dtype=ttnn.bfloat8_b, input_tensor_a_activations=[ttnn.UnaryOpType.SQUARE]
    )  # relu2 * routing weight
    ye = ttnn.matmul(act, ttnn.reshape(down_cat, [Eloc, I, H]), compute_kernel_config=ckc, dtype=ttnn.bfloat8_b)
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
