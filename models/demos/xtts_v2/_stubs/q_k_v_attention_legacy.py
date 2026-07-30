# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `q_k_v_attention_legacy` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_encoder.attn.0.attention`` — a
``QKVAttentionLegacy`` (n_heads=16). It takes a PACKED qkv tensor
``[N, H*3*C, T]`` (no learnable projections here — q/k/v are already produced
upstream) and returns ``[N, H*C, T]``::

    q, k, v = qkv.reshape(N*H, 3C, T).split(C, dim=1)
    scale   = 1 / sqrt(sqrt(C))
    weight  = einsum('bct,bcs->bts', q*scale, k*scale)   # [N*H, T, T]
    weight  = softmax(weight, dim=-1)
    a       = einsum('bts,bcs->bct', weight, v)          # [N*H, C, T]
    return a.reshape(N, H*C, T)

(mask and rel_pos are None for this capture.)

TP=8 scheme (math unchanged)
----------------------------
This module has NO weights — the packed qkv is the input, which the shard harness
replicates across the mesh. With nothing to shard, every chip computes the
identical attention; the harness gathers (concat over the mesh axis) and slices
back the single replica, so the gathered output equals the single-device golden.
float32 + fp32 accumulation hold PCC through the two batched matmuls + softmax.
"""

from __future__ import annotations

import math

import ttnn


def build(device, torch_module):
    n_heads = int(torch_module.n_heads)

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    def forward(qkv, mask=None, rel_pos=None, **_):
        if isinstance(qkv, ttnn.Tensor) and qkv.get_dtype() != ttnn.float32:
            qkv = ttnn.typecast(qkv, ttnn.float32)
        bs = int(qkv.shape[0])
        width = int(qkv.shape[1])
        T = int(qkv.shape[2])
        ch = width // (3 * n_heads)
        scale = 1.0 / math.sqrt(math.sqrt(ch))

        # [N, H*3C, T] -> [N*H, 3C, T], then split into q/k/v each [N*H, C, T].
        x = ttnn.reshape(qkv, [bs * n_heads, 3 * ch, T])
        q = ttnn.slice(x, [0, 0, 0], [bs * n_heads, ch, T])
        k = ttnn.slice(x, [0, ch, 0], [bs * n_heads, 2 * ch, T])
        v = ttnn.slice(x, [0, 2 * ch, 0], [bs * n_heads, 3 * ch, T])

        qs = ttnn.multiply(q, scale)
        ks = ttnn.multiply(k, scale)
        # weight[b,t,s] = sum_c qs[b,c,t]*ks[b,c,s]  ->  (qs^T) @ ks
        weight = ttnn.matmul(ttnn.transpose(qs, 1, 2), ks, compute_kernel_config=kcfg)  # [N*H, T, T]
        weight = ttnn.softmax(weight, dim=-1)
        # a[b,c,t] = sum_s weight[b,t,s]*v[b,c,s]  ->  v @ (weight^T)
        a = ttnn.matmul(v, ttnn.transpose(weight, 1, 2), compute_kernel_config=kcfg)  # [N*H, C, T]
        return ttnn.reshape(a, [bs, n_heads * ch, T])       # [N, H*C, T]

    return forward
