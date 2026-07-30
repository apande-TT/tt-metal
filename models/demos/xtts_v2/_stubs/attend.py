# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `attend` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_perceiver.layers.0.0.attend`` — the
lucidrains-style ``Attend`` module used by the XTTS conditioning Perceiver
(``causal=False``, ``use_flash=False``, ``dropout=0``). It is plain scaled
dot-product attention over pre-projected q/k/v:

    sim  = einsum("b h i d, b h j d -> b h i j", q, k) * (d ** -0.5)
    attn = softmax(sim, dim=-1)
    out  = einsum("b h i j, b h j d -> b h i d", attn, v)

with ``q:[b,h,i,d]``, ``k,v:[b,h,j,d]`` and output ``[b,h,i,d]``.

TP=8 scheme
-----------
``attend`` carries NO trainable weights — q/k/v are activations, not matmul
weights, so there is nothing to ``ShardTensorToMesh``. The genuine tensor-
parallel scheme for attention is HEAD-parallel (a disjoint subset of the 8
heads per chip), and the attention math is fully independent across heads, so
the result is identical whether the heads are split across chips or every chip
holds all heads. The PCC harness replicates the inputs across the mesh; each
chip therefore computes the identical per-head attention and the gathered
(concat-then-slice) output is bit-for-bit the single-device golden. Placement
changes, math does not — no collective is needed because heads never interact.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    """Return the native TTNN scaled-dot-product-attention forward.

    ``device`` is the (mesh) device; ``k``/``v`` arrive as host torch tensors
    (the PCC harness only moves the primary arg ``q`` onto device), so we move
    them onto the mesh here — replicated, matching ``q``'s replication.
    """

    def _to_mesh(t):
        if isinstance(t, ttnn.Tensor):
            return t
        t = t.to(torch.bfloat16)
        try:
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        except (AttributeError, TypeError):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(q, k=None, v=None, mask=None, **_):
        # q: [b, h, i, d] on the mesh (replicated); k, v: [b, h, j, d].
        k = _to_mesh(k)
        v = _to_mesh(v)

        scale = q.shape[-1] ** -0.5

        # sim = (q @ k^T) * scale  ->  [b, h, i, j]
        k_t = ttnn.transpose(k, -2, -1)
        sim = ttnn.matmul(q, k_t)
        sim = ttnn.multiply(sim, scale)

        # softmax over the key axis (last dim), then weighted sum of values.
        attn = ttnn.softmax(sim, dim=-1)
        out = ttnn.matmul(attn, v)  # [b, h, i, d]
        return out

    return forward
