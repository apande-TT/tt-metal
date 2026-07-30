# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `group_norm32` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_encoder.attn.0.norm`` — a tortoise-style
``GroupNorm32`` (num_groups=32, num_channels=1024, eps=1e-5, affine). Input is
channel-first ``x: [b, C, T]``; it normalizes over each group's (channels/group,
time) block and applies a per-channel affine:

    xr = x.reshape(b, G, C//G, T)
    xn = (xr - mean) * rsqrt(var + eps)        # mean/var over the last two axes
    y  = xn.reshape(b, C, T) * gamma + beta    # gamma/beta broadcast over T

This mirrors the GroupNorm already used inside the graduated
``conditioning_encoder`` / ``attention_block`` stubs.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    G = int(torch_module.num_groups)
    C = int(torch_module.num_channels)
    eps = float(torch_module.eps)

    def _rep(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    gamma = _rep(torch_module.weight.detach().reshape(1, C, 1))
    beta = _rep(torch_module.bias.detach().reshape(1, C, 1))

    def forward(x, *_, **__):
        T = int(x.shape[-1])
        xr = ttnn.reshape(x, [1, G, C // G, T])
        m = ttnn.mean(ttnn.mean(xr, dim=3, keepdim=True), dim=2, keepdim=True)
        xc = ttnn.subtract(xr, m)
        var = ttnn.mean(ttnn.mean(ttnn.multiply(xc, xc), dim=3, keepdim=True), dim=2, keepdim=True)
        xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
        xn = ttnn.reshape(xn, [1, C, T])
        return ttnn.add(ttnn.multiply(xn, gamma), beta)

    return forward


def group_norm32(device, torch_module=None):
    return build(device, torch_module)
