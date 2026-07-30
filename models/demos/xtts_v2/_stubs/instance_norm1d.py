# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `instance_norm1d` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder.instancenorm`` — a
``torch.nn.InstanceNorm1d(num_features=64, affine=False, eps=1e-5,
track_running_stats=False)``.

InstanceNorm1d normalizes each (sample, channel) slice independently across the
time axis:

    mean = x.mean(dim=time, keepdim=True)                 # [N, C, 1]
    var  = ((x - mean) ** 2).mean(dim=time, keepdim=True) # population variance
    y    = (x - mean) * rsqrt(var + eps)
    if affine: y = y * weight[C] + bias[C]

Native strategy
---------------
Straight elementwise ttnn: two reductions over the last (time) dim for mean and
variance, then a broadcasted normalize. No weights to split; affine params (when
present) are per-channel and stay replicated. Compute runs in float32 with fp32
accumulation so the reductions hold PCC. This module has affine=False, so the
`weight`/`bias` branch is inert here but kept for generality.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    m = torch_module
    eps = float(getattr(m, "eps", 1e-5))
    affine = bool(getattr(m, "affine", False))

    def _rep(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    weight = None
    bias = None
    if affine and getattr(m, "weight", None) is not None:
        weight = _rep(m.weight.detach().reshape(1, -1, 1))
    if affine and getattr(m, "bias", None) is not None:
        bias = _rep(m.bias.detach().reshape(1, -1, 1))

    def forward(x, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        # Normalize each (sample, channel) across the last (time) axis.
        mean = ttnn.mean(x, dim=-1, keepdim=True)               # [N, C, 1]
        xc = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(var, eps))
        y = ttnn.multiply(xc, inv)
        if weight is not None:
            y = ttnn.multiply(y, weight)
        if bias is not None:
            y = ttnn.add(y, bias)
        return y

    return forward
