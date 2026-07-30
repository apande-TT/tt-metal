# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `s_e_layer` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder.layer1.0.se`` — a
``SELayer`` (Squeeze-and-Excitation), here for 32 channels with reduction to 4.
``forward(x)`` on ``[N, C, H, W]``::

    y = avg_pool(x).view(N, C)          # global average over (H, W)
    y = fc(y).view(N, C, 1, 1)          # Linear(C, C/r) -> relu -> Linear(C/r, C) -> sigmoid
    return x * y                        # per-channel gate

Native strategy
---------------
Pure ttnn: ``ttnn.mean`` over the spatial axes for the squeeze, two small
matmuls (relu / sigmoid) for the excitation MLP, then a broadcast multiply.
float32 + fp32 accumulation holds PCC.

TP=8 scheme
-----------
The only weights are the two tiny SE projections (32<->4); neither is a large
model-dim matmul that benefits from a column/row split. Per the TP principles
(keep small projections REPLICATED), both are staged REPLICATED across the mesh
via ``ReplicateTensorToMesh``, so each chip computes the identical gate and the
gathered output equals the single-device golden bit-for-bit.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    se_layer = torch_module

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    def _rep(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    fc = se_layer.fc
    W1 = _rep(fc[0].weight.detach().float().t())               # [C, C/r]
    b1 = _rep(fc[0].bias.detach().float().reshape(1, -1))
    W2 = _rep(fc[2].weight.detach().float().t())               # [C/r, C]
    b2 = _rep(fc[2].bias.detach().float().reshape(1, -1))

    def forward(x, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        C = int(x.shape[1])
        y = ttnn.mean(x, dim=[2, 3])                           # squeeze -> [1, C]
        y = ttnn.reshape(y, [1, C])
        y = ttnn.add(ttnn.matmul(y, W1, compute_kernel_config=kcfg), b1)
        y = ttnn.relu(y)
        y = ttnn.add(ttnn.matmul(y, W2, compute_kernel_config=kcfg), b2)
        y = ttnn.sigmoid(y)                                    # excitation -> [1, C]
        y = ttnn.reshape(y, [1, C, 1, 1])
        return ttnn.multiply(x, y)                             # per-channel gate

    return forward
