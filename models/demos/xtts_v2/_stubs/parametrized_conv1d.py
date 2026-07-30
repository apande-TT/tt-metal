# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrized_conv1d` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.waveform_decoder.resblocks.0.convs1.0`` — a
weight-normed ``Conv1d`` (in=256, out=256, kernel=3, stride=1, pad=1, dil=1,
bias=True). ``forward(x)`` is a plain 1D convolution; reading ``conv.weight``
materializes the weight-norm parametrization, so no special handling is needed
beyond fetching the effective weight.

Native strategy
---------------
Realize the conv as im2col + matmul: reflect-free zero-pad the time axis, gather
the ``k`` dilated taps by slicing, concat along channels, and matmul the reshaped
weight ``[k*Cin, Cout]``. Weight is REPLICATED across the mesh (channels are not
TP-divisible; replication gathers bit-for-bit to the single-device golden).
float32 + fp32 accumulation hold PCC.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    conv = torch_module

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

    W = conv.weight.detach()                                   # [Cout, Cin, k]
    b = conv.bias.detach() if conv.bias is not None else None
    k = int(conv.kernel_size[0])
    dil = int(conv.dilation[0])
    pad = int(conv.padding[0])
    Cout, Cin, _ = W.shape
    Wm = _rep(W.permute(2, 1, 0).reshape(k * Cin, Cout))       # [k*Cin, Cout]
    B = _rep(b.reshape(1, 1, -1)) if b is not None else None

    def _pad_time(x, left, right):
        if left == 0 and right == 0:
            return x
        C = int(x.shape[1])
        parts = []
        if left:
            parts.append(ttnn.multiply(ttnn.slice(x, [0, 0, 0], [1, C, left]), 0.0))
        parts.append(x)
        if right:
            parts.append(ttnn.multiply(ttnn.slice(x, [0, 0, 0], [1, C, right]), 0.0))
        return ttnn.concat(parts, dim=2)

    def forward(x, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        xp = _pad_time(x, pad, pad)
        T2 = int(xp.shape[-1])
        Lout = T2 - dil * (k - 1)
        taps = [ttnn.slice(xp, [0, 0, t * dil], [1, Cin, t * dil + Lout]) for t in range(k)]
        xc = ttnn.concat(taps, dim=1) if k > 1 else taps[0]    # [1, k*Cin, Lout]
        y = ttnn.matmul(ttnn.transpose(xc, 1, 2), Wm, compute_kernel_config=kcfg)  # [1,Lout,Cout]
        if B is not None:
            y = ttnn.add(y, B)
        return ttnn.transpose(y, 1, 2)                         # [1, Cout, Lout]

    return forward
