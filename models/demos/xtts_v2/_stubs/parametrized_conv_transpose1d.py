# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrized_conv_transpose1d` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.waveform_decoder.ups.0`` — a weight-normed
``ConvTranspose1d`` (in=512, out=256, kernel=16, stride=8, pad=4, output_padding=0,
bias=True). ``forward(x)`` upsamples ``[1,512,26] -> [1,256,208]``. Reading
``conv.weight`` materializes the weight-norm parametrization.

Native strategy
---------------
ConvTranspose1d is the transpose identity: dilate the input by inserting
``stride-1`` zeros between samples, then run a stride-1 conv1d with the kernel
flipped and in/out channels swapped (``Weff = W.permute(1,0,2).flip(-1)``), with
effective padding ``k-1-pad``. The conv1d itself is im2col + matmul. Weight is
REPLICATED across the mesh (channels are not TP-divisible; replication gathers
bit-for-bit to the single-device golden). float32 + fp32 accumulation hold PCC.
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

    W = conv.weight.detach()                                   # convT: [Cin, Cout, k]
    b = conv.bias.detach() if conv.bias is not None else None
    k = int(conv.kernel_size[0])
    stride = int(conv.stride[0])
    pad = int(conv.padding[0])
    outpad = int(conv.output_padding[0])
    # Effective stride-1 conv weight: swap in/out, flip taps.
    Weff = W.permute(1, 0, 2).flip(-1).contiguous()            # [Cout, Cin, k]
    Cout, Cin, _ = Weff.shape
    Wm = _rep(Weff.permute(2, 1, 0).reshape(k * Cin, Cout))    # [k*Cin, Cout]
    B = _rep(b.reshape(1, 1, -1)) if b is not None else None
    eff_pad = k - 1 - pad                                       # padding for the stride-1 conv

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

    def _conv1d_stride1(x):
        xp = _pad_time(x, eff_pad, eff_pad)
        T2 = int(xp.shape[-1])
        Lout = T2 - (k - 1)
        Cin_x = int(x.shape[1])
        taps = [ttnn.slice(xp, [0, 0, t], [1, Cin_x, t + Lout]) for t in range(k)]
        xc = ttnn.concat(taps, dim=1) if k > 1 else taps[0]    # [1, k*Cin, Lout]
        y = ttnn.matmul(ttnn.transpose(xc, 1, 2), Wm, compute_kernel_config=kcfg)  # [1,Lout,Cout]
        if B is not None:
            y = ttnn.add(y, B)
        return ttnn.transpose(y, 1, 2)                         # [1, Cout, Lout]

    def forward(x, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        C = int(x.shape[1])
        T = int(x.shape[-1])
        # Dilate: insert (stride-1) zeros between input samples.
        if stride > 1:
            xd = ttnn.reshape(x, [1, C, T, 1])
            xd = ttnn.pad(xd, [(0, 0), (0, 0), (0, 0), (0, stride - 1)], value=0.0)  # [1,C,T,stride]
            xd = ttnn.reshape(xd, [1, C, T * stride])
            xd = ttnn.slice(xd, [0, 0, 0], [1, C, (T - 1) * stride + 1])
        else:
            xd = x
        y = _conv1d_stride1(xd)
        if outpad:
            y = _pad_time(y, 0, outpad)
        return y

    return forward
