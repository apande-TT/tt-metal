# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `res_block1` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.waveform_decoder.resblocks.0`` — a HiFi-GAN
``ResBlock1``. It holds two ``ModuleList``s of three weight-normed ``Conv1d``s
each (in=out=256, kernel=3, bias=True): ``convs1`` with dilations (1, 3, 5) and
matching pads, and ``convs2`` with dilation=1 pad=1. ``forward(x)`` runs, for
each ``(c1, c2)`` pair::

    xt = leaky_relu(x, 0.1); xt = c1(xt)
    xt = leaky_relu(xt, 0.1); xt = c2(xt)
    x  = xt + x

Each conv is realized as im2col + matmul (the same port used for the sibling
`parametrized_conv1d`, which is ``convs1.0`` of this very block): zero-pad the
time axis, gather the ``k`` dilated taps by slicing, concat along channels, and
matmul the reshaped effective weight ``[k*Cin, Cout]`` (reading ``conv.weight``
materializes the weight-norm parametrization). float32 + fp32 accumulation hold
PCC across the six stacked convs and residual adds.

TP=8 scheme (genuine column-parallel per conv)
----------------------------------------------
Every conv is a matmul whose ``Cout`` output feeds a per-element op
(``leaky_relu`` / residual add), so each is COLUMN-parallel: split the 256
output features across the 8 chips (``ShardTensorToMesh(dim=1)`` on the reshaped
weight ``[k*Cin, Cout]`` and ``dim=2`` on the bias). Chip ``i`` computes its
32-wide output slice from the replicated input, then ``all_gather(dim=-1)``
reassembles the full ``[1, Cout, Lout]`` on every chip — exactly the scheme
proven for `conv1_d`. leaky_relu and the residual run on the gathered full
tensor, so the gathered output equals the single-device golden; only the
placement of each conv's output columns differs.
"""

from __future__ import annotations

import torch

import ttnn


_LRELU_SLOPE = 0.1


def build(device, torch_module):
    rb = torch_module

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    def _shard_cols(t):
        # weight [k*Cin, Cout] -> split Cout across the mesh
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )

    def _shard_bias(t):
        # bias [1, 1, Cout] -> split Cout across the mesh (matches weight cols)
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2),
        )

    def _prep(conv):
        W = conv.weight.detach()                                  # [Cout, Cin, k]
        b = conv.bias.detach()                                    # [Cout]
        k = int(conv.kernel_size[0])
        dil = int(conv.dilation[0])
        pad = int(conv.padding[0])
        Cout, Cin, _ = W.shape
        Wm = _shard_cols(W.permute(2, 1, 0).reshape(k * Cin, Cout))  # [k*Cin, Cout/8]
        B = _shard_bias(b.reshape(1, 1, -1))                         # [1, 1, Cout/8]
        return {"Wm": Wm, "B": B, "k": k, "dil": dil, "pad": pad, "Cin": Cin}

    convs1 = [_prep(c) for c in rb.convs1]
    convs2 = [_prep(c) for c in rb.convs2]

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

    def _conv(x, p):
        # x: [1, Cin, T] replicated (full channels on every chip)
        k, dil, pad, Cin = p["k"], p["dil"], p["pad"], p["Cin"]
        xp = _pad_time(x, pad, pad)
        T2 = int(xp.shape[-1])
        Lout = T2 - dil * (k - 1)
        taps = [ttnn.slice(xp, [0, 0, t * dil], [1, Cin, t * dil + Lout]) for t in range(k)]
        xc = ttnn.concat(taps, dim=1) if k > 1 else taps[0]       # [1, k*Cin, Lout]
        # Column-parallel: chip i produces its Cout/8 output slice.
        y = ttnn.matmul(ttnn.transpose(xc, 1, 2), p["Wm"], compute_kernel_config=kcfg)  # [1,Lout,Cout/8]
        y = ttnn.add(y, p["B"])
        # Reassemble the full output-channel axis across the mesh.
        y = ttnn.all_gather(y, dim=-1, num_links=1, topology=ttnn.Topology.Linear)       # [1,Lout,Cout]
        return ttnn.transpose(y, 1, 2)                            # [1, Cout, Lout]

    def forward(x, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        for p1, p2 in zip(convs1, convs2):
            xt = ttnn.leaky_relu(x, _LRELU_SLOPE)
            xt = _conv(xt, p1)
            xt = ttnn.leaky_relu(xt, _LRELU_SLOPE)
            xt = _conv(xt, p2)
            x = ttnn.add(xt, x)
        return x

    return forward
