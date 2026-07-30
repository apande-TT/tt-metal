# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hifigan_generator` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.waveform_decoder`` — the XTTS ``HifiganGenerator``
(a HiFi-GAN vocoder). ``forward(x, g)``:

    o = conv_pre(x)                                      # [.,1024,T]->[.,512,T]
    o = o + cond_layer(g)                                # g conditioning
    for i in range(num_upsamples):
        o = leaky_relu(o, 0.1)
        o = ups[i](o)                                    # ConvTranspose1d up
        o = o + conds[i](g)
        o = mean_j MRF_ResBlock1(o)                      # num_kernels ResBlock1s
    o = leaky_relu(o)                                    # default slope 0.01
    o = conv_post(o)
    return tanh(o)                                       # [1, 1, 6656]

Native strategy
---------------
Every conv1d is realized as im2col + matmul (pad time, gather the k dilated taps
by slicing, concat along channels, matmul the reshaped weight). ConvTranspose1d
is the transpose identity: dilate the input by inserting stride-1 zeros, then a
stride-1 conv1d with the kernel flipped and in/out channels swapped. Weight-norm
is already materialized by reading ``conv.weight`` (the parametrization applies on
access). All weights are REPLICATED across the mesh (channels are not TP-divisible;
native replication gathers bit-for-bit to the single-device golden). Compute runs
in float32 with fp32 accumulation to hold PCC through the deep conv stack — bf16
rounding accumulates below the 0.99 bar (this is the exact scheme that graduated
the enclosing ``hifi_decoder``).
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    gen = torch_module

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

    def _conv_meta(conv, transpose=False):
        W = conv.weight.detach()
        b = conv.bias.detach() if conv.bias is not None else None
        k = int(conv.kernel_size[0])
        if transpose:
            # convT weight [Cin, Cout, k] -> effective conv weight [Cout, Cin, k] flipped.
            Weff = W.permute(1, 0, 2).flip(-1).contiguous()
            Cout, Cin, _ = Weff.shape
            Wm = Weff.permute(2, 1, 0).reshape(k * Cin, Cout)
            return {
                "Wm": _rep(Wm), "b": _rep(b.reshape(1, 1, -1)) if b is not None else None,
                "k": k, "stride": int(conv.stride[0]), "pad": int(conv.padding[0]),
                "outpad": int(conv.output_padding[0]), "transpose": True,
            }
        Cout, Cin, _ = W.shape
        Wm = W.permute(2, 1, 0).reshape(k * Cin, Cout)
        return {
            "Wm": _rep(Wm), "b": _rep(b.reshape(1, 1, -1)) if b is not None else None,
            "k": k, "dil": int(conv.dilation[0]), "pad": int(conv.padding[0]),
            "transpose": False,
        }

    conv_pre = _conv_meta(gen.conv_pre)
    cond_layer = _conv_meta(gen.cond_layer)
    ups = [_conv_meta(u, transpose=True) for u in gen.ups]
    conds = [_conv_meta(c) for c in gen.conds]
    resblocks = []
    for rb in gen.resblocks:
        resblocks.append([
            (_conv_meta(c1), _conv_meta(c2)) for c1, c2 in zip(rb.convs1, rb.convs2)
        ])
    num_up = int(gen.num_upsamples)
    num_k = int(gen.num_kernels)
    conv_post = _conv_meta(gen.conv_post)

    def _pad_time(x, left, right):
        # TILE-layout ttnn.pad forbids nonzero FRONT padding, so build the zero
        # pad natively by zeroing a leading slice and concatenating.
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

    def _conv1d(x, meta):
        # x: [1, Cin, T]; stride-1 im2col + matmul.
        k, dil, pad = meta["k"], meta["dil"], meta["pad"]
        xp = _pad_time(x, pad, pad)
        T2 = int(xp.shape[-1])
        Lout = T2 - dil * (k - 1)
        Cin = int(x.shape[1])
        taps = [ttnn.slice(xp, [0, 0, t * dil], [1, Cin, t * dil + Lout]) for t in range(k)]
        xc = ttnn.concat(taps, dim=1) if k > 1 else taps[0]     # [1, k*Cin, Lout]
        y = ttnn.matmul(ttnn.transpose(xc, 1, 2), meta["Wm"], compute_kernel_config=kcfg)  # [1,Lout,Cout]
        if meta["b"] is not None:
            y = ttnn.add(y, meta["b"])
        return ttnn.transpose(y, 1, 2)                          # [1, Cout, Lout]

    def _convT1d(x, meta):
        # Dilate input by inserting stride-1 zeros, then stride-1 conv with Weff.
        s, k, pad, outpad = meta["stride"], meta["k"], meta["pad"], meta["outpad"]
        Cin = int(x.shape[1])
        T = int(x.shape[-1])
        if s > 1:
            xd = ttnn.reshape(x, [1, Cin, T, 1])
            xd = ttnn.pad(xd, [(0, 0), (0, 0), (0, 0), (0, s - 1)], value=0.0)  # [1,Cin,T,s]
            xd = ttnn.reshape(xd, [1, Cin, T * s])
            xd = ttnn.slice(xd, [0, 0, 0], [1, Cin, (T - 1) * s + 1])
        else:
            xd = x
        conv_meta = {"Wm": meta["Wm"], "b": meta["b"], "k": k, "dil": 1, "pad": k - 1 - pad}
        y = _conv1d(xd, conv_meta)
        if outpad:
            y = _pad_time(y, 0, outpad)
        return y

    def _lrelu(x, slope):
        return ttnn.leaky_relu(x, negative_slope=slope)

    def forward(x, g=None, **_):
        if not isinstance(g, ttnn.Tensor):
            g = ttnn.from_torch(
                g.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        elif g.get_dtype() != ttnn.float32:
            g = ttnn.typecast(g, ttnn.float32)
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        # The captured primary input is [Cin, T] (batch squeezed); restore the
        # leading batch axis the conv im2col path expects. The torch golden
        # promotes to batch=1 the moment the batched g conditioning is added, so
        # this matches bit-for-bit.
        if len(x.shape) == 2:
            C0, T0 = int(x.shape[0]), int(x.shape[1])
            x = ttnn.reshape(x, [1, C0, T0])

        o = _conv1d(x, conv_pre)
        o = ttnn.add(o, _conv1d(g, cond_layer))
        for i in range(num_up):
            o = _lrelu(o, 0.1)
            o = _convT1d(o, ups[i])
            o = ttnn.add(o, _conv1d(g, conds[i]))
            z_sum = None
            for j in range(num_k):
                x_rb = o
                for c1, c2 in resblocks[i * num_k + j]:
                    xt = _conv1d(_lrelu(x_rb, 0.1), c1)
                    xt = _conv1d(_lrelu(xt, 0.1), c2)
                    x_rb = ttnn.add(xt, x_rb)
                z_sum = x_rb if z_sum is None else ttnn.add(z_sum, x_rb)
            o = ttnn.multiply(z_sum, 1.0 / num_k)
        o = _lrelu(o, 0.01)                                     # final: default slope 0.01
        o = _conv1d(o, conv_post)
        return ttnn.tanh(o)                                     # [1, 1, 6656]

    return forward
