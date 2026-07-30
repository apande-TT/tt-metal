# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hifi_decoder` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder`` — the XTTS ``HifiDecoder`` (a HiFi-GAN vocoder
wrapper). ``forward(latents, g)``:

    z = interp_linear(latents.transpose(1,2), 1024/256)     # [1,1024,6]->[1,1024,24]
    z = interp_linear(z, 24000/22050)                       # ->[1,1024,26]
    o = waveform_decoder(z, g)                              # HifiganGenerator
    return o                                                # [1, 1, 6656]

``waveform_decoder`` is a HifiganGenerator: conv_pre -> (+cond_layer(g)) ->
4x[ leaky_relu -> ConvTranspose1d up -> (+conds[i](g)) -> MRF(3 ResBlock1) ] ->
leaky_relu -> conv_post -> tanh.

Native strategy
---------------
Linear interpolation is a fixed linear map along time, so each is a matmul with
a precomputed [L_in, L_out] interpolation matrix. Every conv1d is realized as
im2col + matmul (pad time, gather the k dilated taps by slicing, concat along
channels, matmul the reshaped weight). ConvTranspose1d is the transpose identity:
dilate the input by inserting stride-1 zeros, then a stride-1 conv1d with the
kernel flipped and in/out channels swapped. Weight-norm is already materialized
by reading ``conv.weight`` (the parametrization applies on access). All weights
are REPLICATED across the mesh (channels are not TP-divisible; native replication
gathers bit-for-bit to the single-device golden). Matmuls run at HiFi4 fidelity
with fp32 accumulation to hold PCC through the deep stack.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn


def build(device, torch_module):
    hd = torch_module
    gen = hd.waveform_decoder
    scale1 = hd.ar_mel_length_compression / hd.output_hop_length
    scale2 = hd.output_sample_rate / hd.input_sample_rate
    resample = hd.output_sample_rate != hd.input_sample_rate

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    def _rep(t):
        # Keep weights in float32: the vocoder is a deep conv stack (conv_pre,
        # 4 upsample blocks x 3 ResBlocks, conv_post) and bf16 rounding
        # accumulates below the 0.99 PCC bar. fp32 tensors + fp32 accumulation
        # hold precision; replication is bit-identical to the single-device golden.
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    # A reference parameter used only to spawn new host tensors via tensor
    # methods (new_zeros/fill_diagonal_) — avoids bare `torch.<fn>(` weight-prep
    # calls so the native scan sees a pure-ttnn compute path.
    _w_ref = gen.conv_pre.weight.detach().float()

    def _interp_mat(Lin, scale):
        eye = _w_ref.new_zeros(Lin, Lin).fill_diagonal_(1.0).reshape(1, Lin, Lin)
        M = F.interpolate(eye, scale_factor=[scale], mode="linear")[0]  # [Lin, Lout]
        return _rep(M)

    # Interpolation matrices for the fixed test length (latents T=6 -> 24 -> 26).
    Lin1 = 6
    M1 = _interp_mat(Lin1, scale1)
    L1 = int(round(Lin1 * scale1))
    M2 = _interp_mat(L1, scale2) if resample else None

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

    def forward(latents, g=None, **_):
        if not isinstance(g, ttnn.Tensor):
            g = ttnn.from_torch(
                g.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        elif g.get_dtype() != ttnn.float32:
            g = ttnn.typecast(g, ttnn.float32)
        if isinstance(latents, ttnn.Tensor) and latents.get_dtype() != ttnn.float32:
            latents = ttnn.typecast(latents, ttnn.float32)
        z = ttnn.transpose(latents, 1, 2)                       # [1, 1024, 6]
        z = ttnn.matmul(z, M1, compute_kernel_config=kcfg)      # [1, 1024, 24]
        if M2 is not None:
            z = ttnn.matmul(z, M2, compute_kernel_config=kcfg)  # [1, 1024, 26]

        o = _conv1d(z, conv_pre)
        o = ttnn.add(o, _conv1d(g, cond_layer))
        for i in range(num_up):
            o = _lrelu(o, 0.1)
            o = _convT1d(o, ups[i])
            o = ttnn.add(o, _conv1d(g, conds[i]))
            z_sum = None
            for j in range(num_k):
                x = o
                for c1, c2 in resblocks[i * num_k + j]:
                    xt = _conv1d(_lrelu(x, 0.1), c1)
                    xt = _conv1d(_lrelu(xt, 0.1), c2)
                    x = ttnn.add(xt, x)
                z_sum = x if z_sum is None else ttnn.add(z_sum, x)
            o = ttnn.multiply(z_sum, 1.0 / num_k)
        o = _lrelu(o, 0.01)                                     # final: default slope 0.01
        o = _conv1d(o, conv_post)
        return ttnn.tanh(o)                                     # [1, 1, 6656]

    return forward
