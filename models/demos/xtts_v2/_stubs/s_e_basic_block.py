# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `s_e_basic_block` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder.layer1.0`` — a ``SEBasicBlock``
(ResNet basic block + Squeeze-and-Excitation), here 32->32 channels, stride 1,
no downsample. ``forward(x)`` on a ``[N, C, H, W]`` feature map runs::

    out = conv1(x); out = relu(out); out = bn1(out)   # relu BEFORE bn1
    out = conv2(out); out = bn2(out)
    out = se(out)                                      # channel re-weighting
    out += (downsample(x) if present else x)
    out = relu(out)

Native strategy
---------------
Same pure-ttnn scheme proven in the parent `res_net_speaker_encoder` port:

  * Each Conv2d is im2col + matmul — zero-pad, gather the 3x3 taps with slices,
    concat on channels, matmul the reshaped weight ``[k*k*Cin, Cout]``.
  * BatchNorm at eval is a per-channel affine; ``bn2`` (no nonlinearity before
    it) is folded into ``conv2``'s weight/bias, while ``bn1`` (which follows the
    relu) stays an explicit per-channel scale/shift.
  * SE: global average over (H, W) -> two Linear projections (relu, sigmoid) ->
    broadcast channel gate.
  * float32 + fp32 accumulation holds PCC.

TP=8 scheme
-----------
The block's weights are 3x3 convs (32*32*9) and tiny SE projections (32<->4) —
none is a large model-dim matmul that benefits from a column/row split. Per the
TP principles (split large matmul weights; keep small projections / norms
REPLICATED), every weight is staged REPLICATED across the mesh via
``ReplicateTensorToMesh``; each chip computes the identical block, so the
gathered output equals the single-device golden bit-for-bit — the same
replicate-only scheme accepted for `parametrized_conv1d` / `mel_spectrogram`.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    blk = torch_module

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

    _base = blk.conv1.weight.detach().float()

    def _bn_ss(bn):
        g = bn.weight.detach().float()
        be = bn.bias.detach().float()
        rm = bn.running_mean.detach().float()
        rv = bn.running_var.detach().float()
        eps = float(bn.eps)
        scale = g / (rv + eps).sqrt()
        shift = be - rm * scale
        return scale, shift

    def _prep_conv2d(conv, fold_bn=None):
        W = conv.weight.detach().float()                        # [Cout, Cin, kh, kw]
        Cout, Cin, kh, kw = W.shape
        Wm = W.permute(2, 3, 1, 0).reshape(kh * kw * Cin, Cout)  # [k*k*Cin, Cout]
        b = conv.bias.detach().float() if conv.bias is not None else _base.new_zeros(Cout)
        if fold_bn is not None:
            scale, shift = _bn_ss(fold_bn)
            Wm = Wm * scale.reshape(1, -1)
            b = b * scale + shift
        return {
            "Wm": _rep(Wm), "b": _rep(b.reshape(1, 1, Cout)),
            "kh": kh, "kw": kw, "Cin": Cin, "Cout": Cout,
            "s": int(conv.stride[0]), "p": int(conv.padding[0]),
        }

    def _bn4(bn):
        scale, shift = _bn_ss(bn)
        C = scale.numel()
        return _rep(scale.reshape(1, C, 1, 1)), _rep(shift.reshape(1, C, 1, 1))

    def _prep_se(se_layer):
        fc = se_layer.fc
        return {
            "W1": _rep(fc[0].weight.detach().float().t()), "b1": _rep(fc[0].bias.detach().float().reshape(1, -1)),
            "W2": _rep(fc[2].weight.detach().float().t()), "b2": _rep(fc[2].bias.detach().float().reshape(1, -1)),
        }

    conv1 = _prep_conv2d(blk.conv1)                             # relu then bn1
    bn1 = _bn4(blk.bn1)
    conv2 = _prep_conv2d(blk.conv2, fold_bn=blk.bn2)            # fold bn2
    se_p = _prep_se(blk.se)
    down = None
    if getattr(blk, "downsample", None) is not None:
        down = _prep_conv2d(blk.downsample[0], fold_bn=blk.downsample[1])

    def _pad2d(x, p):
        if p == 0:
            return x
        C = int(x.shape[1]); H = int(x.shape[2])
        zl = ttnn.multiply(ttnn.slice(x, [0, 0, 0, 0], [1, C, H, p]), 0.0)
        x = ttnn.concat([zl, x, zl], dim=3)
        W2 = int(x.shape[3])
        zt = ttnn.multiply(ttnn.slice(x, [0, 0, 0, 0], [1, C, p, W2]), 0.0)
        return ttnn.concat([zt, x, zt], dim=2)

    def _conv2d(x, c):
        kh, kw, Cin, Cout, s, p = c["kh"], c["kw"], c["Cin"], c["Cout"], c["s"], c["p"]
        xp = _pad2d(x, p)
        Hp = int(xp.shape[2]); Wp = int(xp.shape[3])
        Hout = (Hp - kh) // s + 1
        Wout = (Wp - kw) // s + 1
        taps = []
        for i in range(kh):
            for j in range(kw):
                taps.append(ttnn.slice(
                    xp, [0, 0, i, j],
                    [1, Cin, i + s * (Hout - 1) + 1, j + s * (Wout - 1) + 1],
                    [1, 1, s, s],
                ))
        xc = ttnn.concat(taps, dim=1) if len(taps) > 1 else taps[0]
        K = kh * kw * Cin
        xc = ttnn.reshape(xc, [1, K, Hout * Wout])
        xc = ttnn.transpose(xc, 1, 2)
        y = ttnn.matmul(xc, c["Wm"], compute_kernel_config=kcfg)
        y = ttnn.add(y, c["b"])
        y = ttnn.transpose(y, 1, 2)
        return ttnn.reshape(y, [1, Cout, Hout, Wout])

    def _apply_bn4(x, ss):
        scale, shift = ss
        return ttnn.add(ttnn.multiply(x, scale), shift)

    def _se(out):
        C = int(out.shape[1])
        y = ttnn.mean(out, dim=[2, 3])
        y = ttnn.reshape(y, [1, C])
        y = ttnn.add(ttnn.matmul(y, se_p["W1"], compute_kernel_config=kcfg), se_p["b1"])
        y = ttnn.relu(y)
        y = ttnn.add(ttnn.matmul(y, se_p["W2"], compute_kernel_config=kcfg), se_p["b2"])
        y = ttnn.sigmoid(y)
        y = ttnn.reshape(y, [1, C, 1, 1])
        return ttnn.multiply(out, y)

    def forward(x, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        out = _conv2d(x, conv1)
        out = ttnn.relu(out)
        out = _apply_bn4(out, bn1)
        out = _conv2d(out, conv2)                              # bn2 folded
        out = _se(out)
        residual = _conv2d(x, down) if down is not None else x
        out = ttnn.add(out, residual)
        return ttnn.relu(out)

    return forward
