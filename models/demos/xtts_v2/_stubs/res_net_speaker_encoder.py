# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `res_net_speaker_encoder` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder`` — a ``ResNetSpeakerEncoder``
(ASP-pooled SE-ResNet34 style speaker embedder). ``forward(x, l2_norm=False)``
takes a raw waveform ``[N, 1, T]`` and produces a ``[N, 512]`` speaker embedding:

    x = torch_spec(x.squeeze(1))          # PreEmphasis + MelSpectrogram -> [N,64,F]
    x = (x + 1e-6).log()                  # log_input
    x = instancenorm(x).unsqueeze(1)      # per-channel norm over time -> [N,1,64,F]
    x = relu(bn1(conv1(x)))               # note: relu BEFORE bn1 in the block
    x = layer4(layer3(layer2(layer1(x)))) # SE-ResNet stages -> [N,256,8,F/8]
    x = x.reshape(N, 2048, F/8)
    w = softmax(attention(x))             # channel attention -> weights
    x = cat(sum(x*w), sqrt(sum(x^2 w)-mu^2))  # ASP pooling -> [N,4096]
    x = fc(x)                             # -> [N,512]

Native strategy
---------------
Everything is expressed as ttnn matmuls + elementwise ops so the native probe
sees a pure-ttnn compute path (no torch compute during forward):

  * STFT (in torch_spec) is linear up to |.|^2, so it is realized as DFT matmuls
    with precomputed cos/sin bases + a mel filterbank matmul — the exact scheme
    proven for the graduated `mel_spectrogram` component. Reflect padding uses an
    anti-diagonal exchange matmul (ttnn has no flip).
  * Every Conv2d is a native ``ttnn.conv2d`` and the whole SE-ResNet trunk is kept
    in the flattened channels-LAST form conv2d both consumes and produces, so
    consecutive convs chain with no layout op between them (only the trunk entry
    and exit convert). The 1x1 Conv1ds of the attention head stay matmuls.
    BatchNorm at eval is a per-channel affine, folded into the preceding conv's
    weight/bias whenever no nonlinearity sits between them (conv2->bn2,
    downsample->bn); the relu-then-bn1 pair keeps bn1 as an explicit per-channel
    scale/shift.
  * The trunk runs in bf16 (it is bytes-moved bound) with fp32 accumulation; the
    STFT front end, the instancenorm statistics and the ASP sum-of-squares pooling
    stay float32, which is what holds PCC across the deep chain.

TP=8 scheme
-----------
This encoder is convolution/attention-pooling dominated with only small
per-channel and 1x1 matmuls; none of its weights is a large model-dim projection
that benefits from a column/row split. Per the TP principles (split large matmul
weights; keep norms/embeddings/filters/small projections REPLICATED), every fixed
basis and every weight is staged REPLICATED across the mesh via
``ReplicateTensorToMesh``. Each chip computes the identical forward, so the
gathered output equals the single-device golden bit-for-bit — the placement is a
valid (replicate-only) TP scheme, exactly as for the sibling `mel_spectrogram`,
`parametrized_conv1d`, and `adaptive_avg_pool2d` shard graduations.
"""

from __future__ import annotations

import math

import torch

import ttnn


def build(device, torch_module):
    se = torch_module

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    def _rep(t):
        # A bias / norm scale has logical height 1, so a DEVICE tilize has to val-pad it
        # 1 -> 32 rows, which ttnn runs on a SINGLE core -- the profile's grid=tiny
        # TilizeDeviceOperation, hundreds of calls for a few KB each. Tilizing those on the
        # HOST is free by comparison. Real matrices are already tile-shaped (no val padding)
        # and keep the multicore device path, where host-tilizing megabytes is the worse trade.
        if t.dim() >= 2 and int(t.shape[-2]) == 1:
            return ttnn.to_device(
                ttnn.from_torch(t.contiguous().to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(device)),
                device)
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    # ---------------- torch_spec: PreEmphasis + MelSpectrogram ----------------
    pe = se.torch_spec[0]
    _f = pe.filter.detach().flatten().tolist()                  # [-coef, 1.0]
    pe_w0, pe_w1 = float(_f[0]), float(_f[1])

    ms = se.torch_spec[1]
    spec = ms.spectrogram
    fb_t = ms.mel_scale.fb.detach().float()                     # [n_freq, n_mels]
    win_t = spec.window.detach().float()
    n_fft = int(spec.n_fft)
    hop = int(spec.hop_length)
    win_length = int(spec.win_length)
    pad = n_fft // 2
    n_freq = n_fft // 2 + 1
    n_mels = int(fb_t.shape[1])

    # fixed STFT tensors (tensor-method construction, no torch.<fn>( calls)
    base = win_t
    n = base.new_ones(n_fft).cumsum(0) - 1
    kk = base.new_ones(n_freq).cumsum(0) - 1
    ang = (n.unsqueeze(1) * kk.unsqueeze(0)) * (2.0 * math.pi / n_fft)
    Dcos = _rep(ang.cos())                                      # [n_fft, n_freq]
    Dsin = _rep(ang.sin())
    _off = (n_fft - win_length) // 2
    wpad_t = base.new_zeros(n_fft)
    wpad_t[_off:_off + win_length] = win_t
    Wpad = _rep(wpad_t.reshape(1, n_fft))                       # [1, n_fft]
    _ar = (base.new_ones(pad).cumsum(0) - 1).long()
    J_t = base.new_zeros(pad, pad)
    J_t[_ar, (pad - 1 - _ar)] = 1.0
    J = _rep(J_t)                                               # [pad, pad]
    FB = _rep(fb_t)                                             # [n_freq, n_mels]

    log_input = bool(getattr(se, "log_input", True))
    inorm_eps = float(getattr(se.instancenorm, "eps", 1e-5))

    # ---------------- narrow TRUNK format ----------------
    # The SE-ResNet trunk is DATAMOVE-bound, not compute-bound: each im2col conv gathers
    # k*k taps into a multi-MB stack and then merges both tile dims, so its time is bytes
    # MOVED, and halving the stored format ~halves it. Carry the trunk (activations AND
    # weights) in bf16 and cast back at the trunk exit, leaving the wide format where
    # cancellation actually matters: the STFT/mel front end, the instancenorm statistics,
    # and the sum-of-squares ASP pooling. fp32_dest_acc_en stays on, so this changes the
    # STORED format only, never accumulation precision.
    TRUNK_DT = ttnn.bfloat16

    def _rep_t(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=TRUNK_DT,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    # ---------------- conv / bn helpers ----------------
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
        """Weights for a NATIVE ttnn.conv2d: HOST tensors in torch [Cout,Cin,kh,kw]
        order, which conv2d prepares (tilizes/shards) itself on first use."""
        W = conv.weight.detach().float()                        # [Cout, Cin, kh, kw]
        Cout, Cin, kh, kw = W.shape
        b = conv.bias.detach().float() if conv.bias is not None else base.new_zeros(Cout)
        if fold_bn is not None:
            scale, shift = _bn_ss(fold_bn)
            W = W * scale.reshape(-1, 1, 1, 1)
            b = b * scale + shift
        return {
            "W": ttnn.from_torch(W.contiguous().to(torch.bfloat16), TRUNK_DT),
            "b": ttnn.from_torch(b.reshape(1, 1, 1, Cout).contiguous().to(torch.bfloat16),
                                 TRUNK_DT),
            "kh": kh, "kw": kw, "Cin": Cin, "Cout": Cout,
            "s": int(conv.stride[0]), "p": int(conv.padding[0]),
        }

    def _bn4(bn):
        # channels-LAST per-channel affine: [1,1,1,C] broadcasts over the flattened NHWC
        scale, shift = _bn_ss(bn)
        C = scale.numel()
        return _rep_t(scale.reshape(1, 1, 1, C)), _rep_t(shift.reshape(1, 1, 1, C))

    def _prep_se(se_layer):
        fc = se_layer.fc
        W1 = fc[0].weight.detach().float()                      # [r, C]
        b1 = fc[0].bias.detach().float()
        W2 = fc[2].weight.detach().float()                      # [C, r]
        b2 = fc[2].bias.detach().float()
        return {
            "W1": _rep_t(W1.t()), "b1": _rep_t(b1.reshape(1, -1)),
            "W2": _rep_t(W2.t()), "b2": _rep_t(b2.reshape(1, -1)),
        }

    def _prep_block(blk):
        d = {
            "conv1": _prep_conv2d(blk.conv1),                   # bias=False, relu then bn1
            "bn1": _bn4(blk.bn1),
            "conv2": _prep_conv2d(blk.conv2, fold_bn=blk.bn2),  # fold bn2
            "se": _prep_se(blk.se),
            "down": None,
        }
        if getattr(blk, "downsample", None) is not None:
            d["down"] = _prep_conv2d(blk.downsample[0], fold_bn=blk.downsample[1])
        return d

    top_conv1 = _prep_conv2d(se.conv1)                          # bias=True, relu then bn1
    top_bn1 = _bn4(se.bn1)
    layers = [[_prep_block(b) for b in getattr(se, f"layer{i}")] for i in range(1, 5)]

    # ---------------- attention + fc ----------------
    def _prep_conv1d_1x1(conv):
        W = conv.weight.detach().float()                        # [Cout, Cin, 1]
        Cout, Cin, _ = W.shape
        b = conv.bias.detach().float() if conv.bias is not None else base.new_zeros(Cout)
        return {"WT": _rep(W.reshape(Cout, Cin).t()), "b": _rep(b.reshape(1, 1, Cout))}

    att_c1 = _prep_conv1d_1x1(se.attention[0])
    att_bn = se.attention[2]
    _asc, _ash = _bn_ss(att_bn)
    att_bn_scale = _rep(_asc.reshape(1, -1, 1))
    att_bn_shift = _rep(_ash.reshape(1, -1, 1))
    att_c2 = _prep_conv1d_1x1(se.attention[3])

    fc_W = _rep(se.fc.weight.detach().float().t())              # [4096, 512]
    fc_b = _rep(se.fc.bias.detach().float().reshape(1, -1))

    # ================= forward =================
    def _stft_mel(x):
        # x: [1, L] waveform -> [1, n_mels, n_frame] mel power spectrogram
        L = int(x.shape[-1])
        if len(x.shape) != 2:
            x = ttnn.reshape(x, [1, L])
        left = ttnn.slice(x, [0, 1], [1, 1 + pad])
        right = ttnn.slice(x, [0, L - 1 - pad], [1, L - 1])
        left_rev = ttnn.matmul(left, J, compute_kernel_config=kcfg)
        right_rev = ttnn.matmul(right, J, compute_kernel_config=kcfg)
        padded = ttnn.concat([left_rev, x, right_rev], dim=1)
        Lp = L + 2 * pad
        n_frame = 1 + (Lp - n_fft) // hop
        frames = ttnn.concat(
            [ttnn.slice(padded, [0, i * hop], [1, i * hop + n_fft]) for i in range(n_frame)],
            dim=0,
        )                                                       # [n_frame, n_fft]
        fw = ttnn.multiply(frames, Wpad)
        re = ttnn.matmul(fw, Dcos, compute_kernel_config=kcfg)
        im = ttnn.matmul(fw, Dsin, compute_kernel_config=kcfg)
        power = ttnn.add(ttnn.multiply(re, re), ttnn.multiply(im, im))
        mel = ttnn.matmul(power, FB, compute_kernel_config=kcfg)  # [n_frame, n_mels]
        mel = ttnn.reshape(mel, [1, n_frame, n_mels])
        return ttnn.transpose(mel, 1, 2)                        # [1, n_mels, n_frame]

    def _conv2d(x, c, H, W):
        """ONE native ttnn.conv2d. Returns (y, Hout, Wout) with y in the flattened
        channels-last form conv2d both produces and accepts, so consecutive convs chain
        with NO layout op between them -- H/W travel alongside as plain ints.

        This replaces a hand-rolled im2col (zero-pad by slice+concat, gather k*k taps with
        strided slices, concat them on channels, merge both tile dims, transpose, matmul,
        transpose, split back). That expansion built a k*k-times inflated stack -- 14.8 MB
        fp32 at the widest layer -- whose 4D->3D merge alone was 94 calls x ~954 us = 90 ms,
        25% of device time, and it is pure plumbing: conv2d does the same convolution with
        a fused sliding-window kernel and never materializes the stack."""
        y, [Hout, Wout], [wt, bt] = ttnn.conv2d(
            input_tensor=x, weight_tensor=c["W"], bias_tensor=c["b"], device=device,
            in_channels=c["Cin"], out_channels=c["Cout"], batch_size=1,
            input_height=H, input_width=W,
            kernel_size=(c["kh"], c["kw"]), stride=(c["s"], c["s"]),
            padding=(c["p"], c["p"]), dilation=(1, 1), groups=1,
            dtype=TRUNK_DT, compute_config=kcfg,
            # DRAM out on purpose: a persistent multi-MB L1 residency here moves the L1
            # high-water mark for every LATER stage on this device.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True, return_weights_and_bias=True,
        )
        # conv2d hands back the prepared (tilized/laid-out) weights; keep them so the
        # preparation happens once per conv instead of once per call.
        c["W"], c["b"] = wt, bt
        return y, Hout, Wout

    def _apply_bn4(x, ss):
        scale, shift = ss
        return ttnn.add(ttnn.multiply(x, scale), shift)

    def _se(out, se_p, C):
        # squeeze over the flattened spatial axis of the channels-last tensor
        y = ttnn.mean(out, dim=2, keepdim=True)                  # [1,1,1,C]
        y = ttnn.reshape(y, [1, C])
        y = ttnn.add(ttnn.matmul(y, se_p["W1"], compute_kernel_config=kcfg), se_p["b1"])
        y = ttnn.relu(y)
        y = ttnn.add(ttnn.matmul(y, se_p["W2"], compute_kernel_config=kcfg), se_p["b2"])
        y = ttnn.sigmoid(y)                                     # [1, C]
        y = ttnn.reshape(y, [1, 1, 1, C])
        return ttnn.multiply(out, y)

    def _block(x, b, H, W):
        out, H1, W1 = _conv2d(x, b["conv1"], H, W)
        out = ttnn.relu(out)
        out = _apply_bn4(out, b["bn1"])
        out, H2, W2 = _conv2d(out, b["conv2"], H1, W1)          # bn2 folded
        out = _se(out, b["se"], b["conv2"]["Cout"])
        # no downsample => stride 1 => (H2,W2)==(H,W), so x is already conformant
        residual = _conv2d(x, b["down"], H, W)[0] if b["down"] is not None else x
        out = ttnn.add(out, residual)
        return ttnn.relu(out), H2, W2

    def _conv1d_1x1(x, c):
        xt = ttnn.transpose(x, 1, 2)                            # [1, T, Cin]
        y = ttnn.add(ttnn.matmul(xt, c["WT"], compute_kernel_config=kcfg), c["b"])
        return ttnn.transpose(y, 1, 2)                          # [1, Cout, T]

    def forward(x, l2_norm=False, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        L = int(x.shape[-1])
        x = ttnn.reshape(x, [1, L])                             # squeeze(1)

        # torch_spec: pre-emphasis then mel spectrogram
        xp = ttnn.concat([ttnn.slice(x, [0, 1], [1, 2]), x], dim=1)  # reflect-pad 1 left
        a = ttnn.slice(xp, [0, 1], [1, L + 1])
        bb = ttnn.slice(xp, [0, 0], [1, L])
        x = ttnn.add(ttnn.multiply(bb, pe_w0), ttnn.multiply(a, pe_w1))  # [1, L]
        x = _stft_mel(x)                                        # [1, 64, F]

        if log_input:
            x = ttnn.log(ttnn.add(x, 1e-6))

        # instancenorm (affine=False): per-channel norm over time
        mean = ttnn.mean(x, dim=-1, keepdim=True)
        xc = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)
        x = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, inorm_eps)))  # [1, 64, F]

        F = int(x.shape[-1])
        # TRUNK ENTRY. The trunk is channels-LAST (what ttnn.conv2d consumes/produces) and
        # bf16 (see _rep_t). unsqueeze(1) on [1,64,F] gives NCHW [1,1,64,F] with ONE
        # channel, whose element order is already NHWC [1,64,F,1], so this costs nothing.
        Hc, Wc = 64, F
        x = ttnn.typecast(ttnn.reshape(x, [1, Hc, Wc, 1]), TRUNK_DT)

        # stem
        x, Hc, Wc = _conv2d(x, top_conv1, Hc, Wc)
        x = ttnn.relu(x)
        x = _apply_bn4(x, top_bn1)

        # SE-ResNet stages
        for stage in layers:
            for blk in stage:
                x, Hc, Wc = _block(x, blk, Hc, Wc)

        # TRUNK EXIT: back to fp32 NCHW, then [1, C, H, W] -> [1, C*H, W]. The attention
        # softmax and the ASP sum-of-squares below subtract mu^2 from E[x^2], so they stay
        # in the wide format. This is the ONLY channels-last -> channels-first move left.
        C = int(x.shape[-1])
        x = ttnn.typecast(ttnn.reshape(x, [1, Hc, Wc, C]), ttnn.float32)
        x = ttnn.permute(x, [0, 3, 1, 2])                      # [1, C, H, W]
        x = ttnn.reshape(x, [1, C * Hc, Wc])                   # [1, 2048, F']
        H = Hc

        # attention: conv1d -> relu -> bn1d -> conv1d -> softmax(time)
        w = _conv1d_1x1(x, att_c1)
        w = ttnn.relu(w)
        w = ttnn.add(ttnn.multiply(w, att_bn_scale), att_bn_shift)
        w = _conv1d_1x1(w, att_c2)
        w = ttnn.softmax(w, dim=-1)                            # [1, 2048, F']

        # ASP pooling
        mu = ttnn.sum(ttnn.multiply(x, w), dim=2)             # [1, 2048]
        ex2 = ttnn.sum(ttnn.multiply(ttnn.multiply(x, x), w), dim=2)
        mu = ttnn.reshape(mu, [1, C * H])
        ex2 = ttnn.reshape(ex2, [1, C * H])
        sg = ttnn.sqrt(ttnn.clamp(ttnn.subtract(ex2, ttnn.multiply(mu, mu)), 1e-5, 1e12))
        x = ttnn.concat([mu, sg], dim=1)                       # [1, 4096]

        # fc
        x = ttnn.add(ttnn.matmul(x, fc_W, compute_kernel_config=kcfg), fc_b)  # [1, 512]
        return x

    return forward
