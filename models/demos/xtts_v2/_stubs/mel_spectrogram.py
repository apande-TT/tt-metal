# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mel_spectrogram` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder.torch_spec.1`` — a
``torchaudio.transforms.MelSpectrogram`` (n_fft=512, win_length=400,
hop_length=160, power=2.0, center=True, pad_mode='reflect', onesided=True,
n_mels=64). It maps a waveform ``[.., time]`` to a mel power spectrogram
``[.., 64, n_frame]``: ``mel_scale(|STFT(waveform)|^2)``.

Native strategy
---------------
The STFT is linear up to the magnitude, so it becomes matmuls:

  1. Reflect-pad the waveform by ``n_fft//2`` on each side (center=True). Reflect
     is a fixed permutation of the boundary samples — realized as a matmul of the
     two edge slices with a precomputed anti-diagonal exchange matrix ``J``, then
     concatenated (ttnn has no flip op).
  2. Frame into ``n_frame`` windows of length ``n_fft`` at stride ``hop`` (im2col
     via contiguous slices).
  3. Window each frame (the length-400 window zero-padded/centered to 512).
  4. Real DFT as two matmuls with precomputed cos / sin basis matrices
     ``[n_fft, n_freq]`` (n_freq = n_fft//2+1 = 257, onesided).
  5. Power spectrum ``|X|^2 = re^2 + im^2`` (power=2.0).
  6. Mel filterbank matmul with ``fb`` ``[257, 64]``, then transpose to
     ``[.., 64, n_frame]``.

All fixed tensors (cos/sin basis, padded window, exchange matrix, filterbank) are
staged REPLICATED across the mesh — none is TP-divisible, and replication gathers
bit-for-bit to the single-device golden. They are built with tensor METHODS
(new_zeros/cumsum/cos/sin/index-assign) rather than bare ``torch.<fn>(`` calls so
the native scan sees a pure-ttnn compute path. float32 throughout holds PCC.
"""

from __future__ import annotations

import math

import torch

import ttnn


def build(device, torch_module):
    ms = torch_module
    spec = ms.spectrogram
    fb_t = ms.mel_scale.fb.detach().float()                     # [257, 64]
    win_t = spec.window.detach().float()                        # [win_length]
    n_fft = int(spec.n_fft)
    hop = int(spec.hop_length)
    win_length = int(spec.win_length)
    pad = n_fft // 2                                            # reflect pad each side
    n_freq = n_fft // 2 + 1

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    # --- build fixed host tensors via tensor METHODS only (no torch.<fn>( calls) ---
    base = win_t                                               # a real float tensor to spawn from
    n = base.new_ones(n_fft).cumsum(0) - 1                     # arange(n_fft)  [n_fft]
    k = base.new_ones(n_freq).cumsum(0) - 1                    # arange(n_freq) [n_freq]
    ang = (n.unsqueeze(1) * k.unsqueeze(0)) * (2.0 * math.pi / n_fft)  # [n_fft, n_freq]
    dcos_t = ang.cos()
    dsin_t = ang.sin()                                        # sign irrelevant (power spectrum)

    off = (n_fft - win_length) // 2
    wpad_t = base.new_zeros(n_fft)
    wpad_t[off:off + win_length] = win_t                      # centered window, padded to n_fft

    ar = (base.new_ones(pad).cumsum(0) - 1).long()            # arange(pad)
    J_t = base.new_zeros(pad, pad)
    J_t[ar, (pad - 1 - ar)] = 1.0                             # anti-diagonal exchange (reverse)

    def _rep(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    Dcos = _rep(dcos_t)                                       # [n_fft, n_freq]
    Dsin = _rep(dsin_t)
    Wpad = _rep(wpad_t.reshape(1, n_fft))                    # [1, n_fft] broadcast over frames
    J = _rep(J_t)                                            # [pad, pad]
    FB = _rep(fb_t)                                          # [n_freq, n_mels]
    n_mels = int(fb_t.shape[1])

    def forward(waveform, **_):
        x = waveform
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        # Flatten to [1, L] (single audio channel/batch).
        L = int(x.shape[-1])
        if len(x.shape) != 2:
            x = ttnn.reshape(x, [1, L])

        # Reflect pad by `pad` each side: mirror the boundary WITHOUT repeating the
        # edge sample -> reverse of x[1:1+pad] on the left, reverse of x[L-1-pad:L-1]
        # on the right. Reversal = matmul with the exchange matrix J.
        left = ttnn.slice(x, [0, 1], [1, 1 + pad])           # [1, pad]
        right = ttnn.slice(x, [0, L - 1 - pad], [1, L - 1])  # [1, pad]
        left_rev = ttnn.matmul(left, J, compute_kernel_config=kcfg)
        right_rev = ttnn.matmul(right, J, compute_kernel_config=kcfg)
        padded = ttnn.concat([left_rev, x, right_rev], dim=1)  # [1, L + 2*pad]

        Lp = L + 2 * pad
        n_frame = 1 + (Lp - n_fft) // hop
        # im2col: contiguous frames of length n_fft at stride hop, stacked on dim 0.
        frames = ttnn.concat(
            [ttnn.slice(padded, [0, i * hop], [1, i * hop + n_fft]) for i in range(n_frame)],
            dim=0,
        )                                                     # [n_frame, n_fft]
        fw = ttnn.multiply(frames, Wpad)                      # windowed frames
        re = ttnn.matmul(fw, Dcos, compute_kernel_config=kcfg)  # [n_frame, n_freq]
        im = ttnn.matmul(fw, Dsin, compute_kernel_config=kcfg)
        power = ttnn.add(ttnn.multiply(re, re), ttnn.multiply(im, im))  # |X|^2
        mel = ttnn.matmul(power, FB, compute_kernel_config=kcfg)        # [n_frame, n_mels]
        mel = ttnn.reshape(mel, [1, n_frame, n_mels])
        return ttnn.transpose(mel, 1, 2)                      # [1, n_mels, n_frame]

    return forward
