# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mel_scale` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder.torch_spec.1.mel_scale`` — a
``torchaudio.transforms.MelScale`` with a fixed mel filterbank buffer
``fb`` of shape ``[n_stft=257, n_mels=64]``. Its ``forward(specgram)`` is::

    mel = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

mapping a linear-frequency spectrogram ``[.., 257, time]`` to a mel spectrogram
``[.., 64, time]``.

Native strategy
---------------
The filterbank is a fixed linear map, so this is a single matmul. Stage ``fb``
onto the mesh (replicated) once; the forward transposes, matmuls, and transposes
back. float32 weights + fp32 accumulation keep PCC tight.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    m = torch_module

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    fb = ttnn.from_torch(
        m.fb.detach().contiguous().to(torch.float32), dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT, device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    def forward(specgram, **_):
        if isinstance(specgram, ttnn.Tensor) and specgram.get_dtype() != ttnn.float32:
            specgram = ttnn.typecast(specgram, ttnn.float32)
        xt = ttnn.transpose(specgram, -1, -2)                   # [.., time, 257]
        y = ttnn.matmul(xt, fb, compute_kernel_config=kcfg)     # [.., time, 64]
        return ttnn.transpose(y, -1, -2)                        # [.., 64, time]

    return forward
