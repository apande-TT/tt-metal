# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `adaptive_avg_pool2d` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder.layer1.0.se.avg_pool`` — an
``nn.AdaptiveAvgPool2d(output_size=1)`` inside the ResNet speaker encoder's
squeeze-and-excitation block. Given ``x`` of shape ``[N, C, H, W]`` it produces
the global average pool ``[N, C, 1, 1]`` (the per-channel mean over the H and W
spatial axes).

TP=8 scheme
-----------
This op has NO trainable weights: it is a pure per-channel spatial reduction.
Per the tensor-parallel principles it is therefore a *replicate-only* role —
there is no matmul weight to column/row-split. The harness replicates the input
across the 8-chip mesh; every chip runs the identical reduction and produces the
identical ``[N, C, 1, 1]`` result, so the gathered (concat-then-slice) output is
bit-for-bit the single-device golden. No collective is needed because the math
is per-channel independent and every shard already holds the full channel set.
Placement changes, math does not.
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    """Return the native TTNN forward for the global average pool.

    ``torch_module`` is the ``nn.AdaptiveAvgPool2d`` reference; its output size
    is 1x1 (global pool), so we reduce the full H and W extent. There are no
    parameters to move onto the mesh.
    """

    def forward(x, **kwargs):
        # AdaptiveAvgPool2d(output_size=1) is a global average pool over the
        # two trailing spatial dims, for either a 4D (N, C, H, W) or a 3D
        # (C, H, W) input. Reduce the last two dims (negative indices adapt to
        # whichever rank the harness feeds) with keepdim to preserve the
        # golden's rank.
        return ttnn.mean(x, dim=[-2, -1], keepdim=True)

    return forward
