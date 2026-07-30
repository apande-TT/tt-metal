# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `dropout1d` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_dropout`` — an ``nn.Dropout1d(p=0.1)``. At
inference (``eval`` mode, which the PCC harness uses) dropout is the identity
map, so the native forward returns its input unchanged. There are no weights,
so this is a replicate-only role under TP.
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    def forward(x, **_):
        # Dropout is identity in eval mode; pass the activation through.
        # ttnn.clone materialises an owned ttnn tensor (native, no torch).
        return ttnn.clone(x)

    return forward
