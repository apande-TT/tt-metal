# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_e_g_l_u` for coqui/XTTS-v2.

HF submodule: ``gpt.conditioning_perceiver.layers.0.1.1`` — a ``GEGLU`` gate:

    x, gates = x.chunk(2, dim=-1)
    return x * gelu(gates)

It has no weights (a pure elementwise gate that halves the last dim), so under
TP it is a replicate-only role: the harness replicates the input across the
mesh, every chip computes the identical gate, and the gathered output equals
the single-device golden.
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    def forward(x, **_):
        shape = list(x.shape)
        rank = len(shape)
        half = shape[-1] // 2

        starts = [0] * rank
        ends_a = list(shape)
        ends_a[-1] = half
        a = ttnn.slice(x, starts, ends_a)          # [..., :half]

        starts_g = [0] * rank
        starts_g[-1] = half
        g = ttnn.slice(x, starts_g, list(shape))   # [..., half:]

        return ttnn.multiply(a, ttnn.gelu(g))

    return forward
