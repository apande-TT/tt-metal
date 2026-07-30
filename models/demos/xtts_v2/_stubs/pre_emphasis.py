# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `pre_emphasis` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.speaker_encoder.torch_spec.0`` — a ``PreEmphasis``
first-order high-pass filter. ``forward(x)`` (x is ``[batch, time]``)::

    x = F.pad(x.unsqueeze(1), (1, 0), "reflect")   # reflect-pad 1 sample on left
    return F.conv1d(x, self.filter).squeeze(1)     # filter = [-coef, 1.0]

which is ``y[t] = x[t] - coef * x[t-1]`` with the leading sample using the
reflected neighbour (``x[1]``). coef = 0.97.

Native strategy
---------------
Pure shift-and-subtract on the time axis: reflect-pad the left edge by
concatenating ``x[:, 1:2]``, then form the two length-L views (padded[1:] and
padded[:-1]) by slicing and combine with the two fixed filter taps. No matmul,
no weights to split — REPLICATED across the mesh, gathering bit-for-bit to the
single-device golden. float32 throughout.
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    pe = torch_module
    f = pe.filter.detach().flatten().tolist()                  # [w0, w1] = [-coef, 1.0]
    w0 = float(f[0])
    w1 = float(f[1])

    def forward(x, **_):
        if isinstance(x, ttnn.Tensor) and x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        L = int(x.shape[-1])
        # reflect pad 1 on the left: prepend x[1] (reflection excludes the border).
        left = ttnn.slice(x, [0, 1], [1, 2])                   # [1, 1] = x[1]
        xp = ttnn.concat([left, x], dim=1)                     # [1, L+1]
        a = ttnn.slice(xp, [0, 1], [1, L + 1])                 # padded[1:]  == x[t]
        b = ttnn.slice(xp, [0, 0], [1, L])                     # padded[:-1] == x[t-1]/reflect
        # conv with filter [w0, w1]: out = w0*padded[t] + w1*padded[t+1] = w0*b + w1*a
        return ttnn.add(ttnn.multiply(b, w0), ttnn.multiply(a, w1))

    return forward
