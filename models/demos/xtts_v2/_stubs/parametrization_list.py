# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrization_list` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.waveform_decoder.ups.0.parametrizations.weight``
— a ``torch.nn.utils.parametrize.ParametrizationList`` holding a single
``_WeightNorm`` (dim=0). Its ``forward()`` takes NO inputs; it materializes the
weight-normalized ConvTranspose weight from the stored originals::

    g = original0  # [512, 1, 1]   (magnitude, per output-index along dim 0)
    v = original1  # [512, 256, 16] (direction)
    weight = v * g / ||v||          # norm over all dims except dim 0

Native strategy
---------------
Pure elementwise/reduction ttnn on the two stored parameters (staged replicated;
this is a weight-materialization, not a matmul — replication gathers bit-for-bit
to the single-device golden). The forward ignores any positional input the PCC
harness feeds (the module's real forward is nullary). float32 throughout.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    pl = torch_module
    g_t = pl.original0.detach().float()                        # [C, 1, 1]
    v_t = pl.original1.detach().float()                        # [C, *rest]
    C = int(v_t.shape[0])
    rest = int(v_t.numel() // C)
    out_shape = [int(s) for s in v_t.shape]

    def _rep(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    # Flatten to [C, rest] up front; weight-norm (dim=0) reduces over `rest`.
    V = _rep(v_t.reshape(C, rest))                             # [C, rest]
    G = _rep(g_t.reshape(C, 1))                                # [C, 1]

    def forward(*_a, **_k):
        sq = ttnn.multiply(V, V)
        ss = ttnn.sum(sq, dim=1, keepdim=True)                 # [C, 1]  ||v||^2 per row
        inv = ttnn.rsqrt(ss)                                   # 1 / ||v||
        scale = ttnn.multiply(G, inv)                          # g / ||v||   [C, 1]
        w = ttnn.multiply(V, scale)                            # broadcast -> [C, rest]
        return ttnn.reshape(w, out_shape)                      # [C, *rest]

    return forward
