# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `learned_position_embeddings` for coqui/XTTS-v2.

HF submodule: ``gpt.mel_pos_embedding`` — an XTTS ``LearnedPositionEmbeddings``
wrapping an ``nn.Embedding(608, 1024)``. Its ``forward(x)`` (non-relative mode,
``relative=False``) is::

    sl = x.shape[1]
    return self.emb(torch.arange(0, sl))       # -> emb.weight[0:sl, :]

i.e. it returns the first ``sl`` rows of the embedding table, where ``sl`` is the
sequence length of the (integer) token tensor ``x``. The token VALUES are unused
in non-relative mode — only ``x.shape[1]`` matters.

Native strategy
---------------
Stage the embedding weight ``[608, 1024]`` onto the mesh (replicated) once, then
the forward is a pure ``ttnn.slice`` of the leading ``sl`` rows. No matmul, no
lookup by value; the result is bit-identical to the torch golden.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    m = torch_module
    W = m.emb.weight.detach()                                   # [num_pos, dim]
    num_pos, dim = int(W.shape[0]), int(W.shape[1])

    emb_w = ttnn.from_torch(
        W.contiguous().to(torch.float32), dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT, device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    def forward(x, **_):
        # Non-relative: return the first sl = x.shape[1] rows of the table.
        sl = int(x.shape[1])
        return ttnn.slice(emb_w, [0, 0], [sl, dim])             # [sl, dim]

    return forward
