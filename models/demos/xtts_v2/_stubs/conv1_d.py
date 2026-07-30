# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `conv1_d` for coqui/XTTS-v2.

HF submodule: ``gpt.gpt.h.0.attn.c_attn`` — a HuggingFace GPT2 ``Conv1D``, which
despite the name is a linear projection: ``y = x @ weight + bias`` with
``weight:[in=1024, out=3072]`` (the fused q/k/v projection). Input ``[b, T, 1024]``
-> output ``[b, T, 3072]``.

TP=8 scheme (genuine column-parallel)
-------------------------------------
``c_attn``'s output feeds the per-head attention split, so it is COLUMN-parallel:
the 3072 output features are split across the 8 chips (``ShardTensorToMesh(dim=1)``
on ``weight`` and ``dim=0`` on ``bias``). Chip ``i`` computes its 384-wide output
slice from the replicated input, then ``all_gather(dim=-1)`` reassembles the full
``[b, T, 3072]`` on every chip. The gathered output equals the single-device
golden; only the placement of the output columns differs.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    W = torch_module.weight.detach()   # [in, out] = [1024, 3072]
    b = torch_module.bias.detach()     # [out] = [3072]

    wt = ttnn.from_torch(
        W.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),   # split OUTPUT cols
    )
    bt = ttnn.from_torch(
        b.reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2),   # matching output cols
    )

    def forward(x, **_):
        # Column-parallel: chip i produces output cols [i*384:(i+1)*384].
        y = ttnn.matmul(x, wt)            # [1, T, 3072/8]
        y = ttnn.add(y, bt)
        # Reassemble the full output feature axis across the mesh.
        return ttnn.all_gather(y, dim=-1, num_links=1, topology=ttnn.Topology.Linear)

    return forward
