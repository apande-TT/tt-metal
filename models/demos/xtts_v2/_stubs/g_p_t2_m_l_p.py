# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t2_m_l_p` for coqui/XTTS-v2.

HF submodule: ``gpt.gpt.h.0.mlp`` — a HuggingFace ``GPT2MLP`` (embed=1024,
inner=4096, NewGELU/tanh activation):

    y = c_proj(gelu_new(c_fc(x)))          # c_fc: Conv1D(1024->4096)
                                           # c_proj: Conv1D(4096->1024)

TP=8 scheme (genuine tensor-parallel, math unchanged)
-----------------------------------------------------
Column-then-gather (mirrors the graduated ``g_e_g_l_u`` / ``g_p_t2_block`` MLP):
  * ``c_fc`` (Conv1D 1024->4096) is COLUMN-parallel (``ShardTensorToMesh(dim=1)``
    on the stored [in,out] weight); each chip computes 512 of the 4096 hidden
    features and its tanh-GELU locally. Bias split the same way.
  * ``all_gather(dim=2)`` reassembles the 4096 hidden dim; the REPLICATED
    ``c_proj`` (Conv1D 4096->1024) projects back to model dim.
Only the placement of the c_fc projection changes; the gathered output equals
the single-device golden.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module=None):
    mlp = torch_module

    def _rep(t):
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def _shard(t, dim):
        return ttnn.from_torch(
            t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
        )

    # HF Conv1D stores weight as [in, out]; matmul(x, W) is the projection.
    wt_fc = _shard(mlp.c_fc.weight.detach(), dim=1)                 # [1024, 4096] col-parallel
    b_fc = _shard(mlp.c_fc.bias.detach().reshape(1, 1, -1), dim=2)
    wt_proj = _rep(mlp.c_proj.weight.detach())                     # [4096, 1024] replicated
    b_proj = _rep(mlp.c_proj.bias.detach().reshape(1, 1, -1))

    def forward(x, *_, **__):
        ff = ttnn.add(ttnn.matmul(x, wt_fc), b_fc)                 # [1, T, 512] per chip
        ff = ttnn.gelu(ff, variant=ttnn.GeluVariant.Tanh)
        ff = ttnn.all_gather(ff, dim=2, num_links=1, topology=ttnn.Topology.Linear)
        return ttnn.add(ttnn.matmul(ff, wt_proj), b_proj)          # [1, T, 1024]

    return forward


def g_p_t2_m_l_p(device, torch_module=None):
    return build(device, torch_module)
