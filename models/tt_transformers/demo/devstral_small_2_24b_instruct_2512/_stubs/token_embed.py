# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `token_embed` (nn.Embedding, `language_model.embed_tokens`, 131072 x 5120)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    out = embed_tokens.weight[input_ids]

A plain row gather with ttnn.embedding; Ministral3 applies no embedding scale. The table is replicated
on a mesh (embeddings are a lookup, not a matmul — they are not tensor-parallel split).
"""
from __future__ import annotations

import torch

import ttnn


class TtTokenEmbed:
    # On a mesh, a second copy of the table split on the hidden dim (D/TP columns per device): prefill's
    # hidden-sharded residual gathers its own slice of each row (shard()), instead of full-width rows that a
    # mesh_partition then cuts down (the gather wrote 4x the bytes the layers read: 773 + 182 us at 32 x 416 rows).
    SHARDED_TABLE = True

    def __init__(self, device, torch_module) -> None:
        self.device = device
        mesh = isinstance(device, ttnn.MeshDevice)
        self._repl = ttnn.ReplicateTensorToMesh(device) if mesh else None
        # an owned copy: from_torch crawls on a mmapped checkpoint view
        w = torch_module.weight.detach().to(torch.bfloat16).clone()
        put = lambda mapper: ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.weight = put(self._repl)
        tp = int(device.get_num_devices()) if mesh else 1
        self.weight_shard = None
        if tp > 1 and self.SHARDED_TABLE and w.shape[-1] % (tp * ttnn.TILE_SIZE) == 0:
            self.weight_shard = put(ttnn.ShardTensorToMesh(device, dim=-1))
        self.dim = torch_module.embedding_dim
        self.shard_dim = self.dim // tp

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, input_ids, *args, **kwargs):
        return self._gather(input_ids, self.weight, self.dim)

    def shard(self, input_ids, memory_config=None):
        """This device's hidden slice of the embeddings, (..., D / TP) (needs weight_shard)."""
        return self._gather(input_ids, self.weight_shard, self.shard_dim, memory_config)

    def _gather(self, input_ids, weight, dim, memory_config=None):
        tok = input_ids
        if isinstance(tok, torch.Tensor):
            tok = ttnn.from_torch(
                tok.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=self._repl,
            )
        shape = tuple(tok.shape)
        tok = ttnn.reshape(tok, (1, shape[-1]) if len(shape) == 1 else (-1, shape[-1]))
        out = ttnn.embedding(tok, weight, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
        return ttnn.reshape(out, shape + (dim,))


def build(device, torch_module=None):
    return TtTokenEmbed.build(device, torch_module)


def token_embed(device, torch_module=None):
    return TtTokenEmbed.build(device, torch_module)
