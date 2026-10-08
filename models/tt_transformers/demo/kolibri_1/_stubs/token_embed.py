# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN Kolibri-1 token embedding (`token_embed`, `model.embed_tokens`).

The canonical `Embedding` (models/tt_transformers/tt/embedding.py) is built from `ModelArgs`, which
cannot load the `kolibri1` config (not a transformers model type), so the lookup is a direct
`ttnn.embedding` over the shipped bf16 table (128000 x 2560):

    h = embed_tokens.weight[input_ids]

The table is replicated on every chip: a lookup is a gather, not a matmul, so there is nothing to split.
Token ids come in as uint32 row-major (a torch tensor is uploaded the same way).
"""
from __future__ import annotations

import torch

import ttnn


class TtKolibri1Embedding:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self._mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        self.weight = ttnn.from_torch(
            torch_module.weight.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=self._mapper,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, input_ids, **kwargs):
        if isinstance(input_ids, torch.Tensor):
            input_ids = ttnn.from_torch(
                input_ids.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=self._mapper,
            )
        return ttnn.embedding(input_ids, self.weight, layout=ttnn.TILE_LAYOUT)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtKolibri1Embedding.build(device, torch_module)


# Backward-compatible slug shim.
def token_embed(device, torch_module=None):
    return TtKolibri1Embedding.build(device, torch_module)
