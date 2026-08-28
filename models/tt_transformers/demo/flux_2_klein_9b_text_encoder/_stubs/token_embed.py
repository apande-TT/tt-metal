# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of the token embedding table.

Component `token_embed` of `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
(`model.embed_tokens`): `nn.Embedding(151936, 4096)`, a plain row gather ->
`ttnn.embedding`.

No tensor-parallel split: the TP principles keep lookup tables REPLICATED --
an embedding has no reduction to split and every chip needs the full hidden
row for the residual stream it feeds. Under a mesh the table is replicated
and each chip gathers the same rows.

Ids stay uint32 / ROW_MAJOR end to end. bfloat16 has 8 mantissa bits, so it
cannot hold ids above 256 exactly (a vocab of 151936 needs 18) -- casting
them rounds to a neighbouring row and silently returns the wrong embedding.
"""
from __future__ import annotations

import torch

import ttnn


def _is_mesh(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


class TtTokenEmbed:
    """Native ttnn embedding lookup; table replicated on every chip of a mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)

        w = torch_module.state_dict()["weight"].detach()
        self.num_embeddings, self.embedding_dim = int(w.shape[0]), int(w.shape[1])
        self.weight = ttnn.from_torch(
            w.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, *args, **kwargs):
        ids = kwargs.pop("input", None)
        for name in ("input_ids", "x", "inputs", "hidden_states"):
            if ids is None:
                ids = kwargs.pop(name, None)
        if ids is None:
            for a in args:
                if a is not None:
                    ids = a
                    break
        if ids is None:
            raise ValueError("token_embed stub: no input ids supplied")
        if torch.is_tensor(ids):
            ids = ttnn.from_torch(
                ids.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.embedding(ids, self.weight, layout=ttnn.TILE_LAYOUT)

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("token_embed stub needs the torch reference module to source its table")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtTokenEmbed.build(device, torch_module)


def token_embed(device, torch_module=None):
    return TtTokenEmbed.build(device, torch_module)
