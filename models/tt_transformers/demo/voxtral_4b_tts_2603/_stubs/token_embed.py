# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `token_embed` -- the text backbone's token table (`model.embed_tokens`).

A 131072 x 3072 `nn.Embedding`, tied to `lm_head`. In the checkpoint it is
`mm_audio_embeddings.tok_embeddings` (402.65 M of the 430.57 M under `mm_audio_embeddings`; the
remaining 27.92 M is the separate audio codebook table ported in
`_stubs/multi_vocab_embeddings.py`).

`ttnn.embedding` requires the table in bfloat16 (`embedding_device_operation.cpp:36`) and takes the
ids as an integer tensor; the harness marshals integer args as uint32 ROW_MAJOR, which is exactly
what the op wants.
"""

from __future__ import annotations

import torch

import ttnn


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if layout == ttnn.TILE_LAYOUT and t.numel() <= 262144:
        # A small tensor (a gamma, a bias, a layer scale) tilizes on the HOST: tilized on the device it is
        # a whole single-core op (~80 us for a 1 x 1024 float32 row) at load time.
        mapper = ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None
        return ttnn.to_device(ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper), device)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def build(device, torch_module):
    table = _from_torch(torch_module.weight.detach().contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT)

    def token_embed(input_ids, **kwargs):
        return ttnn.embedding(input_ids, table, layout=ttnn.TILE_LAYOUT)

    return token_embed
