# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `ministral3_rotary_embedding` (Ministral3RotaryEmbedding, `language_model.rotary_emb`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    freqs    = position_ids (x) inv_freq                    (yarn inv_freq from the checkpoint)
    cos, sin = cos(cat(freqs, freqs)) * attention_scaling, sin(...) * attention_scaling

The canonical tt_transformers RotaryEmbedding is a host-side torch module, so it is not used. The
outer product is an fp32 element-wise broadcast multiply (not a matmul) so the angles keep fp32
precision. Replicated on a mesh — rotary tables are never sharded.
"""
from __future__ import annotations

import torch

import ttnn


class TtMinistral3RotaryEmbedding:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.scaling = float(torch_module.attention_scaling)
        inv_freq = torch_module.inv_freq.detach().float()
        self.head_dim = 2 * inv_freq.shape[0]
        self._repl = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        self.inv_freq = ttnn.from_torch(
            torch.concat((inv_freq, inv_freq)).reshape(1, 1, 1, self.head_dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=self._repl,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, position_ids=None, **kwargs):
        pos = position_ids
        if isinstance(pos, torch.Tensor):
            b, s = pos.shape
            pos = ttnn.from_torch(
                pos.to(torch.float32).reshape(b, 1, s, 1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=self._repl,
            )
        b, _, s, _ = pos.shape
        freqs = ttnn.multiply(pos, self.inv_freq)  # (B, 1, S, head_dim), fp32
        cos = ttnn.multiply(ttnn.cos(freqs), self.scaling)
        sin = ttnn.multiply(ttnn.sin(freqs), self.scaling)
        out_dtype = x.dtype if isinstance(x, ttnn.Tensor) else ttnn.bfloat16
        cos = ttnn.reshape(ttnn.typecast(cos, out_dtype), (b, s, self.head_dim))
        sin = ttnn.reshape(ttnn.typecast(sin, out_dtype), (b, s, self.head_dim))
        return cos, sin


def build(device, torch_module=None):
    return TtMinistral3RotaryEmbedding.build(device, torch_module)


def ministral3_rotary_embedding(device, torch_module=None):
    return TtMinistral3RotaryEmbedding.build(device, torch_module)
