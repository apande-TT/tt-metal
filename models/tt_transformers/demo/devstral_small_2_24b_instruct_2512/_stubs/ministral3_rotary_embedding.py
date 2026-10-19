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
    def __init__(self, device, torch_module, interleaved=False) -> None:
        """interleaved: emit the tables in interleaved-pair head_dim order (dims 2j and 2j+1 share frequency j, the
        order ttnn's fused rotary_embedding_llama rotates) instead of HF's half-split order (j and j + head_dim/2).
        The values are the same, only their column order differs (see TtMinistral3Attention meta_rope)."""
        self.device = device
        self.scaling = float(torch_module.attention_scaling)
        inv_freq = torch_module.inv_freq.detach().float()
        self.head_dim = 2 * inv_freq.shape[0]
        self._repl = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        freq = inv_freq.repeat_interleave(2) if interleaved else torch.concat((inv_freq, inv_freq))
        self.inv_freq = ttnn.from_torch(
            freq.reshape(1, 1, 1, self.head_dim),
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

    def table(self, positions, dtype=ttnn.bfloat16):
        """(cos, sin) of every position 0 .. positions - 1 as (1, 1, positions, head_dim) row-major tables, one row a
        position. Computed by the very ops __call__ runs on a decode step's position column -- the same fp32
        position x inv_freq product, cos / sin, scale and cast, element by element -- so row p is bit-identical to the
        row a decode step at position p derives. The positions run down one dense (1, 1, positions, 1) column (32
        positions a tile), not one tile row each as a step's (B, 1, 1, 1) column holds them: 32x fewer tiles to build."""
        pos = ttnn.from_torch(
            torch.arange(positions, dtype=torch.float32).reshape(1, 1, positions, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self._repl,
        )
        freqs = ttnn.multiply(pos, self.inv_freq)
        ttnn.deallocate(pos)
        out = []
        for fn in (ttnn.cos, ttnn.sin):
            t = fn(freqs)
            v = ttnn.multiply(t, self.scaling)
            ttnn.deallocate(t)
            c = ttnn.typecast(v, dtype)
            ttnn.deallocate(v)
            out.append(ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT))
            ttnn.deallocate(c)
        ttnn.deallocate(freqs)
        return tuple(out)


def build(device, torch_module=None):
    return TtMinistral3RotaryEmbedding.build(device, torch_module)


def ministral3_rotary_embedding(device, torch_module=None):
    return TtMinistral3RotaryEmbedding.build(device, torch_module)
