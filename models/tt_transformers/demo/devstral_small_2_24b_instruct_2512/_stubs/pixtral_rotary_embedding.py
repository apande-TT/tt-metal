# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `pixtral_rotary_embedding` (PixtralRotaryEmbedding,
`vision_tower.patch_positional_embedding`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    cos, sin = cos(inv_freq[position_ids]), sin(inv_freq[position_ids])

`inv_freq` is Pixtral's fixed 2D table: one row of (h-freqs | w-freqs) per patch position of the
max_patches_per_side^2 grid. The angles reach ~1e2 rad, so cos/sin are evaluated once at build in
fp32 (as tt_transformers' rotary setup does) — neither bf16 angles nor an on-device cos of large
arguments would hold precision — and the resulting values in [-1, 1] are stored as bf16 tables
(ttnn.embedding's weight dtype) and gathered on device. Replicated on a mesh.
"""
from __future__ import annotations

import torch

import ttnn


class TtPixtralRotaryEmbedding:
    def __init__(self, device, torch_module, interleaved=False) -> None:
        """interleaved: emit the tables in interleaved-pair head_dim order (dims 2j and 2j+1 <- HF's j and
        j + head_dim/2, which share a frequency), the order of a TtPixtralAttention built with META_ROPE. The values
        are the same, only their column order differs."""
        self.device = device
        self._repl = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        inv_freq = torch_module.inv_freq.detach().float()  # (positions, head_dim)
        self.head_dim = hd = inv_freq.shape[-1]
        if interleaved:
            inv_freq = inv_freq[:, torch.arange(hd).reshape(2, hd // 2).T.reshape(-1)]

        def _table(t):
            return ttnn.from_torch(
                t.to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=self._repl,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.cos_table = _table(inv_freq.cos())
        self.sin_table = _table(inv_freq.sin())

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, position_ids=None, **kwargs):
        patch_pos = position_ids
        out_shape = tuple(patch_pos.shape) + (self.head_dim,)
        if isinstance(patch_pos, torch.Tensor):
            patch_pos = ttnn.from_torch(
                patch_pos.reshape(1, -1).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=self._repl,
            )
        cos = ttnn.embedding(patch_pos, self.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(patch_pos, self.sin_table, layout=ttnn.TILE_LAYOUT)
        out_dtype = x.dtype if isinstance(x, ttnn.Tensor) else ttnn.bfloat16
        if out_dtype != ttnn.bfloat16:
            cos, sin = ttnn.typecast(cos, out_dtype), ttnn.typecast(sin, out_dtype)
        return ttnn.reshape(cos, out_shape), ttnn.reshape(sin, out_shape)


def build(device, torch_module=None):
    return TtPixtralRotaryEmbedding.build(device, torch_module)


def pixtral_rotary_embedding(device, torch_module=None):
    return TtPixtralRotaryEmbedding.build(device, torch_module)
