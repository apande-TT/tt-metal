# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of Qwen3RotaryEmbedding.

Component `rotary_embedding` of `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
(`model.rotary_emb`). Returns the `(cos, sin)` pair the decoder layers apply
to q and k, shaped (batch, seq, head_dim=128), for rope_type "default" with
rope_theta = 1e6 and attention_scaling = 1.0:

    inv_freq[j] = theta ** (-2j / head_dim),        j = 0 .. head_dim/2 - 1
    emb[p]      = cat(p * inv_freq, p * inv_freq)
    cos[p], sin[p] = cos(emb[p]), sin(emb[p])

Implementation: the table depends only on the config, never on the
activations, so both halves are materialised ONCE over
`max_position_embeddings` at build time and the forward is a pure device-side
row gather -- `ttnn.embedding` indexed by position_ids. That keeps the
forward free of host work (which the native probe counts) and keeps the
angles exact instead of pushing arguments up to tens of radians through a
device trig approximation.

No tensor-parallel split: this is a lookup table, which the TP principles
keep REPLICATED (every chip needs the same cos/sin for its own head slice --
head_dim is not an axis any scheme shards). Under a mesh it is replicated.

position_ids are indices: they are staged as uint32 / ROW_MAJOR, never cast
to bfloat16, which cannot represent ids above 256 exactly.
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


class TtRotaryEmbedding:
    """Native ttnn RoPE table; replicated on every chip of a mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)

        cfg = torch_module.config
        self.head_dim = int(getattr(cfg, "head_dim", 0) or (cfg.hidden_size // cfg.num_attention_heads))
        rope = getattr(cfg, "rope_parameters", None) or {}
        theta = float(rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0)))
        self.max_positions = int(getattr(cfg, "max_position_embeddings", 0) or 40960)
        scaling = float(getattr(torch_module, "attention_scaling", 1.0) or 1.0)

        # Prefer the reference's own inv_freq buffer when it is present: it
        # already carries whatever rope_type the config selected.
        inv_freq = getattr(torch_module, "inv_freq", None)
        if inv_freq is None:
            inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim))
        inv_freq = inv_freq.detach().float().reshape(-1)

        positions = torch.arange(self.max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (max_pos, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_pos, head_dim)
        self.cos_table = self._table(emb.cos() * scaling)
        self.sin_table = self._table(emb.sin() * scaling)

    def _table(self, t):
        return ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _ids(self, position_ids):
        if torch.is_tensor(position_ids):
            return ttnn.from_torch(
                position_ids.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return position_ids

    def __call__(self, *args, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        kwargs.pop("x", None)
        # forward(x, position_ids): x only carries dtype/device for the
        # reference, so the positions are the only argument that matters here.
        for a in args:
            if a is None:
                continue
            if position_ids is None and getattr(a, "shape", None) is not None and len(a.shape) <= 2:
                position_ids = a
        if position_ids is None and args:
            position_ids = args[-1]
        if position_ids is None:
            raise ValueError("rotary_embedding stub: no position_ids supplied")

        ids = self._ids(position_ids)
        cos = ttnn.embedding(ids, self.cos_table, layout=ttnn.ROW_MAJOR_LAYOUT)
        sin = ttnn.embedding(ids, self.sin_table, layout=ttnn.ROW_MAJOR_LAYOUT)
        return cos, sin

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("rotary_embedding stub needs the torch reference module for its config")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtRotaryEmbedding.build(device, torch_module)


def rotary_embedding(device, torch_module=None):
    return TtRotaryEmbedding.build(device, torch_module)
