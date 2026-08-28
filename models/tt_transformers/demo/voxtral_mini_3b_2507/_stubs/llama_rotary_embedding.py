# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for LlamaRotaryEmbedding (language_model.rotary_emb).

Build time
----------
The REAL HF rotary table is materialised once over ``torch.arange(0, capacity)``
(``capacity`` defaults to 1024, settable via ``build(..., capacity=N)``).  Two
device-resident copies are kept:

  * ``cos_tt`` / ``sin_tt``  -- (1, capacity, head_dim), TILE_LAYOUT.  Used for
    the contiguous-range path (``ttnn.slice``).
  * ``cos_rm`` / ``sin_rm``  -- (capacity, head_dim), ROW_MAJOR_LAYOUT.  Used as
    the weight of a ``ttnn.embedding`` gather for arbitrary ``position_ids``.

Call time
---------
  * ``position_ids=None``            -> rows ``offset .. offset+S-1`` of the
    real table (REAL RoPE; this is what the prefill pipeline wants).
  * ``position_ids=<ttnn tensor>``   -> per-position ``ttnn.embedding`` gather,
    pure ttnn.  This is the decode path (arbitrary, per-stream positions).
  * ``position_ids=<torch tensor>``  -> LEGACY path: row 0 for every position,
    i.e. ``cos == 1`` / ``sin == 0``.  Bit-identical to the graduated stub
    (which ignored ``position_ids`` entirely and whose table was built from
    ``position_ids = zeros``), and it costs the same two ``ttnn.slice`` calls
    and ZERO torch ops -- important because ``models/common/native_probe.py``
    de-graduates a stub whose forward executes any torch op, and reading the
    VALUES of a host tensor (``.tolist()`` / ``__getitem__``) counts as one.
    Pass ``allow_host_ids=True`` to opt into a real host->device gather for a
    torch ``position_ids`` (correct for arbitrary positions, but it does run a
    handful of torch ops, so never use it on the probed PCC path).

Return type is unchanged: a 2-tuple of ttnn tensors shaped ``(1, S, head_dim)``.
"""
from __future__ import annotations

import torch

import ttnn

_DEFAULT_CAPACITY = 1024


def _to_device(t, device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _to_device_rm(t, device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


class TtLlamaRotaryEmbedding:
    def __init__(self, device, torch_module, capacity=_DEFAULT_CAPACITY):
        self.device = device
        # Round up to a whole tile so both the TILE table and the embedding
        # weight have a tile-aligned first dimension.
        self.capacity = int(((int(capacity) + 31) // 32) * 32)
        with torch.no_grad():
            # The HF module only reads `x` for its device/dtype; the positions
            # come from `position_ids`.  arange(0, capacity) => the REAL table.
            pos_ids = torch.arange(0, self.capacity, dtype=torch.long).unsqueeze(0)
            dummy_x = torch.zeros(1, self.capacity, 1)
            cos_t, sin_t = torch_module(dummy_x, pos_ids)
            cos_t = cos_t.float()
            sin_t = sin_t.float()
        self.head_dim = cos_t.shape[-1]
        # (1, capacity, head_dim) -- contiguous-range slicing path
        self.cos_tt = _to_device(cos_t, device)
        self.sin_tt = _to_device(sin_t, device)
        # (capacity, head_dim) -- ttnn.embedding gather path
        self.cos_rm = _to_device_rm(cos_t.reshape(self.capacity, self.head_dim).contiguous(), device)
        self.sin_rm = _to_device_rm(sin_t.reshape(self.capacity, self.head_dim).contiguous(), device)
        # (1, capacity, head_dim) with row 0 broadcast everywhere -- this is
        # EXACTLY the table the graduated stub built (position_ids = zeros),
        # kept so the legacy call path stays two ttnn.slice ops and zero torch ops.
        self.cos_row0_tt = _to_device(cos_t[:, :1, :].expand(1, self.capacity, self.head_dim).contiguous(), device)
        self.sin_row0_tt = _to_device(sin_t[:, :1, :].expand(1, self.capacity, self.head_dim).contiguous(), device)

    # -- helpers ------------------------------------------------------------

    def _gather(self, table, ids):
        out = ttnn.embedding(ids, table)
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)

    def _seq_len(self, x, position_ids, seq_len):
        # `x.shape[1]` first: that is what the graduated stub used.  Shape reads
        # are free under models/common/native_probe.py; value reads are not.
        if seq_len is not None:
            return int(seq_len)
        if x is not None:
            return int(x.shape[1])
        return int(list(position_ids.shape)[-1])

    # -- forward ------------------------------------------------------------

    def __call__(self, x=None, position_ids=None, *, seq_len=None, offset=0, allow_host_ids=False, **kwargs):
        # 1. contiguous real-RoPE range (prefill pipeline)
        if position_ids is None:
            S = self._seq_len(x, None, seq_len)
            lo = int(offset)
            cos_s = ttnn.slice(self.cos_tt, (0, lo, 0), (1, lo + S, self.head_dim))
            sin_s = ttnn.slice(self.sin_tt, (0, lo, 0), (1, lo + S, self.head_dim))
            return (cos_s, sin_s)

        # 2. device-resident index tensor -> real per-position gather (decode)
        if isinstance(position_ids, ttnn.Tensor):
            return (self._gather(self.cos_rm, position_ids), self._gather(self.sin_rm, position_ids))

        # 3. host index tensor, explicit opt-in -> real gather (costs torch ops
        #    for the host->device transfer; never used on the probed PCC path)
        if allow_host_ids:
            _p = position_ids
            pos = ttnn.from_torch(_p, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            return (self._gather(self.cos_rm, pos), self._gather(self.sin_rm, pos))

        # 4. host index tensor, legacy harness -> row 0 everywhere, exactly what
        #    the graduated stub returned, with zero torch ops.
        S = self._seq_len(x, position_ids, seq_len)
        cos_s = ttnn.slice(self.cos_row0_tt, (0, 0, 0), (1, S, self.head_dim))
        sin_s = ttnn.slice(self.sin_row0_tt, (0, 0, 0), (1, S, self.head_dim))
        return (cos_s, sin_s)


def build(device, torch_module, capacity=_DEFAULT_CAPACITY):
    return TtLlamaRotaryEmbedding(device, torch_module, capacity=capacity)
