# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, expert-parallel port of `Glm5NextTextMoE` (GLM-5.3-Flash sparse MLP).

Reference:
    logits, w, idx = gate(x)                       (Glm5NextTextTopkRouter)
    out = experts(x, idx, w) + shared_experts(x)   (Glm5NextTextExperts + Glm5NextTextMLP)

Composed of the graduated component ports, one implementation each:
  * router   -> _stubs/text_topk_router.py  (expert-axis column-parallel, ~16-bit logits, exact top-k)
  * experts  -> _stubs/text_experts.py      (EXPERT-parallel: E/N whole experts per chip, all-reduce)
  * shared   -> _stubs/text_m_l_p.py        (column/row parallel + all-reduce)
Rows are processed in chunks of ROW_CHUNK tokens: the router's exact rank is an [T, E, E] comparison.
"""
from __future__ import annotations

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _Ctx
from models.demos.glm_5_3_flash._stubs.text_experts import TtTextExperts
from models.demos.glm_5_3_flash._stubs.text_m_l_p import TtTextMLP
from models.demos.glm_5_3_flash._stubs.text_topk_router import TtTextTopkRouter

ROW_CHUNK = 512


class TtTextMoE:
    def __init__(self, device, m):
        self.ctx = c = _Ctx(device)
        self.gate = TtTextTopkRouter(device, m.gate)
        self.experts = TtTextExperts(device, m.experts)
        if self.experts.E != self.gate.E:
            raise ValueError("router / experts expert-count mismatch")
        self.shared = TtTextMLP(c, m.shared_experts)
        self.D = self.gate.D

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _routed(self, x, T):
        _, w, idx = self.gate(x)  # [T,k] weights (row-major), [T,k] int32 expert ids
        w = ttnn.to_layout(ttnn.reshape(w, [1, 1, T, w.shape[-1]]), ttnn.TILE_LAYOUT)
        idx = ttnn.to_layout(ttnn.reshape(idx, [1, 1, T, idx.shape[-1]]), ttnn.TILE_LAYOUT)
        return self.experts(x, top_k_index=idx, top_k_weights=w)  # [1,1,T,D] fp32, all-reduced

    def __call__(self, hidden_states, *args, **kwargs):
        x = hidden_states
        in_shape = list(x.shape)
        D = in_shape[-1]
        T = 1
        for d in in_shape[:-1]:
            T *= d
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, T, D]), ttnn.float32)
        if T <= ROW_CHUNK:
            routed = self._routed(x, T)
        else:
            parts = []
            for r0 in range(0, T, ROW_CHUNK):
                r1 = min(T, r0 + ROW_CHUNK)
                parts.append(self._routed(ttnn.slice(x, [0, 0, r0, 0], [1, 1, r1, D]), r1 - r0))
            routed = ttnn.concat(parts, dim=2)
        shared = ttnn.typecast(self.shared(x), ttnn.float32)
        out = ttnn.add(ttnn.typecast(routed, ttnn.float32), shared)
        return ttnn.reshape(out, in_shape)


def build(device, torch_module=None):
    return TtTextMoE.build(device, torch_module)


def text_mo_e(device, torch_module=None):
    return TtTextMoE.build(device, torch_module)
