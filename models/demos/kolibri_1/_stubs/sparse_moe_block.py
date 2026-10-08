# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN Kolibri-1 sparse MoE block (`sparse_moe_block`, `model.layers.0.mlp`), expert-parallel.

The implementation lives in `_kolibri_moe.py` so the decoder layer runs the identical block: fp32
router with the selection-only bias, top-6 of 384 experts weighted by the unbiased sigmoid, plus the
ungated shared expert. Each TP chip owns 384/TP experts and evaluates them as two wide matmuls; the
shared expert is split gate/up column-parallel, down row-parallel; one all_reduce combines both
partial sums. See that module's docstring for the scheme.
"""
from __future__ import annotations

from models.demos.kolibri_1._stubs._kolibri_moe import TtKolibri1MoE


class TtKolibri1SparseMoeBlock(TtKolibri1MoE):
    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, hidden_states, **kwargs):
        return super().__call__(hidden_states)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtKolibri1SparseMoeBlock.build(device, torch_module)


# Backward-compatible slug shim.
def sparse_moe_block(device, torch_module=None):
    return TtKolibri1SparseMoeBlock.build(device, torch_module)
