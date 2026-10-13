# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `pixtral_m_l_p` (PixtralMLP,
`vision_tower.transformer.layers.0.feed_forward`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    out = down(silu(gate(x)) * up(x))            (1024 -> 4096 -> 1024)

Same SwiGLU structure as Ministral3MLP, so it reuses that tensor-parallel port: gate/up column-parallel
on the intermediate dim, down row-parallel then all_reduce.
"""
from __future__ import annotations

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_m_l_p import TtMinistral3MLP


class TtPixtralMLP(TtMinistral3MLP):
    # The vision residual stream is fp32 (see attention.py); its down projection lands there unrounded.
    OUT_DTYPE = ttnn.float32
    DECODE_WEIGHTS = False  # the vision tower never decodes

    def __init__(self, device, torch_module) -> None:
        act = type(torch_module.act_fn).__name__
        assert act in ("SiLU", "SiLUActivation"), f"expected silu, got {act}"
        super().__init__(device, torch_module)


def build(device, torch_module=None):
    return TtPixtralMLP.build(device, torch_module)


def pixtral_m_l_p(device, torch_module=None):
    return TtPixtralMLP.build(device, torch_module)
