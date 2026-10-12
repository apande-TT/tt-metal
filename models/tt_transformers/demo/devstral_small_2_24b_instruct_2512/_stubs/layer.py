# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `layer` (PixtralAttentionLayer, `vision_tower.transformer.layers.0`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

The same module as the `attention` component, so it reuses that tensor-parallel port: q/k/v and
gate/up column-parallel (by head / intermediate), o_proj and down row-parallel + all_reduce, norms and
rotary tables replicated.
"""
from __future__ import annotations

import ttnn  # noqa: F401  (native ttnn forward lives in TtAttention)

from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.attention import TtAttention


class TtLayer(TtAttention):
    pass


def build(device, torch_module=None):
    return TtLayer.build(device, torch_module)


def layer(device, torch_module=None):
    return TtLayer.build(device, torch_module)
