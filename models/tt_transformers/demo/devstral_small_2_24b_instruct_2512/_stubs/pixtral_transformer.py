# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `pixtral_transformer` (PixtralTransformer, `vision_tower.transformer`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    h = x;  for layer in layers: h = PixtralAttentionLayer(h, mask, (cos, sin))

The same module as the `encoder_stack` component, so it reuses that tensor-parallel port: every layer
has q/k/v and gate/up column-parallel (by head / intermediate), o_proj and down row-parallel +
all_reduce, norms replicated; rotary tables and mask are uploaded once and shared by all layers.
"""
from __future__ import annotations

import ttnn  # noqa: F401  (native ttnn forward lives in TtEncoderStack / TtAttention)

from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.encoder_stack import TtEncoderStack


class TtPixtralTransformer(TtEncoderStack):
    pass


def build(device, torch_module=None):
    return TtPixtralTransformer.build(device, torch_module)


def pixtral_transformer(device, torch_module=None):
    return TtPixtralTransformer.build(device, torch_module)
