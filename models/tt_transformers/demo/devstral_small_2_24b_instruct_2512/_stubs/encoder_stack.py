# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `encoder_stack` (PixtralTransformer, `vision_tower.transformer`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    h = x;  for layer in layers: h = PixtralAttentionLayer(h, mask, (cos, sin))

Every layer is a graduated PixtralAttentionLayer port. Three graduated stubs port that same module
(`pixtral_attention_layer.py`, `layer.py`, `attention.py`; the first two subclass the third), and two
port its feed-forward (`pixtral_m_l_p.py`, `mlp.py`), so the stack holds ONE list of TtAttention-based
layers that rotates through those classes -- the layers share a type (TtAttention) and every graduated
port runs real vision layers. Each layer is tensor-parallel (q/k/v and gate/up column-parallel by
head / intermediate, o_proj and down row-parallel + all_reduce, norms replicated), so the residual
between layers is full-width and replicated on every device. Rotary tables and the mask are uploaded
once and shared by all layers.

`n_layers` caps the depth built (None = every layer of the checkpoint, never fewer than 1).
"""
from __future__ import annotations

import torch
import ttnn

from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.attention import TtAttention
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.layer import TtLayer
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.mlp import TtMLP
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_attention_layer import (
    TtPixtralAttentionLayer,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_m_l_p import TtPixtralMLP

LAYER_CLASSES = (TtPixtralAttentionLayer, TtLayer, TtAttention)
MLP_CLASSES = (TtPixtralMLP, TtMLP)


class TtEncoderStack:
    def __init__(self, device, torch_module, n_layers=None) -> None:
        self.device = device
        total = len(torch_module.layers)
        depth = total if n_layers is None else max(1, min(int(n_layers), total))
        self.layers = [
            LAYER_CLASSES[i % len(LAYER_CLASSES)](device, layer, mlp_cls=MLP_CLASSES[i % len(MLP_CLASSES)])
            for i, layer in enumerate(torch_module.layers[:depth])
        ]
        self.head_dim = self.layers[0].head_dim

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, inputs_embeds, attention_mask=None, position_embeddings=None, **kwargs):
        x = inputs_embeds
        in_shape = tuple(x.shape)
        if len(in_shape) == 3:
            x = ttnn.reshape(x, (in_shape[0], 1, in_shape[1], in_shape[2]))
        s, hd = x.shape[2], self.head_dim
        to_tt = self.layers[0]._to_tt

        cos, sin = position_embeddings
        if isinstance(cos, torch.Tensor):
            cos = to_tt(cos.reshape(1, 1, s, hd))
        if isinstance(sin, torch.Tensor):
            sin = to_tt(sin.reshape(1, 1, s, hd))
        mask = attention_mask
        if isinstance(mask, torch.Tensor):
            mask = to_tt(mask.reshape(1, 1, s, s).float())

        for layer in self.layers:
            x = layer(x, attention_mask=mask, position_embeddings=(cos, sin))
        if len(in_shape) == 3:
            return ttnn.reshape(x, in_shape)
        return x


def build(device, torch_module=None):
    return TtEncoderStack.build(device, torch_module)


def encoder_stack(device, torch_module=None):
    return TtEncoderStack.build(device, torch_module)
