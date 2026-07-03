# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from models.demos.hf_eager.acestep_v15_base._stubs.ace_step_condition_encoder import build as _ce_build

from .common import resolve, to_torch


class ConditionEncoderTT:
    def __init__(self, device, hf_model):
        self.device = device
        self.mod = _ce_build(device, resolve(hf_model, "encoder"))

    def __call__(
        self,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask,
    ):
        enc_hidden, enc_mask = self.mod(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )
        enc_hidden_t = to_torch(enc_hidden, self.device)
        enc_mask_t = enc_mask if isinstance(enc_mask, torch.Tensor) else to_torch(enc_mask, self.device)
        return enc_hidden_t, enc_mask_t.bool()


def build_condition_encoder(device, hf_model):
    return ConditionEncoderTT(device, hf_model)
