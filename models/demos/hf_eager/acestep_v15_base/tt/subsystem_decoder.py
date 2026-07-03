# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from models.demos.hf_eager.acestep_v15_base._stubs.ace_step_di_t_model import build as _decoder_build

from .common import resolve, to_torch


class DecoderTT:
    def __init__(self, device, hf_model):
        self.device = device
        self.mod = _decoder_build(device, resolve(hf_model, "decoder"))

    def __call__(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
    ):
        out = self.mod(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
        )
        vt = out[0] if isinstance(out, (tuple, list)) else out
        return to_torch(vt, self.device)
