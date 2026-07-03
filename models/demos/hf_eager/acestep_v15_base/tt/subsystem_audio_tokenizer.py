# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from models.demos.hf_eager.acestep_v15_base._stubs.ace_step_audio_tokenizer import build as _tok_build

from .common import from_torch, resolve, to_torch


class AudioTokenizerTT:
    def __init__(self, device, hf_model):
        self.device = device
        self.mod = _tok_build(device, resolve(hf_model, "tokenizer"))

    def __call__(self, x_patched):
        # The stub's first op is a ttnn.typecast, so lift the host input to TTNN
        # exactly as the per-component test harness does (bf16, TILE layout).
        if isinstance(x_patched, torch.Tensor):
            x_patched = from_torch(x_patched, self.device)
        out = self.mod(x_patched)
        if isinstance(out, (tuple, list)):
            quantized, indices = out[0], out[1]
        else:
            quantized, indices = out, None
        return to_torch(quantized, self.device), (
            to_torch(indices, self.device) if indices is not None and hasattr(indices, "to_torch") else indices
        )
