# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from models.demos.hf_eager.acestep_v15_base._stubs.audio_token_detokenizer import build as _detok_build

from .common import from_torch, resolve, to_torch


class DetokenizerTT:
    def __init__(self, device, hf_model):
        self.device = device
        self.mod = _detok_build(device, resolve(hf_model, "detokenizer"))

    def __call__(self, quantized):
        # Input: torch quantized tokens [1, 10, 2048]. The graduated stub runs
        # entirely on-device, so lift a host tensor into TTNN before invoking it.
        if isinstance(quantized, torch.Tensor):
            quantized = from_torch(quantized, self.device)
        out = self.mod(quantized)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return to_torch(out, self.device)
