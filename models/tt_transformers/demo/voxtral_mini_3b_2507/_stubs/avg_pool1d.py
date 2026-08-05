# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for AvgPool1d (audio_tower.avg_pooler).

Implements nn.AvgPool1d(kernel_size=2, stride=2) using reshape + mean.
Input: (B, C, L) -> Output: (B, C, L//2)
"""
from __future__ import annotations

import ttnn


class TtAvgPool1d:
    def __init__(self, device, torch_module):
        self.device = device
        self.kernel_size = torch_module.kernel_size[0]
        self.stride = torch_module.stride[0]

    def __call__(self, x, **kwargs):
        shape = x.shape
        B, C, L = shape[0], shape[1], shape[2]
        out_len = L // self.stride

        x = ttnn.reshape(x, (B * C, out_len, self.kernel_size))
        x = ttnn.mean(x, dim=-1, keepdim=False)
        x = ttnn.reshape(x, (B, C, out_len))
        return x


def build(device, torch_module=None):
    return TtAvgPool1d(device, torch_module)
