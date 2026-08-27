# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for decoder_head (lm_head).

Maps to: lm_head on VoxtralForConditionalGeneration
Simple linear projection: hidden_size -> vocab_size, no bias.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


class TtLMHead:
    def __init__(self, device, torch_module):
        self.device = device
        self.weight = ttnn.from_torch(
            torch_module.weight.T.contiguous().bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, x, **kwargs):
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4_CFG)


def build(device, torch_module=None):
    return TtLMHead(device, torch_module)
