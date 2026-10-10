# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `layer` (model.layers.0, LlamaDecoderLayer) for `openbmb/MiniCPM5-2B`.

The repeated transformer block of this decoder-only model; it is the same module as
`decoder_layer`, so it reuses that native port (pre-norm attention + SwiGLU MLP with residuals). Every matmul in
that port runs with packer_l1_acc off: with fp32 dest accumulation it intermittently stalled readback.
"""
from __future__ import annotations

from models.demos.minicpm5_2b._stubs.decoder_layer import TtDecoderLayer


def build(device, torch_module=None, parts=None):
    return TtDecoderLayer.build(device, torch_module, parts=parts)


def layer(device, torch_module=None, parts=None):
    return TtDecoderLayer.build(device, torch_module, parts=parts)
