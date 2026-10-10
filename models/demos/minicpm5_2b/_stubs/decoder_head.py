# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `lm_head` (decoder_head) for `openbmb/MiniCPM5-2B`.

HF semantics: logits = x @ W^T, Linear(2048 -> 130560, bias=False).
"""
from __future__ import annotations

import torch

import ttnn

_VOCAB_CHUNK = 16320  # 510 tiles; 130560 = 8 chunks


def _mesh_kw(device, **kw):
    kw["device"] = device
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except AttributeError:
        pass
    return kw


class DecoderHead:
    def __init__(self, device, torch_module):
        self.device = device
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Split the 130560-wide vocab into tile-aligned column chunks (as tt_transformers' LMHead does)
        # so no single matmul spans the full vocab.
        w = torch_module.weight.detach().to(torch.float32).t().contiguous()
        self.weights = [
            ttnn.from_torch(
                chunk.contiguous(),
                **_mesh_kw(device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            )
            for chunk in torch.split(w, _VOCAB_CHUNK, dim=-1)
        ]

    def __call__(self, x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, **_mesh_kw(self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        elif x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        outs = [
            ttnn.linear(x, w, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for w in self.weights
        ]
        out = ttnn.concat(outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            ttnn.deallocate(t)
        return out


def build(device, torch_module=None):
    return DecoderHead(device, torch_module)


def decoder_head(device, torch_module=None):
    return DecoderHead(device, torch_module)
