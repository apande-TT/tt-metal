# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN Kolibri-1 backbone (`model`, `model`), tensor/expert-parallel.

Reference (`Kolibri1Model` in tests/pcc/_reference_loader.py):

    h = embed_tokens(input_ids)
    for layer in layers: h = layer(h, position_ids)      # 50 sandwich-norm attention + MoE layers
    return norm(h)

Every decoder layer is the graduated `decoder_layer` stub: attention heads split across the TP chips
and experts split across them, each sublayer closing with an all_reduce, so the hidden state is
whole and identical on every chip between layers. The embedding table and the final norm are
replicated (the token lookup is a gather, not a matmul, so there is nothing to split).
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.kolibri_1._stubs.decoder_layer import TtKolibri1DecoderLayer


class TtKolibri1Model:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self._mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        self.embed = ttnn.from_torch(
            torch_module.embed_tokens.weight.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=self._mapper,
        )
        self.layers = [TtKolibri1DecoderLayer(device, layer) for layer in torch_module.layers]
        self.eps = float(torch_module.norm.variance_epsilon)
        self.norm_w = ttnn.from_torch(
            torch_module.norm.weight.float().reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=self._mapper,
        )
        self.norm_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(
        self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, all_hidden=None, **kwargs
    ):
        if past_key_values is not None or all_hidden is not None:
            raise NotImplementedError("Kolibri-1 TT model: the KV cache and hidden-state capture are not wired yet")
        if isinstance(input_ids, torch.Tensor):
            tokens = input_ids.to(torch.int32)
            input_ids = ttnn.from_torch(
                tokens,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=self._mapper,
            )
        # fp32 residual stream (the reference runs the whole stack in fp32); sublayers compute in bf16.
        h = ttnn.typecast(ttnn.embedding(input_ids, self.embed, layout=ttnn.TILE_LAYOUT), ttnn.float32)
        for layer in self.layers:
            h = layer(h, position_ids, attention_mask)
        return ttnn.rms_norm(h, epsilon=self.eps, weight=self.norm_w, compute_kernel_config=self.norm_cfg)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtKolibri1Model.build(device, torch_module)


# Backward-compatible slug shim.
def model(device, torch_module=None):
    return TtKolibri1Model.build(device, torch_module)
