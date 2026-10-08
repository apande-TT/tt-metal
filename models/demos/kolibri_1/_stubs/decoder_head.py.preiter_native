# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN Kolibri-1 LM head (`decoder_head`, `lm_head`), tensor-parallel.

The canonical `LMHead` (models/tt_transformers/tt/lm_head.py) is built from `ModelArgs`, which cannot
load the `kolibri1` config (not a transformers model type), so the head is a direct ttnn linear:

    logits = x @ lm_head.weight.T        # untied, bias-free, 2560 -> 128000 (reference runs it in fp32)

Tensor parallel over the last mesh axis (TP chips): column-parallel over the vocabulary. Each chip
holds a contiguous vocab/TP slice of the weight, computes its slice of the logits, and one all_gather
along the vocab axis gives every chip the full logits. The input hidden state is replicated.
"""
from __future__ import annotations

import ttnn


class TtKolibri1LMHead:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        w = torch_module.weight.float().t().contiguous()  # [hidden, vocab]
        vocab = w.shape[-1]

        is_mesh = isinstance(device, ttnn.MeshDevice)
        mesh_shape = list(device.shape) if is_mesh else [1, 1]
        tp = mesh_shape[-1]
        if vocab % (tp * 32):
            tp = 1
        self.tp = tp
        self.tp_axis = len(mesh_shape) - 1

        mapper = None
        if is_mesh:
            mapper = (
                ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=(None, -1))
                if tp > 1
                else ttnn.ReplicateTensorToMesh(device)
            )
        self.w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)
        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, **kwargs):
        shape = list(x.shape)
        x = ttnn.reshape(x, [1, 1, -1, shape[-1]])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        out = ttnn.linear(x, self.w, compute_kernel_config=self.mm_cfg)
        if self.tp > 1:
            out = ttnn.all_gather(out, dim=3, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.reshape(out, shape[:-1] + [out.shape[-1]])


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtKolibri1LMHead.build(device, torch_module)


# Backward-compatible slug shim.
def decoder_head(device, torch_module=None):
    return TtKolibri1LMHead.build(device, torch_module)
