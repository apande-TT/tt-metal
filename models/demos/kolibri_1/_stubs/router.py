# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN Kolibri-1 MoE router logits (`router`, `model.layers.0.mlp.gate`).

Reference (`Kolibri1Router` in tests/pcc/_reference_loader.py): fp32 logits from the bf16 weight,

    logits = x.float() @ gate.weight.float().T        # [tokens, 384]

`e_score_correction_bias` is not part of this forward -- it only biases the top-k SELECTION, which
the MoE block (`_kolibri_moe.py`) applies.

Placement on the mesh: replicated, deliberately. Every chip needs all 384 logits to pick each
token's top-6 experts, and the weight is 2 MB, so splitting it would only add an all_gather in
front of the selection. Each chip computes the identical logits with no collective.
"""
from __future__ import annotations

import ttnn


class TtKolibri1Router:
    def __init__(self, device, torch_module) -> None:
        mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        self.w = ttnn.from_torch(
            torch_module.weight.float().t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mapper,
        )
        self.cfg = ttnn.init_device_compute_kernel_config(
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
        return ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtKolibri1Router.build(device, torch_module)


# Backward-compatible slug shim.
def router(device, torch_module=None):
    return TtKolibri1Router.build(device, torch_module)
