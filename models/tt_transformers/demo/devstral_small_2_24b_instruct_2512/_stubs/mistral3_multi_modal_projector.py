# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `mistral3_multi_modal_projector` (Mistral3MultiModalProjector,
`multi_modal_projector`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    x   = norm(image_features)                               (RMSNorm over vision hidden 1024)
    x   = patch_merger(x)                                    (2x2 unfold + merging_layer: 4*1024 -> 1024)
    out = linear_2(gelu(linear_1(x)))                        (1024 -> 5120 -> 5120)

Composed from the graduated child ports:
  * norm          `mistral3_r_m_s_norm.py` (replicated weight)
  * patch_merger  `mistral3_patch_merger.py` (merging_layer column-parallel + all_gather)

Layouts: packed (N, d) tokens of all images -> (M, 5120); images side by side (B, h*w, d) /
(B, 1, h*w, d) of one size -> (B, 1, h*w/4, 5120).

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * linear_1  column-parallel on the text hidden dim (gelu stays per-device).
  * linear_2  row-parallel on the same dim, then all_reduce.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.mistral3_patch_merger import (
    TtMistral3PatchMerger,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.mistral3_r_m_s_norm import TtMistral3RMSNorm


def _mesh_size(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


class TtMistral3MultiModalProjector:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.tp = tp = _mesh_size(device)
        self.norm = TtMistral3RMSNorm.build(device, torch_module.norm)
        self.patch_merger = TtMistral3PatchMerger(device, torch_module.patch_merger)
        self.d = torch_module.norm.weight.shape[0]
        text_hidden = torch_module.linear_1.weight.shape[0]
        assert text_hidden % tp == 0, f"text hidden {text_hidden} not divisible by TP={tp}"

        is_mesh = tp > 1
        shard = lambda dim: ttnn.ShardTensorToMesh(device, dim=dim) if is_mesh else None
        self._repl = repl = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

        def _put(t, mapper):
            return ttnn.from_torch(
                t.detach().to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.w1 = _put(torch_module.linear_1.weight.T, shard(-1))
        self.w2 = _put(torch_module.linear_2.weight.T, shard(-2))
        b1, b2 = torch_module.linear_1.bias, torch_module.linear_2.bias
        self.b1 = _put(b1.reshape(1, 1, 1, -1), shard(-1)) if b1 is not None else None
        self.b2 = _put(b2.reshape(1, 1, 1, -1), repl) if b2 is not None else None

        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, image_features, image_sizes=None, **kwargs):
        x = image_features
        side_by_side = x.shape[0] > 1 and len(x.shape) >= 3
        if not side_by_side:
            x = ttnn.reshape(x, (1, 1, x.shape[-2], self.d))
        x = self.patch_merger(self.norm(x), image_sizes)
        if not side_by_side:
            x = ttnn.reshape(x, (1, 1, x.shape[-2], x.shape[-1]))

        hdn = ttnn.linear(x, self.w1, bias=self.b1, compute_kernel_config=self.compute_cfg)
        hdn = ttnn.gelu(hdn, fast_and_approximate_mode=False)
        # minimal_matmul folds the images into its rows over the full grid (ttnn.linear ran the side-by-side batch
        # of 32 x 64 rows, K 1280, N 5120 as a batched matmul on 80 cores: 1.14 ms).
        out = ttnn.experimental.minimal_matmul(
            hdn, self.w2, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.tp > 1:
            out = ttnn.all_reduce(out, topology=ttnn.Topology.Linear)
        if self.b2 is not None:
            out = ttnn.add(out, self.b2)
        if side_by_side:
            return out
        m = out.shape[-2]
        return ttnn.reshape(out, (m, out.shape[-1]))


def build(device, torch_module=None):
    return TtMistral3MultiModalProjector.build(device, torch_module)


def mistral3_multi_modal_projector(device, torch_module=None):
    return TtMistral3MultiModalProjector.build(device, torch_module)
