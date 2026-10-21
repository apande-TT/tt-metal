# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `mistral3_patch_merger` (Mistral3PatchMerger,
`multi_modal_projector.patch_merger`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    out = merging_layer(unfold_2x2(image_features per image))       (4*1024 -> 1024, no bias)

The 2x2 unfold is a device reshape/permute: per image the (h*w, d) tokens become (h/2, 2, w/2, 2d),
permuted to (h/2, w/2, 2, 2d) and flattened to (h*w/4, 4d) in (row a, col b, channel c) order. torch's
unfold flattens in (c, a, b) order, so the merging weight's input columns are permuted once at build
to match -- the math is unchanged.

Layouts: packed (N, d) / (1, 1, N, d) tokens of all images -> (M, d); or images side by side
(B, 1, h*w, d) / (B, h*w, d) of one size -> (B, 1, h*w/4, d), unfolded in ONE op chain for the batch.

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * merging_layer column-parallel on its output dim, then all_gather on that dim.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_attention import vision_tp

# The side-by-side batch's 2 x 2 unfold as one tile-layout program (tt/patch_unfold.py) instead of untilize + reshape +
# permute + reshape + tilize (286 -> 40 us for 8 images of 16 x 16 patches, byte-identical).
PATCH_UNFOLD_PROGRAM = True


class TtMistral3PatchMerger:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.tp, repl = vision_tp(device)
        tp = self.tp
        self.merge = k = torch_module.spatial_merge_size
        self.patch_size = torch_module.patch_size
        wm = torch_module.merging_layer.weight.detach()  # (d, d*k*k)
        self.d = d = wm.shape[0]
        assert d % tp == 0, f"merge out dim {d} not divisible by TP={tp}"
        # unfold order (c, a, b) -> device order (a, b, c)
        wm = wm.view(d, d, k, k).permute(0, 2, 3, 1).reshape(d, k * k * d)
        self.w_merge = ttnn.from_torch(
            wm.T.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if tp > 1 else repl,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
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

    def grids(self, image_sizes):
        sizes = image_sizes
        if isinstance(sizes, torch.Tensor):
            sizes = sizes.tolist()
        return [(int(hh) // self.patch_size, int(ww) // self.patch_size) for hh, ww in sizes]

    def _unfold(self, x, b, h, w):
        """(b, 1, h*w, d) row-major -> (b, 1, h*w/k^2, k^2*d) in (a, b, c) order."""
        k, d = self.merge, self.d
        x = ttnn.reshape(x, (b * (h // k), k, w // k, k * d))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (b, 1, (h // k) * (w // k), k * k * d))

    def _merge(self, x):
        # minimal_matmul folds the images into its rows and spreads them over the full grid; ttnn.linear's auto
        # config ran the side-by-side batch (32 x 64 rows, K 4096, N 256) as a batched matmul on 16 cores (2.8 ms).
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        if minimal_mm.supported(x, self.w_merge, self.compute_cfg):
            # the stock op's default blocking as minimal_mm's program, its core ranges' padding skipped (SKIP_PADDING)
            out = minimal_mm.matmul(
                x,
                self.w_merge,
                minimal_mm.default_config(self.device, self.compute_cfg, x, self.w_merge),
                self.compute_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=x.dtype,  # the stock op's output dtype when none is given
            )
        else:
            out = ttnn.experimental.minimal_matmul(
                x, self.w_merge, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if self.tp > 1:
            out = ttnn.all_gather(out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        return out

    def __call__(self, image_features, image_sizes=None, **kwargs):
        grids = self.grids(image_sizes)
        x = image_features
        side_by_side = len(x.shape) == 4 and x.shape[0] > 1 or (len(x.shape) == 3 and x.shape[0] > 1)
        if side_by_side:
            assert len(set(grids)) == 1, "side-by-side images must share one size"
            h, w = grids[0]
            b = x.shape[0]
            x = ttnn.reshape(x, (b, 1, h * w, self.d))
            # (imported here: a module held on the instance would be walked by the perf harness's weight census)
            from models.demos.devstral_small_2_24b_instruct_2512.tt import patch_unfold

            if PATCH_UNFOLD_PROGRAM and patch_unfold.supports(x, h, w, self.merge):
                # the unfold as one tile-layout data-movement program (tt/patch_unfold.py), byte-identical
                y = patch_unfold.apply(x, h, w)
                return self._merge(y)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_layout(self._unfold(x, b, h, w), ttnn.TILE_LAYOUT)
            return self._merge(x)

        n = x.shape[-2]
        x = ttnn.to_layout(ttnn.reshape(x, (1, 1, n, self.d)), ttnn.ROW_MAJOR_LAYOUT)
        merged, start = [], 0
        for h, w in grids:
            tok = ttnn.slice(x, (0, 0, start, 0), (1, 1, start + h * w, self.d))
            merged.append(self._unfold(tok, 1, h, w))
            start += h * w
        x = merged[0] if len(merged) == 1 else ttnn.concat(merged, dim=2)
        out = self._merge(ttnn.to_layout(x, ttnn.TILE_LAYOUT))
        m = out.shape[-2]
        return ttnn.reshape(out, (m, out.shape[-1]))


def build(device, torch_module=None):
    return TtMistral3PatchMerger.build(device, torch_module)


def mistral3_patch_merger(device, torch_module=None):
    return TtMistral3PatchMerger.build(device, torch_module)
