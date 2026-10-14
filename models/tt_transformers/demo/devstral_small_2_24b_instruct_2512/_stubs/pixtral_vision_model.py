# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `pixtral_vision_model` (PixtralVisionModel, `vision_tower`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    per image i (cropped to image_sizes[i]):  tokens_i = patch_conv(pixels_i)   (14x14, stride 14)
    x   = ln_pre(tokens)
    out = transformer(x, mask, rope2d(patch ids))

patch_conv has stride == kernel, so it is patchify + matmul: the (3, h*14, w*14) pixels are rearranged
on device by reshape/permute into (h*w, 3*14*14) rows in (c, ky, kx) order -- the conv weight's own
flatten order -- and multiplied by the flattened conv weight.

Composed from the graduated child ports:
  * ln_pre                    `pixtral_r_m_s_norm.py`
  * patch_positional_embedding `pixtral_rotary_embedding.py` (2D rope gathered on device by patch id)
  * transformer               `pixtral_transformer.py` (== `encoder_stack.py`, the TP layer stack)

Two layouts:
  * equal image sizes (the batched path): B images run side by side as (B, 1, h*w, D) with no mask --
    exactly HF's block-diagonal mask, since images never attend to each other. Returns (B, h*w, D).
  * mixed sizes: images are packed along one sequence with a block-diagonal mask (HF's own layout).
    Returns (1, sum h*w, D).

The patch ids (h * max_side + w per image, as position_ids_in_meshgrid builds them) depend only on
image_sizes; `patch_position_ids(image_sizes)` builds them once so a caller can upload them ahead of
the forward and pass the device tensor as `position_ids`.

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * transformer  per layer q/k/v and gate/up column-parallel, o_proj and down row-parallel + all_reduce.
  * patch_conv weight, ln_pre, rotary tables and mask are replicated (patch_conv is 0.6M params).
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_r_m_s_norm import TtPixtralRMSNorm
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_rotary_embedding import (
    TtPixtralRotaryEmbedding,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_transformer import (
    TtPixtralTransformer,
)


class TtPixtralVisionModel:
    def __init__(self, device, torch_module, n_layers=None) -> None:
        self.device = device
        cfg = torch_module.config
        self.patch_size = p = torch_module.patch_size
        self.max_side = cfg.image_size // cfg.patch_size
        self.hidden = hidden = cfg.hidden_size
        self.transformer = TtPixtralTransformer(device, torch_module.transformer, n_layers=n_layers)
        self.ln_pre = TtPixtralRMSNorm.build(device, torch_module.ln_pre)
        self.patch_positional_embedding = TtPixtralRotaryEmbedding(device, torch_module.patch_positional_embedding)
        self.head_dim = self.transformer.head_dim
        self._repl = repl = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        w = torch_module.patch_conv.weight  # (hidden, C, p, p)
        self.in_ch = w.shape[1]
        self.w_patch = ttnn.from_torch(
            w.reshape(hidden, self.in_ch * p * p).T.detach().to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=repl,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )

    @property
    def rope(self):
        return self.patch_positional_embedding

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def grids(self, image_sizes):
        """[(h_patches, w_patches)] per image from image_sizes (list / tensor of (H, W))."""
        sizes = image_sizes
        if isinstance(sizes, torch.Tensor):
            sizes = sizes.tolist()
        p = self.patch_size
        return [(int(hh) // p, int(ww) // p) for hh, ww in sizes]

    def patch_position_ids(self, image_sizes, batched=True):
        """Host patch ids h * max_side + w: one image's ids when every size matches (batched), else all images'."""
        return self._grid_position_ids(self.grids(image_sizes), batched=batched)

    def _grid_position_ids(self, grids, batched=True):
        """Patch ids for (h_patches, w_patches) grids (already in patches, not pixels)."""
        if batched and len(set(grids)) == 1:
            grids = grids[:1]
        per_image = []
        for h, w in grids:
            hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            per_image.append((hh * self.max_side + ww).reshape(-1))
        return torch.concat(per_image)

    def patchify_host(self, pixel_values, image_sizes):
        """Host (torch) patchify of equal-size images: (B, C, H, W) cropped to image_sizes -> (B, 1, h*w, C*p*p) rows
        in (c, ky, kx) order -- exactly what _patchify_batched builds on device -- or None for mixed sizes. A caller
        that uploads these rows (tile layout) and passes prepatched=True skips the device reshape / permute chain,
        whose row-major sticks are 14 pixels wide (~3.6 ms for 32 images of 224 x 224)."""
        grids = self.grids(image_sizes)
        if len(set(grids)) != 1:
            return None
        (h, w), p, c = grids[0], self.patch_size, self.in_ch
        pv = pixel_values[..., : h * p, : w * p]
        b = pv.shape[0]
        return pv.reshape(b, c, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(b, 1, h * w, c * p * p).contiguous()

    def _patchify(self, pixel_values, image_sizes):
        """Device patchify: (B, C, H, W) pixels -> (patch rows, grids, batched)."""
        pv = pixel_values
        if isinstance(pv, torch.Tensor):
            pv = ttnn.from_torch(
                pv.to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=self._repl,
            )
        b, _, hp, wp = pv.shape
        grids = self.grids(image_sizes if image_sizes is not None else [(hp, wp)] * b)
        p = self.patch_size
        pv = ttnn.to_layout(pv, ttnn.ROW_MAJOR_LAYOUT)

        batched = len(set(grids)) == 1
        if batched:
            h, w = grids[0]
            if (h * p, w * p) != (hp, wp):
                pv = ttnn.slice(pv, (0, 0, 0, 0), (b, self.in_ch, h * p, w * p))
            return self._patchify_batched(pv, h, w), grids, batched
        rows = []
        for i, (h, w) in enumerate(grids):
            img = ttnn.slice(pv, (i, 0, 0, 0), (i + 1, self.in_ch, h * p, w * p))
            rows.append(self._patchify_batched(img, h, w))
        return ttnn.concat(rows, dim=2), grids, batched

    def _patchify_batched(self, pv, h, w):
        """(B, C, h*p, w*p) row-major -> (B, 1, h*w, C*p*p) in (c, ky, kx) order."""
        p, c, b = self.patch_size, self.in_ch, pv.shape[0]
        x = ttnn.reshape(pv, (b * c * h, p, w, p))  # (b,c,i), ky, j, kx
        x = ttnn.permute(x, (0, 2, 1, 3))  # (b,c,i), j, ky, kx
        x = ttnn.reshape(x, (b, c, h * w, p * p))
        x = ttnn.permute(x, (0, 2, 1, 3))  # b, (i,j), c, (ky,kx)
        return ttnn.reshape(x, (b, 1, h * w, c * p * p))

    def __call__(self, pixel_values, image_sizes=None, position_ids=None, prepatched=False, **kwargs):
        """prepatched: pixel_values is patchify_host's (B, 1, h*w, C*p*p) rows already on device (equal sizes)."""
        if prepatched:
            x, grids, batched = pixel_values, self.grids(image_sizes), True
        else:
            x, grids, batched = self._patchify(pixel_values, image_sizes)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.linear(x, self.w_patch, compute_kernel_config=self.compute_cfg)
        x = self.ln_pre(x)
        s = x.shape[-2]

        if position_ids is None:
            position_ids = self._grid_position_ids(grids)
        cos, sin = self.patch_positional_embedding(x, position_ids=position_ids)
        cos = ttnn.reshape(cos, (1, 1, s, self.head_dim))
        sin = ttnn.reshape(sin, (1, 1, s, self.head_dim))

        # SDPA chunks the patch sequence, so it must be a tile multiple: pad it (zero rows) and mask the padded
        # keys; the padded rows are sliced off the output. A 1540 px image is 110 x 110 = 12100 patches.
        sp = self.padded_len(s)
        if sp != s:
            pad = [(0, 0), (0, 0), (0, sp - s), (0, 0)]
            x = ttnn.pad(x, padding=pad, value=0.0)
            cos, sin = ttnn.pad(cos, padding=pad, value=0.0), ttnn.pad(sin, padding=pad, value=0.0)
        mask = None if (batched and sp == s) else self._mask(s, sp, grids, batched)

        out = ttnn.typecast(self.transformer(x, attention_mask=mask, position_embeddings=(cos, sin)), ttnn.bfloat16)
        if sp != s:
            out = ttnn.slice(out, (0, 0, 0, 0), (out.shape[0], 1, s, out.shape[-1]))
        return ttnn.reshape(out, (out.shape[0], s, out.shape[-1]))

    @staticmethod
    def padded_len(s):
        """Sequence length SDPA runs at: s itself when it is a tile multiple, else the next multiple of 32
        (of 512 past 512, so the wide key chunk -- the least level-biased -- still applies)."""
        if s % 32 == 0:
            return s
        unit = 512 if s > 512 else 32
        return -(-s // unit) * unit

    def _mask(self, s, sp, grids, batched):
        """(1, 1, sp, sp) additive mask: block-diagonal over packed images, padded keys masked. Built and
        uploaded once per layout and kept resident, so a repeated (or traced) forward issues no host work."""
        key = (s, sp, tuple(grids), batched)
        cache = self.__dict__.setdefault("_mask_cache", {})
        if key not in cache:
            neg = torch.finfo(torch.bfloat16).min
            m = torch.full((sp, sp), neg, dtype=torch.float32)
            if batched:
                m[:, :s] = 0  # one image per batch row: every query sees every real patch
            else:
                start = 0
                for h, w in grids:
                    m[start : start + h * w, start : start + h * w] = 0
                    start += h * w
                m[s:, :s] = 0  # padded queries read real keys (no all-masked row); their output is dropped
            cache[key] = ttnn.from_torch(
                m.reshape(1, 1, sp, sp).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=self._repl,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return cache[key]


def build(device, torch_module=None):
    return TtPixtralVisionModel.build(device, torch_module)


def pixtral_vision_model(device, torch_module=None):
    return TtPixtralVisionModel.build(device, torch_module)
