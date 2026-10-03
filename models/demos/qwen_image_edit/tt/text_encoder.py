# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen2.5-VL prompt encoder (QwenImageEditPipeline._get_qwen_prompt_embeds) chained from the graduated ports.

    img  = vision_transformer_pretrained_model(pixels)         vision_patch_embed -> 32 x v_l_vision_block -> v_l_patch_merger
    x    = token_embed(ids) with img spliced over the <|image_pad|> run
    h    = v_l_text_model(x)                                    28 x v_l_decoder_layer (language_model_layers_0_mlp) + norm
    emb  = h[:, 64:L]                                           drop the prompt template (prompt_template_encode_start_idx)

Mesh: every port keeps its graduated TP=4 split over the columns. On a 2-row mesh v_l_text_model is
row-staged (its own default there): layers [0, n/2) on row 0, [n/2, n) on row 1.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn
from models.demos.qwen_image_edit.tt.inputs import PROMPT_TEMPLATE_DROP
from models.demos.qwen_image_edit_text_encoder._stubs import (
    v_l_rotary_embedding,
    v_l_text_model,
    vision_transformer_pretrained_model,
)
from models.demos.qwen_image_edit_text_encoder._stubs.attention import pad_to_tile
from models.demos.qwen_image_edit_text_encoder._stubs.layer import text_attention_mask


def replicated(device, t, dtype, layout=ttnn.TILE_LAYOUT):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, **kw)


class TextEncoderInputs:
    """Device-resident encoded inputs of one batched call (uploaded outside the forward)."""


class TtQwenTextEncoder:
    def __init__(self, device, hf_text_encoder, vision_layers=None, text_layers=None, tracker=None):
        self.device = device
        te = hf_text_encoder
        self.config = te.config
        self.image_token_id = int(te.config.image_token_id)
        tc = te.config.text_config
        self.pad_token_id = int(getattr(tc, "pad_token_id", None) or tc.bos_token_id)  # masked out either way
        visual, lm = te.model.visual, te.model.language_model
        n_vis, n_lm = len(visual.blocks), len(lm.layers)
        self.num_vision_layers = n_vis if vision_layers is None else max(1, min(int(vision_layers), n_vis))
        self.num_text_layers = n_lm if text_layers is None else max(1, min(int(text_layers), n_lm))
        rows = tuple(device.shape)[0] if isinstance(device, ttnn.MeshDevice) else 1
        if rows == 2 and self.num_text_layers % 2:
            # the row-staged LM holds one layer per row per slot: a cap rounds up to even
            self.num_text_layers = min(n_lm, self.num_text_layers + 1)
        # a depth cap builds the first N repeats of each stack; embeddings, norms and merger stay whole
        vis_blocks, lm_layers = visual.blocks, lm.layers
        try:
            visual.blocks = torch.nn.ModuleList(list(vis_blocks)[: self.num_vision_layers])
            lm.layers = torch.nn.ModuleList(list(lm_layers)[: self.num_text_layers])
            self.visual = vision_transformer_pretrained_model.build(device, visual)
            self.text_model = v_l_text_model.build(device, lm)
        finally:
            visual.blocks, lm.layers = vis_blocks, lm_layers
        self.rotary = v_l_rotary_embedding.build(device, lm.rotary_emb)
        self.mrope_section = list(te.config.text_config.rope_parameters["mrope_section"])
        self.group = self.text_model.layers[0].self_attn.group
        self.set_precise(True)
        if tracker is not None:
            tracker.track("vision_transformer_pretrained_model", self.visual, ("forward_batched",))
            tracker.track("vision_patch_embed", self.visual.patch_embed)
            for blk in self.visual.blocks:
                tracker.track("v_l_vision_block", blk, ("forward_padded",))
            tracker.track("v_l_patch_merger", self.visual.merger, ("forward_padded",))
            tracker.track("v_l_text_model", self.text_model, ("forward_padded",))
            for lyr in self.text_model.layers:
                tracker.track("v_l_decoder_layer", lyr, ("forward_padded",))
                tracker.track("language_model_layers_0_mlp", lyr.mlp)

    # the repeated blocks, as the plain lists the ports hold (for depth discovery)
    @property
    def vision_blocks(self):
        return self.visual.blocks

    @property
    def text_layers(self):
        return self.text_model.layers

    def set_precise(self, on=True):
        """Float32 / exact-product mode of every text-encoder port (the ports' `precise` switches).

        The vision tower grows massive activations (|x| ~ 2.6e4 at blocks 17 and 31) that amplify
        bf16-level rounding of the branch inputs, so its matmul inputs are carried as 3 bf16 limbs with
        exact-lane accumulation. The LM takes 2-limb exact-lane products: every denoising step is
        conditioned on its output, and its dense-product error (prompt-embedding rel. error 3.2e-3, vs
        2.2e-4 exact-lane; measured on the example) was the largest error source of the final latents."""
        vl = 3 if on else 2
        v = self.visual
        v.precise = on
        v.limbs = v.patch_embed.limbs = vl
        for blk in v.blocks:
            b = blk.block
            b.attn.precise = b.mlp.precise = b.norm1.precise = b.norm2.precise = on
            b.precise_inputs = on
            b.attn.limbs = b.mlp.limbs = vl
        m = v.merger.merger
        m.precise = m.ln_q.precise = on
        m.limbs = vl
        for lyr in self.text_model.layers:
            lyr.self_attn.precise = lyr.mlp.precise = on
            lyr.self_attn.exact = lyr.mlp.exact = True
            lyr.input_layernorm.precise = lyr.post_attention_layernorm.precise = on
            lyr.precise_inputs = on
        self.text_model.norm.precise = on
        if on:  # upload the float32 norm weights at build time, not inside the first forward
            norms = [self.text_model.norm, m.ln_q]
            for blk in v.blocks:
                norms += [blk.block.norm1, blk.block.norm2]
            for lyr in self.text_model.layers:
                norms += [lyr.input_layernorm, lyr.post_attention_layernorm]
            for n in norms:
                n._ensure_w32()

    # ---- input encoding -> device (outside the forward) ------------------------------------------
    def prepare(self, vl):
        """vl: the Qwen2VLProcessor output (input_ids, attention_mask, pixel_values, image_grid_thw)."""
        d = self.device
        ids, am = vl["input_ids"], vl["attention_mask"]
        B, S = ids.shape
        grids = [tuple(int(v) for v in g) for g in vl["image_grid_thw"].tolist()]
        assert len(grids) == B and len(set(grids)) == 1, "one condition image per sequence, all on one grid"

        p = TextEncoderInputs()
        p.B = B
        vc = self.visual.consts(grids[0], B)
        pix = vl["pixel_values"].to(torch.float32).reshape(B, 1, vc.s, -1)
        pix = torch.nn.functional.pad(pix, (0, 0, 0, vc.s_pad - vc.s))
        p.vision_consts = vc
        p.pixels = replicated(d, pix, ttnn.float32)

        s_pad = pad_to_tile(S)
        ids_p = torch.full((B, s_pad), self.pad_token_id, dtype=torch.int64)
        am_p = torch.zeros((B, s_pad), dtype=torch.int64)
        ids_p[:, :S], am_p[:, :S] = ids, am
        pos = [(r == self.image_token_id).nonzero().flatten() for r in ids_p]
        p0, m = int(pos[0][0]), int(pos[0].numel())
        assert all(int(q[0]) == p0 and q.numel() == m for q in pos) and m == vc.m, "image token run differs"
        p.img_start, p.img_len, p.s_pad = p0, m, s_pad
        p.ids = replicated(d, ids_p.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        p.mask = replicated(d, torch.from_numpy(text_attention_mask(am_p, B, s_pad, s_pad, self.group)), ttnn.float32)
        # HF passes no position_ids and no mm_token_type_ids: arange on all three mRoPE axes
        posn = torch.arange(s_pad, dtype=torch.float32).reshape(1, 1, s_pad, 1).expand(3, 1, s_pad, 1)
        p.positions = replicated(d, posn, ttnn.float32)

        # output = h[valid][64:] per sequence, zero-padded to the longest (HF _extract_masked_hidden)
        lens = am.sum(1)
        assert bool((am == (torch.arange(S)[None, :] < lens[:, None]).long()).all()), "right-padded prompts only"
        L = int(lens.max())
        p.out_len = L - PROMPT_TEMPLATE_DROP
        keep = am_p[:, PROMPT_TEMPLATE_DROP:L].to(torch.float32)
        p.keep = None if bool(keep.all()) else replicated(d, keep.reshape(B, p.out_len, 1), ttnn.float32)
        # QwenImageEditPipeline.encode_prompt drops an all-ones mask (None)
        p.prompt_mask = None if bool(keep.all()) else keep
        return p

    # ---- device forward --------------------------------------------------------------------------
    def _mrope(self, t3):
        """[3, 1, S, D] per-axis tables -> [1, 1, S, D] (HF apply_multimodal_rotary_pos_emb section select)."""
        bounds = np.cumsum([0] + self.mrope_section * 2).tolist()
        _, b, s, _ = t3.shape
        parts = [
            ttnn.slice(t3, [i % 3, 0, 0, lo], [i % 3 + 1, b, s, hi])
            for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]))
        ]
        return ttnn.concat(parts, dim=-1)

    def encode_vision(self, p):
        """Vision tower over the B condition images -> image embeddings [B, m, 3584]."""
        _, img = self.visual.forward_batched(p.pixels, p.vision_consts)
        return img

    def encode_text(self, p, img):
        """-> prompt_embeds [B, L - 64, 3584] float32, replicated."""
        tok = ttnn.typecast(self.text_model.embed_tokens(p.ids), ttnn.float32)  # [B, s_pad, C]
        B, s_pad, C = tok.shape[0], tok.shape[1], tok.shape[2]
        a, m = p.img_start, p.img_len
        # HF: inputs_embeds.masked_scatter(image_mask, image_embeds)
        x = ttnn.concat(
            [
                ttnn.slice(tok, [0, 0, 0], [B, a, C]),
                ttnn.typecast(img, ttnn.float32),
                ttnn.slice(tok, [0, a + m, 0], [B, s_pad, C]),
            ],
            dim=1,
        )
        x = ttnn.reshape(x, (B, 1, s_pad, C))
        cos3, sin3 = self.rotary(None, p.positions, dtype=ttnn.float32)
        h = self.text_model.forward_padded(x, self._mrope(cos3), self._mrope(sin3), p.mask)
        L, drop = p.out_len, PROMPT_TEMPLATE_DROP
        h = ttnn.reshape(ttnn.slice(h, [0, 0, drop, 0], [B, 1, drop + L, C]), (B, L, C))
        return h if p.keep is None else ttnn.multiply(h, p.keep)

    def __call__(self, p):
        return self.encode_text(p, self.encode_vision(p))
