# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen2.5-VL text encoder (QwenImageEditPipeline._get_qwen_prompt_embeds), chained from the graduated ports.

    img_emb  = vision_transformer_pretrained_model(pixels)         patch_embed -> 32 blocks -> merger
    x        = token_embed(ids) with img_emb spliced at the <|image_pad|> run
    h        = v_l_text_model(x)                                   28 decoder layers + final norm
    prompt   = h[:, 64:L] * mask                                   drop the template prefix, zero the padding

Mesh: 8x4 (Galaxy). Every layer keeps its graduated TP=4 split over the 4 columns (mesh axis 1). The
LM is stage-split over the 8 rows: rows 0-3 hold layers 0-13 and rows 4-7 hold layers 14-27 (half of
the TP=4 share of the stack per chip), with an exact all_gather hand-off over axis 0 between the two
stages (v_l_text_model row_stages, generalised from the T3K's 2 single-row stages to 2 stages of
rows/2 rows). Replicating the whole LM on every row instead is 4.72 GB per chip (measured on T3K);
next to the 5.36 GB TP=8 transformer and the 0.41 GB VAE that is 10.49 GB, over the registered 10.5
GB usable per chip with a CCL axis once activations are counted. The vision tower splits its images
over the rows.
"""
from __future__ import annotations

import numpy as np
import torch

import ttnn
from models.demos.qwen_image_edit.tt.inputs import PROMPT_TEMPLATE_DROP
from models.demos.qwen_image_edit_text_encoder._stubs import (
    language_model_layers_0_mlp,
    v_l_decoder_layer,
    v_l_patch_merger,
    v_l_rotary_embedding,
    v_l_text_model,
    v_l_vision_block,
    vision_patch_embed,
    vision_transformer_pretrained_model,
)
from models.demos.qwen_image_edit_text_encoder._stubs import attention as _te_attention
from models.demos.qwen_image_edit_text_encoder._stubs.attention import MASK_NEG, pad_to_tile
from models.demos.qwen_image_edit_text_encoder._stubs.layer import text_attention_mask


def _replicated(device, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )


LM_EXACT = False
# the vision tower needs the exact 8-lane K reduction (without it the image PCC measured 0.55), but it
# forms it with one lane pattern instead of the median of three: the median only outvotes a rare
# power-of-two glitch, and it triples the ~24 matmuls + masks + adds of every vision linear.
# Back to the median (True = every lane pattern): on tt-metal f6b166e383 one image's min PCC fell to
# 0.942 with a single pattern and the median holds it at the gate (models without this win: 0.9607).
VISION_EXACT = True
# bf16 limbs of the vision MLP's input in the precise path (the rest of the tower uses 3)
VISION_MLP_LIMBS = 2
# bf16 limbs of the vision attention's operands in the precise path
VISION_ATTN_LIMBS = 2
# negatives reuse the prompts' K/V of their common token prefix (whole tiles): the LM is causal, so
# those positions' hidden states are identical, and only the negatives' tail is run (attending to it)
PREFIX_SHARE = True
# mesh axis the vision batch is split over (the tower's TP=4 runs on the other axis)
VISION_DP_AXIS = 0


class TextEncoderInputs:
    """Device-resident encoded inputs for one batched call (uploaded once, outside the forward)."""


class TtQwenTextEncoder:
    def __init__(self, device, hf_text_encoder, text_layers=None, vision_layers=None, tracker=None):
        self.device = device
        te = hf_text_encoder
        self.hf_config = te.config
        self.image_token_id = int(te.config.image_token_id)
        visual, lm = te.model.visual, te.model.language_model
        n_lm = len(lm.layers)
        self.num_text_layers = n_lm if text_layers is None else max(1, min(int(text_layers), n_lm))
        rows = tuple(device.shape)[0]
        if rows >= 2 and rows % 2 == 0 and self.num_text_layers % 2:
            # the two-stage row-staged stack needs one layer per stage per slot: round a cap up to even
            self.num_text_layers = min(n_lm, self.num_text_layers + 1)
        self.num_vision_layers = (
            len(visual.blocks) if vision_layers is None else max(1, min(int(vision_layers), len(visual.blocks)))
        )
        # Depth caps build fewer repeats of each stack; the rest of each module is intact.
        vis_blocks, lm_layers = visual.blocks, lm.layers
        try:
            visual.blocks = torch.nn.ModuleList(list(vis_blocks)[: self.num_vision_layers])
            lm.layers = torch.nn.ModuleList(list(lm_layers)[: self.num_text_layers])
            self.visual = vision_transformer_pretrained_model.build(device, visual)
            self.text_model = v_l_text_model.build(device, lm)
        finally:
            visual.blocks, lm.layers = vis_blocks, lm_layers
        self.rotary = v_l_rotary_embedding.build(device, lm.rotary_emb)
        # the LM's 2-limb projections run on the full core grid (ttnn.linear's own pick left it partial)
        grid = device.compute_with_storage_grid_size()
        _te_attention.LINEAR_CORE_GRID = ttnn.CoreGrid(y=grid.y, x=grid.x)
        _te_attention.LINEAR_BLOCK = True
        _te_attention.LINEAR_BLOCK_TRANSPOSE = True
        # the exact fp32 TP all-reduce gathers and sums in L1, in chunks of at most 48 MB per gather
        _te_attention.EXACT_REDUCE_L1_BYTES = 48 * 1024 * 1024
        # where the batch splits over the TP devices: all_to_all + local adds + all_gather instead
        _te_attention.A2A_REDUCE = True
        # the row-stage hand-off moves one piece per row instead of every row's full copy
        v_l_text_model.SPLIT_HANDOFF = True
        # the vision attention's exact-lane products: one batch-parallel full-K block per core on the grid
        _te_attention.EXACT_CORE_GRID = ttnn.CoreGrid(y=grid.y, x=grid.x)
        # and its exact-lane linears on the full core grid (ttnn's own pick: 30-48 cores)
        _te_attention.EXACT_LINEAR_CORE_GRID = ttnn.CoreGrid(y=grid.y, x=grid.x)
        # and keeps the lane copies / products / sums of products up to 6 MB in L1
        _te_attention.EXACT_SUM_L1_BYTES = 6 * 1024 * 1024
        # and the attention's exact-lane products (QK^T ~7.9 MB, PV ~2.2 MB) likewise
        _te_attention.EXACT_BMM_L1_BYTES = 8 * 1024 * 1024
        # the vision token permutations as one bf16 matmul per limb (one-hot rows: each output is a single
        # 1 * v product, already exact without the exact-lane decomposition)
        vision_transformer_pretrained_model.ONEHOT_PERMUTE = True
        # MLP gate / up share their input's bf16 limb split (LM and vision tower)
        _te_attention.SHARE_SPLIT = True
        # the LM attention moves heads by per-head slices + concat, not a padded reshape + permute
        from models.demos.qwen_image_edit_text_encoder._stubs import layer as _te_layer

        _te_layer.HEADS_BY_SLICES = True
        self.set_precise(True)
        self.mrope_section = list(te.config.text_config.rope_parameters["mrope_section"])
        self.group = self.text_model.layers[0].self_attn.group

        if tracker is not None:
            tracker.track(
                "vision_transformer_pretrained_model",
                self.visual,
                ("forward_batched",),
                stub_module=vision_transformer_pretrained_model,
            )
            tracker.track("vision_patch_embed", self.visual.patch_embed, stub_module=vision_patch_embed)
            for blk in self.visual.blocks:
                tracker.track("v_l_vision_block", blk, ("forward_padded",), stub_module=v_l_vision_block)
            tracker.track("v_l_patch_merger", self.visual.merger, ("forward_padded",), stub_module=v_l_patch_merger)
            tracker.track("v_l_text_model", self.text_model, ("forward_padded",), stub_module=v_l_text_model)
            for lyr in self.text_model.layers:
                tracker.track("v_l_decoder_layer", lyr, ("forward_padded",), stub_module=v_l_decoder_layer)
                tracker.track("language_model_layers_0_mlp", lyr.mlp, stub_module=language_model_layers_0_mlp)

    def set_precise(self, on=True):
        """Float32 / exact-product path through every text-encoder port (see the ports' `precise` flags).

        Why: both towers grow massive activations (vision |x| 44 -> 8136 at block 17 and 26208 at block
        31) that amplify bf16-level rounding. Measured at B=2 over the full depth: merged image
        embeddings PCC 0.9936 (bf16 branches) -> 0.9992, and the LM on HF's own embeddings 0.99984
        (bf16 MLP) before this flag."""
        # the vision tower sits at an ill-conditioned point for some inputs (e.g. patches next to the
        # massive-activation tokens at blocks 16 and 31 amplify input error ~1000x; HF fp32 vs fp64
        # differs by <0.4% there): its matmul inputs are carried as 3 bf16 limbs (~float32)
        vl = 3 if on else 2
        self.visual.precise = on
        self.visual.limbs = self.visual.patch_embed.limbs = vl
        self.visual.exact = self.visual.patch_embed.exact = VISION_EXACT
        for blk in self.visual.blocks:
            b = blk.block
            b.attn.precise = b.mlp.precise = b.norm1.precise = b.norm2.precise = on
            b.precise_inputs = on
            # attention operands as 2 bf16 limbs too: 3 limb terms per exact product instead of 6
            b.attn.limbs = min(vl, VISION_ATTN_LIMBS)
            # the MLP input carried as 2 bf16 limbs (~16 mantissa bits) instead of 3: 48 lane products, not 72
            b.mlp.limbs = min(vl, VISION_MLP_LIMBS)
            b.attn.exact = b.mlp.exact = VISION_EXACT
        self.visual.merger.merger.precise = on
        self.visual.merger.merger.ln_q.precise = on
        self.visual.merger.merger.limbs = vl
        self.visual.merger.merger.exact = VISION_EXACT
        for lyr in self.text_model.layers:
            lyr.self_attn.precise = lyr.mlp.precise = on
            # the LM is well conditioned: 2-limb dense products hold it at PCC ~1 - 2e-6 on HF inputs,
            # so it skips the 8-lane exact reduction (which would cost ~3x the text-encode time)
            lyr.self_attn.exact = lyr.mlp.exact = LM_EXACT
            lyr.input_layernorm.precise = lyr.post_attention_layernorm.precise = on
            lyr.precise_inputs = on
        self.text_model.norm.precise = on
        if on:  # upload the float32 norm weights now (not lazily inside the first forward)
            norms = [self.text_model.norm, self.visual.merger.merger.ln_q]
            for blk in self.visual.blocks:
                norms += [blk.block.norm1, blk.block.norm2]
            for lyr in self.text_model.layers:
                norms += [lyr.input_layernorm, lyr.post_attention_layernorm]
            for n in norms:
                n._ensure_w32()

    # ---- input encoding -> device (outside the forward) ------------------------------------------
    def prepare(self, enc):
        d = self.device
        cond, uncond = enc.cond, enc.uncond
        B = cond["input_ids"].shape[0]
        grids = [tuple(int(v) for v in g) for g in cond["image_grid_thw"].tolist()]
        assert len(set(grids)) == 1 and torch.equal(cond["image_grid_thw"], uncond["image_grid_thw"])
        assert torch.equal(cond["pixel_values"], uncond["pixel_values"])

        p = TextEncoderInputs()
        p.B = B
        # vision: [N, 1, s_pad, 1176], the images split over the mesh rows (each row's TP group runs
        # its own B / rows images; the rows otherwise repeat the same tower on the same images)
        rows = tuple(d.shape)[VISION_DP_AXIS]
        p.vision_dp = rows if B % rows == 0 else 1
        vc = self.visual.consts(grids[0], B // p.vision_dp)
        pix = cond["pixel_values"].to(torch.float32).reshape(B, 1, vc.s, -1)
        pix = torch.nn.functional.pad(pix, (0, 0, 0, vc.s_pad - vc.s))
        p.vision_consts = vc
        if p.vision_dp > 1:
            dims = [None, None]
            dims[VISION_DP_AXIS] = 0
            mapper = ttnn.ShardTensor2dMesh(d, mesh_shape=tuple(d.shape), dims=tuple(dims))
            p.pixels = ttnn.from_torch(
                pix.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d, mesh_mapper=mapper
            )
        else:
            p.pixels = _replicated(d, pix, ttnn.float32)

        # text: prompts (row 0) and negatives (row 1) right-padded to one length
        pad_id = 151643
        lc, lu = int(cond["attention_mask"].sum(1).max()), int(uncond["attention_mask"].sum(1).max())
        L = max(cond["input_ids"].shape[1], uncond["input_ids"].shape[1])
        s_pad = pad_to_tile(L)

        def _padded(o):
            ids = torch.full((B, s_pad), pad_id, dtype=torch.int64)
            am = torch.zeros((B, s_pad), dtype=torch.int64)
            ids[:, : o["input_ids"].shape[1]] = o["input_ids"]
            am[:, : o["attention_mask"].shape[1]] = o["attention_mask"]
            return ids, am

        ids_c, am_c = _padded(cond)
        ids_u, am_u = _padded(uncond)
        ids = torch.cat([ids_c, ids_u], 0)
        am = torch.cat([am_c, am_u], 0)
        # the <|image_pad|> run must sit at one position in every sequence (shared template prefix)
        pos = [(r == self.image_token_id).nonzero().flatten() for r in ids]
        p0, m = int(pos[0][0]), int(pos[0].numel())
        assert all(int(q[0]) == p0 and q.numel() == m for q in pos) and m == vc.m, "image token run differs"
        p.img_start, p.img_len, p.s_pad = p0, m, s_pad
        # one batch of 2B sequences: prompts [0, B), negative prompts [B, 2B)
        p.ids = _replicated(d, ids.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # causal + key-padding additive mask (GQA row packing), per sequence
        mask = text_attention_mask(am, 2 * B, s_pad, s_pad, self.group)
        p.mask = _replicated(d, torch.from_numpy(mask), ttnn.float32)
        # HF passes position_ids=None -> arange on all three mRoPE axes
        posn = torch.arange(s_pad, dtype=torch.float32).reshape(1, 1, s_pad, 1).expand(3, 1, s_pad, 1)
        p.positions = _replicated(d, posn, ttnn.float32)

        # output: drop the 64 template tokens, keep up to the longest valid sequence, zero the padding
        drop = PROMPT_TEMPLATE_DROP
        p.len_cond, p.len_uncond = lc - drop, lu - drop
        mc = am_c[:, drop:lc].to(torch.float32)
        mu = am_u[:, drop:lu].to(torch.float32)
        p.keep_cond = _replicated(d, mc.reshape(B, lc - drop, 1), ttnn.float32)
        p.keep_uncond = _replicated(d, mu.reshape(B, lu - drop, 1), ttnn.float32)
        # shared prefix: prompts and negatives agree on their first `first_diff` tokens (template +
        # image), so the negatives' first P (whole tiles, >= the dropped template) hidden states are the
        # prompts'; the negatives run only [P, P + suffix) against the prompts' cached prefix K/V
        n = min(ids_c.shape[1], ids_u.shape[1])
        diff = (ids_c[:, :n] != ids_u[:, :n]).any(0).nonzero()
        first_diff = int(diff[0]) if diff.numel() else n
        P = (first_diff // 32) * 32
        p.prefix = None
        if PREFIX_SHARE and drop <= P < lu and bool((am_c[:, :P] == am_u[:, :P]).all()):
            su = pad_to_tile(lu - P)
            if P + su <= s_pad:
                q = np.arange(P, P + su)[:, None]
                keys = np.arange(P + su)[None, :]
                causal = np.where(keys <= q, 0.0, MASK_NEG).astype(np.float32)
                pad = np.where(am_u[:, : P + su].numpy().astype(bool), 0.0, MASK_NEG).astype(np.float32)
                m = np.maximum(causal[None, None] + pad[:, None, None, :], MASK_NEG)
                m = np.tile(m, (1, 1, self.group, 1))
                p.prefix = (P, su)
                p.mask_suffix = _replicated(d, torch.from_numpy(m), ttnn.float32)
        # QwenImageEditPipeline.encode_prompt: an all-ones mask is dropped (None)
        p.mask_cond = None if bool(mc.all()) else mc
        p.mask_uncond = None if bool(mu.all()) else mu
        return p

    # ---- device forward --------------------------------------------------------------------------
    def _rope(self, positions):
        cos3, sin3 = self.rotary(None, positions, dtype=ttnn.float32)
        return self._mrope(cos3), self._mrope(sin3)

    def _mrope(self, t3):
        """[3, 1, S, D] per-axis tables -> [1, 1, S, D] (HF apply_multimodal_rotary_pos_emb section select)."""
        bounds = np.cumsum([0] + self.mrope_section * 2).tolist()
        a, b, s, dim = t3.shape
        parts = []
        for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            ax = i % 3
            parts.append(ttnn.slice(t3, [ax, 0, 0, lo], [ax + 1, b, s, hi]))
        return ttnn.concat(parts, dim=-1)

    def encode_vision(self, p):
        """Vision tower over all B condition images -> image embeddings [B, m, 3584] (replicated)."""
        _, img = self.visual.forward_batched(p.pixels, p.vision_consts)
        if p.vision_dp > 1:  # each row's images -> the whole batch on every chip (exact data movement)
            img = ttnn.all_gather(img, dim=0, cluster_axis=VISION_DP_AXIS, num_links=1, topology=ttnn.Topology.Linear)
        return img

    def __call__(self, p):
        """-> (prompt_embeds [B, Lc, 3584], negative_prompt_embeds [B, Lu, 3584]) fp32, replicated."""
        return self.encode_text(p, self.encode_vision(p))

    def encode_text(self, p, img):
        """Token embed + splice the image embeddings + 28 LM layers, for the prompts [0, B) and the
        negative prompts [B, 2B). The 2B sequences run as TEXT_CHUNKS programs of B each: the text-encode
        peak must fit beside the 896 MB trace region that the precise denoise trace needs. Measured at
        B=32: one 64-sequence program ran out of DRAM (a 704 MB allocation) with that region; two
        32-sequence programs fit."""
        tok = self.text_model.embed_tokens(p.ids)  # [2B, s_pad, 3584]
        B2, s_pad, C = tok.shape[0], tok.shape[1], tok.shape[2]
        B = B2 // 2
        a, m = p.img_start, p.img_len
        # HF: inputs_embeds.masked_scatter(image_mask, image_embeds) in the embeddings' dtype (float32 in
        # the golden): splice in float32 (the bf16 table values are exact in float32)
        tok = ttnn.typecast(tok, ttnn.float32)
        img = ttnn.typecast(img, ttnn.float32)
        cos, sin = self._rope(p.positions)
        drop = PROMPT_TEMPLATE_DROP
        prefix = getattr(p, "prefix", None)
        kv = {"mode": "store", "P": prefix[0], "cache": {}} if prefix else None
        outs = []
        h_cond = None
        for lo, L, keep in ((0, p.len_cond, p.keep_cond), (B, p.len_uncond, p.keep_uncond)):
            t = ttnn.slice(tok, [lo, 0, 0], [lo + B, s_pad, C])
            x = ttnn.concat(
                [ttnn.slice(t, [0, 0, 0], [B, a, C]), img, ttnn.slice(t, [0, a + m, 0], [B, s_pad, C])], dim=1
            )  # sequence lo + b uses condition image b
            x = ttnn.reshape(x, (B, 1, s_pad, C))
            if lo and prefix:
                # negatives: run [P, P + su) only, attending to the prompts' cached prefix K/V
                P, su = prefix
                xs = ttnn.slice(x, [0, 0, P, 0], [B, 1, P + su, C])
                rows = lambda r: ttnn.slice(r, [0, 0, P, 0], list(r.shape)[:2] + [P + su, r.shape[-1]])  # noqa: E731
                kv["mode"] = "use"
                hs = self.text_model.forward_padded(xs, rows(cos), rows(sin), p.mask_suffix, kv=kv)
                h = ttnn.concat(
                    [ttnn.slice(h_cond, [0, 0, drop, 0], [B, 1, P, C]), ttnn.slice(hs, [0, 0, 0, 0], [B, 1, drop + L - P, C])],
                    dim=2,
                )
                kv["cache"].clear()
            else:
                mask = ttnn.slice(p.mask, [lo, 0, 0, 0], [lo + B] + list(p.mask.shape)[1:])
                h_full = self.text_model.forward_padded(x, cos, sin, mask, kv=kv)  # [B, 1, s_pad, C] replicated
                h_cond = h_full if prefix else None
                h = ttnn.slice(h_full, [0, 0, drop, 0], [B, 1, drop + L, C])
            outs.append(ttnn.multiply(ttnn.reshape(h, (B, L, C)), keep))
            ttnn.deallocate(x)
        return outs[0], outs[1]
