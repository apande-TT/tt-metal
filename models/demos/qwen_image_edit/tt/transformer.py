# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QwenImageTransformer2DModel.forward chained from the graduated ports (one denoising forward).

    hs   = img_in(cat[latents, image_latents])                       patch_embed port
    txt  = txt_in(txt_norm(prompt_embeds))                           float32 RMSNorm + patch_embed port
    temb = qwen_timestep_proj_embeddings(t)                          timesteps -> timestep_embedding
    rope = qwen_embed_rope(img_shapes, txt_len)                      constant for a call: built in prepare
    txt, hs = qwen_image_transformer_block_i(hs, txt, temb, rope)    x N (feed_forward inside each block)
    out  = proj_out(ada_layer_norm_continuous(hs, temb))             decoder_head port

TP=8 over the whole mesh: the block ports shard over every device (their collectives run axis 1 then
axis 0), the small projections are replicated.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import (
    _precise,
    ada_layer_norm_continuous,
    qwen_embed_rope,
    qwen_image_transformer_block,
    qwen_timestep_proj_embeddings,
)
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.decoder_head import TtQwenDecoderHead
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.patch_embed import TtQwenPatchEmbed


def _replicated(device, t, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


class TtRMSNorm32:
    """diffusers RMSNorm (txt_norm) in float32: x * rsqrt(mean(x^2) + eps) * w."""

    def __init__(self, device, torch_module):
        self.eps = float(torch_module.eps)
        self.w = _replicated(device, torch_module.weight.detach().to(torch.float32).reshape(1, -1))
        self.cfg = _precise.precise_config()

    def __call__(self, x):
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True, compute_kernel_config=self.cfg)
        return ttnn.multiply(ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.eps))), self.w)


class TtQwenImageTransformer:
    def __init__(self, device, hf_transformer, layers=None, tracker=None, precise=True):
        self.device = device
        tr = hf_transformer
        assert not getattr(tr, "zero_cond_t", False), "zero_cond_t checkpoints are not ported"
        assert not tr.config.guidance_embeds, "guidance-distilled checkpoints are not ported"
        blocks = list(tr.transformer_blocks)
        self.num_layers = len(blocks) if layers is None else max(1, min(int(layers), len(blocks)))
        # precise mode of the ports (see _stubs/_precise.py): float32 matmul inputs as bf16 limbs, exact-lane
        # QK^T with q / k as 3 limbs (the ~4e4 attention logits feed a near-argmax softmax) and exact-lane
        # linears. Measured per-step velocity error vs a float64 HF forward (step 45, 8 seeds, golden
        # conditioning): dense linears + 2-limb QK 5.6e-4..2.5e-3; this 1.1e-4..1.5e-3; HF float32 itself
        # 2.1e-5..6.1e-4.
        _precise.ENABLED = bool(precise)
        if precise:
            _precise.QK_LIMBS = 3
            _precise.EXACT_LINEAR = True
        self.img_in = TtQwenPatchEmbed(device, tr.img_in)
        self.txt_norm = TtRMSNorm32(device, tr.txt_norm)
        self.txt_in = TtQwenPatchEmbed(device, tr.txt_in)
        self.time_text_embed = qwen_timestep_proj_embeddings.build(device, tr.time_text_embed)
        self.pos_embed = qwen_embed_rope.build(device, tr.pos_embed)
        self.transformer_blocks = [qwen_image_transformer_block.build(device, b) for b in blocks[: self.num_layers]]
        self.norm_out = ada_layer_norm_continuous.build(device, tr.norm_out)
        self.proj_out = TtQwenDecoderHead(device, tr.proj_out)
        self.out_channels = int(tr.config.out_channels) * int(tr.config.patch_size) ** 2
        if tracker is not None:
            tracker.track("qwen_timestep_proj_embeddings", self.time_text_embed)
            tracker.track("timesteps", self.time_text_embed.time_proj)
            tracker.track("timestep_embedding", self.time_text_embed.timestep_embedder)
            tracker.track("qwen_embed_rope", self.pos_embed)
            tracker.track("ada_layer_norm_continuous", self.norm_out)
            for blk in self.transformer_blocks:
                tracker.track("qwen_image_transformer_block", blk)
                inner = blk.stack.blocks[0]
                tracker.track("feed_forward", inner.img_ff)
                tracker.track("feed_forward", inner.txt_ff)

    def rotary(self, img_shapes, txt_len):
        """(img [S_img, 128], txt [L, 128]) float32 [cos | sin] tables for one call's shapes."""
        # HF passes img_shapes per batch entry ([[noise_shape, image_shape]] * B) and reads entry 0
        return self.pos_embed([list(img_shapes)], max_txt_seq_len=int(txt_len))

    def __call__(self, latent_in, timestep, prompt_embeds, rotary):
        """latent_in [B, S, 64] fp32, timestep [B, 1] fp32 (= t / 1000), prompt_embeds [B, L, 3584] fp32
        -> [B, S, 64] fp32 (the forward's `sample`)."""
        hs = self.img_in(latent_in)
        txt = self.txt_in(self.txt_norm(prompt_embeds))
        temb = self.time_text_embed(timestep, hs)
        for blk in self.transformer_blocks:
            txt, hs = blk(hidden_states=hs, encoder_hidden_states=txt, temb=temb, image_rotary_emb=rotary)
        return self.proj_out(self.norm_out(hs, temb))
