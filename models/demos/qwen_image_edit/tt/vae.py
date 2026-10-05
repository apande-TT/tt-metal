# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AutoencoderKLQwenImage encode / decode as QwenImageEditPipeline uses them, on the graduated VAE ports.

encode (prepare_latents -> _encode_vae_image, sample_mode="argmax"):
    z = quant_conv(qwen_image_encoder3d(image))[:, :z_dim]          the Gaussian's mode is its mean
    z = (z - latents_mean) / latents_std -> 2x2 pack -> [B, (h/2)(w/2), 4 z_dim]
decode (the __call__ tail):
    z = unpack(latents) * latents_std + latents_mean
    image = clamp(qwen_image_decoder3d(post_quant_conv(z)), -1, 1)[:, :, 0] -> clamp(x / 2 + 0.5, 0, 1)

The encoder / decoder ports route every child of their Wan stack through its graduated port (causal
conv, RMS norm, residual / mid / attention / up block, resample with the zero_pad2d / upsample op
ports). They run W-parallel over the mesh columns and batch-parallel over the rows. quant_conv and
post_quant_conv are 1x1x1 convs, i.e. a channel matmul: here with exact-lane float32 products,
because the decoder amplifies latent error and the image latents condition every denoising step. Both stacks run in float32 with their ports' precise modes on.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import split_linear
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs import qwen_image_decoder3d, qwen_image_encoder3d
from models.tt_dit.utils.conv3d import _walk_conv3d_modules

_PORT_NAMES = {
    "TtQwenImageCausalConv3d": "qwen_image_causal_conv3d",
    "TtQwenImageRMSNorm": "qwen_image_r_m_s",
    "TtQwenImageResidualBlock": "qwen_image_residual_block",
    "TtQwenImageMidBlock": "qwen_image_mid_block",
    "TtQwenImageAttentionBlock": "qwen_image_attention_block",
    "TtQwenImageUpBlock": "qwen_image_up_block",
    "TtQwenImageResample": "qwen_image_resample",
}


def _replicated(device, t, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


def pack_latents(z, B, C, H, W):
    """[B, C, H, W] -> [B, (H/2)(W/2), 4C] (QwenImageEditPipeline._pack_latents)."""
    z = ttnn.reshape(z, (B, C, H // 2, 2, W // 2, 2))
    z = ttnn.permute(z, (0, 2, 4, 1, 3, 5))
    return ttnn.reshape(z, (B, (H // 2) * (W // 2), C * 4))


def unpack_latents(x, B, C, H, W):
    """[B, (H/2)(W/2), 4C] -> [B, C, H, W] (QwenImageEditPipeline._unpack_latents)."""
    x = ttnn.reshape(x, (B, H // 2, W // 2, C, 2, 2))
    x = ttnn.permute(x, (0, 3, 1, 4, 2, 5))
    return ttnn.reshape(x, (B, C, H, W))


class PointwiseConv32:
    """A 1x1x1 QwenImageCausalConv3d as a float32 channel matmul on [B, C, H, W]: exact-lane products
    (the dense 3-term split measured 2.4e-4 relative on the image latents here; exact-lane is ~float32)."""

    def __init__(self, device, torch_module):
        w = torch_module.weight.detach().to(torch.float32)
        assert tuple(w.shape[2:]) == (1, 1, 1), f"expected a 1x1x1 conv, got {tuple(w.shape)}"
        self.cout = w.shape[0]
        self.w = _replicated(device, w.reshape(self.cout, -1).t(), dtype=ttnn.bfloat16)  # [C_in, C_out], exact
        self.b = _replicated(device, torch_module.bias.detach().to(torch.float32).reshape(1, 1, -1))

    def __call__(self, x):
        B, C, H, W = x.shape
        t = ttnn.reshape(ttnn.permute(x, (0, 2, 3, 1)), (B, H * W, C))
        y = split_linear(t, self.w, bias=self.b, exact=True, limbs=2)
        return ttnn.permute(ttnn.reshape(y, (B, H, W, self.cout)), (0, 3, 1, 2))


def _block_fp32_convs(stack, h_block=8, w_block=4):
    """Give a float32 Wan stack's conv3d layers a real output blocking.

    get_conv3d_config has no float32 entry for the VAE's shapes, so every float32 conv falls back to
    (C_in 32, C_out 32, T 1, H 1, W 1): one output pixel per work unit. Only the OUTPUT blocking moves
    here (C_out / H / W); C_in_block stays, so the prepared weights and the reduction order of every
    output element are unchanged."""
    n = 0
    for m in _walk_conv3d_modules(stack):
        c = m.conv_config
        if m.dtype != ttnn.float32 or (c.T_out_block, c.H_out_block, c.W_out_block) != (1, 1, 1):
            continue
        cout = m.out_channels
        c.C_out_block = next(b for b in (96, 64, 32) if cout % b == 0) if cout % 32 == 0 else c.C_out_block
        c.H_out_block, c.W_out_block = h_block, w_block
        n += 1
    return n


class TtQwenVAE:
    def __init__(self, device, hf_vae, tracker=None):
        self.device = device
        cfg = hf_vae.config
        self.z_dim = int(cfg.z_dim)
        rows = tuple(device.shape)[0] if isinstance(device, ttnn.MeshDevice) else 1
        self.batch_factor = rows
        # float32 encoder: the image latents condition every denoising step, and the bf16 encoder's
        # error (image-latent PCC 0.99999) moved the final latents ~10x more than the denoiser's own
        self.encoder = qwen_image_encoder3d.build(device, hf_vae.encoder, batch_parallel=True, dtype=ttnn.float32)
        self.decoder = qwen_image_decoder3d.build(device, hf_vae.decoder, batch_parallel=True)
        _block_fp32_convs(self.encoder.encoder)
        _block_fp32_convs(self.decoder.decoder)
        self.quant_conv = PointwiseConv32(device, hf_vae.quant_conv)
        self.post_quant_conv = PointwiseConv32(device, hf_vae.post_quant_conv)
        mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).reshape(1, self.z_dim, 1, 1)
        std = torch.tensor(cfg.latents_std, dtype=torch.float32).reshape(1, self.z_dim, 1, 1)
        self.mean, self.std, self.inv_std = (_replicated(device, t) for t in (mean, std, 1.0 / std))
        # Images per decoder program. Re-measured 2026-10-03 on this 2x4 T3K mesh with every stage's weights
        # resident (7.95 GB per chip after text encode, B=32): 32 in one program fails allocating a 201 MB
        # DRAM buffer (16.8 MB per bank needed, largest free block 10.1 MB); 16 fits (9.01 GB per chip).
        # 8 (2026-10-05): the precise convs now hold their padded input's tile copy across their runs.
        self.decode_max_batch = 8
        # Images per encoder program. Re-measured 2026-10-03, same setup: 32 in one program fails allocating a
        # 50 MB buffer (4.19 MB per bank needed, largest free block 3.30 MB); 16 fits (8.92 GB per chip).
        # 8 (2026-10-04): the exact convs now hold the padded input's bf16 limbs across their lane runs,
        # which at 16 ran the eager e2e out of memory; 8 per program costs ~1% and the shared limbs save ~7%.
        self.encode_max_batch = 8
        # encoder convs exact-lane (its image latents condition every denoising step), decoder convs median
        self.precise_convs = self._precise_ports(self.encoder, "encoder", hf_vae.encoder, conv_mode="exact")
        self.precise_convs += self._precise_ports(self.decoder, "decoder", hf_vae.decoder, conv_mode="median")
        if tracker is not None:
            tracker.track("qwen_image_encoder3d", self.encoder)
            tracker.track("qwen_image_decoder3d", self.decoder)
            for port in list(self.encoder.ports) + list(self.decoder.ports):
                name = _PORT_NAMES[type(port).__name__]
                tracker.track(name, port, ("forward_sharded",))
                if name == "qwen_image_resample":
                    if port.zero_pad is not None:
                        tracker.track("zero_pad2d", port.zero_pad, ("forward_sharded",))
                    if port.upsample is not None:
                        tracker.track("qwen_image_upsample", port.upsample, ("forward_sharded",))

    @staticmethod
    def _precise_ports(stack_port, attr, hf_stack, conv_mode="median"):
        """A float32 stack's ports in their precise modes: channel norms spelled out in float32, attention
        with float32 products, causal convs as the median of three channel orders (see the ports). The HF module of each conv is
        found by its path, which the Wan stack mirrors (it loads the HF state dict by those names)."""
        by_body = {id(getattr(p, p.BODY_ATTR)): p for p in stack_port.ports}

        def named(mod, prefix=""):
            yield prefix, mod
            for name, child in mod.named_children():
                yield from named(child, f"{prefix}.{name}" if prefix else name)

        n_conv = 0
        for path, mod in named(getattr(stack_port, attr)):
            port = by_body.get(id(mod))
            if port is None:
                continue
            kind = type(port).__name__
            if kind in ("TtQwenImageRMSNorm", "TtQwenImageAttentionBlock"):
                port.precise = True
            elif kind in ("TtQwenImageResidualBlock", "TtQwenImageResample"):
                n_conv += int(port.enable_precise(mode=conv_mode))  # shortcut linear / spatial convs
            elif kind == "TtQwenImageCausalConv3d":
                # the Wan stack renames upsamplers.0 -> upsamplers when it loads the HF state dict
                ref = hf_stack.get_submodule(path.replace("upsamplers.", "upsamplers.0."))
                assert tuple(ref.weight.shape[:2]) == (mod.out_channels, mod.unpadded_in_channels), path
                n_conv += bool(port.enable_precise(ref, mode=conv_mode))
        return n_conv

    # the repeated blocks, as the plain lists the ports hold (for depth discovery)
    @property
    def stacks(self):
        return {
            "encoder.down_blocks": self.encoder.down_blocks,
            "encoder.mid_block.resnets": self.encoder.mid_block_resnets,
            "decoder.up_blocks": self.decoder.up_blocks,
            "decoder.mid_block.resnets": self.decoder.mid_block_resnets,
        }

    def _ccl_managers(self):
        seen = []
        for m in (self.encoder, self.decoder):
            c = getattr(m, "ccl_manager", None)
            if c is not None and all(c is not s for s in seen):
                seen.append(c)
        return seen

    def release_buffers(self):
        """Free the halo ping-pong buffers the Wan convs cache per input shape (they are rebuilt on the
        next call). Called between stages so the VAE's per-shape scratch does not stay resident."""
        for c in self._ccl_managers():
            cache = getattr(c, "_ping_pong_buffer_cache", {})
            for key, bufs in list(cache.items()):
                if isinstance(key, tuple) and key and key[0] == "np":
                    for b in bufs:
                        ttnn.deallocate(b)
                    del cache[key]
                    getattr(c, "_ping_pong_buffer_indices", {}).pop(key, None)

    def encode(self, image):
        """image [B, 3, 1, H, W] in [-1, 1] -> packed image latents [B, S, 64] fp32 (encode_max_batch images
        per program, concatenated)."""
        B = image.shape[0]
        mb = B if not self.encode_max_batch else int(self.encode_max_batch)
        if mb < B:
            s = list(image.shape)
            return ttnn.concat(
                [self.encode(ttnn.slice(image, [lo, 0, 0, 0, 0], [min(B, lo + mb)] + s[1:])) for lo in range(0, B, mb)],
                dim=0,
            )
        enc = self.encoder(ttnn.typecast(image, self.encoder.dtype))  # [B, 2 z, 1, h, w]
        h, w = enc.shape[3], enc.shape[4]
        enc = self.quant_conv(ttnn.typecast(ttnn.reshape(enc, (B, enc.shape[1], h, w)), ttnn.float32))
        z = ttnn.slice(enc, (0, 0, 0, 0), (B, self.z_dim, h, w))
        z = ttnn.multiply(ttnn.subtract(z, self.mean), self.inv_std)
        return pack_latents(z, B, self.z_dim, h, w)

    def decode(self, latents, h, w):
        """packed latents [B, S, 64] fp32 -> image [B, 3, H, W] in [0, 1] fp32.

        decode_max_batch caps the images per decoder program (the float32 decoder carries the widest
        activations of the pipeline); the programs run back to back and are concatenated."""
        B = latents.shape[0]
        mb = B if not self.decode_max_batch else int(self.decode_max_batch)
        if mb >= B:
            return self._decode(latents, h, w)
        outs = []
        for lo in range(0, B, mb):
            hi = min(B, lo + mb)
            outs.append(self._decode(ttnn.slice(latents, [lo, 0, 0], [hi, latents.shape[1], latents.shape[2]]), h, w))
        return ttnn.concat(outs, dim=0)

    def _decode(self, latents, h, w):
        B = latents.shape[0]
        z = unpack_latents(latents, B, self.z_dim, h, w)
        z = self.post_quant_conv(ttnn.add(ttnn.multiply(z, self.std), self.mean))
        out = self.decoder(ttnn.reshape(z, (B, self.z_dim, 1, h, w)))  # [B, 3, 1, H, W]
        H, W = out.shape[3], out.shape[4]
        out = ttnn.clamp(ttnn.reshape(ttnn.typecast(out, ttnn.float32), (B, 3, H, W)), -1.0, 1.0)
        return ttnn.clamp(ttnn.add(ttnn.multiply(out, 0.5), 0.5), 0.0, 1.0)
