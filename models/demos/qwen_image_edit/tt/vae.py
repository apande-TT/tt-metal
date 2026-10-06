# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AutoencoderKLQwenImage encode / decode as used by QwenImageEditPipeline, on the graduated VAE ports.

encode (prepare_latents -> _encode_vae_image, sample_mode="argmax"):
    z = quant_conv(qwen_image_encoder3d(image))[:, :z_dim]         (the Gaussian's mode = its mean)
    z = (z - latents_mean) / latents_std  -> 2x2 pack -> [B, (h/2)(w/2), 4 z_dim]
decode (__call__ tail):
    z = unpack(latents) * latents_std + latents_mean
    image = clamp(qwen_image_decoder3d(post_quant_conv(z)), -1, 1)[:, :, 0]  -> (image + 1) / 2
"""
from __future__ import annotations

import contextlib

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs import mlp as vae_pointwise
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs import qwen_image_decoder3d, qwen_image_encoder3d
from models.tt_dit.utils.conv3d import get_conv3d_config


def _replicated(device, t, dtype=ttnn.float32):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _host_conv3d_weights(w, c_in_block, alignment=32):
    """ttnn.experimental.prepare_conv3d_weights (groups=1) on host: [O, I, kD, kH, kW] -> [-1, O] with
    I zero-padded to `alignment` and split into c_in_block row blocks (permute (2,3,4,1,0), pad, reshape,
    permute (3,0,1,2,4,5), reshape) -- the same layout, without the device round trip at load."""
    w = w.permute(2, 3, 4, 1, 0)
    c = w.shape[3]
    if c % alignment:
        zeros = torch.zeros(*w.shape[:3], alignment - c % alignment, w.shape[4], dtype=w.dtype)
        w = torch.cat([w, zeros], dim=3)
    kd, kh, kw, ca, o = w.shape
    assert ca % c_in_block == 0, (ca, c_in_block)
    w = w.reshape(kd, kh, kw, ca // c_in_block, c_in_block, o).permute(3, 0, 1, 2, 4, 5)
    return w.reshape(-1, o).contiguous()


@contextlib.contextmanager
def _host_weight_prep():
    """While the VAE ports load, the Wan conv layers prepare their weights on host (bit-identical layout)
    instead of uploading each one, permuting it on device and reading it back. Other users of these
    classes are untouched (restored on exit)."""
    from models.tt_dit.models.vae import vae_wan2_1 as wan

    dtypes = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
    saved = {}

    def _make(cls, unsqueeze):
        orig = cls._prepare_torch_state

        def _prepare_torch_state(self, state):
            tdt = dtypes.get(self.dtype)
            if "weight" not in state or tdt is None or not self.conv_config.C_in_block:
                return orig(self, state)
            w = state["weight"]
            w = w.unsqueeze(2) if unsqueeze else w
            # from_torch(dtype) rounds before the permutes; rounding commutes with them
            state["weight"] = _host_conv3d_weights(w.to(tdt), self.conv_config.C_in_block)
            if "bias" in state:
                state["bias"] = state["bias"].reshape(1, -1)

        return orig, _prepare_torch_state

    for cls, unsqueeze in ((wan.WanCausalConv3d, False), (wan.WanConv2d, True)):
        saved[cls], cls._prepare_torch_state = _make(cls, unsqueeze)
    try:
        yield
    finally:
        for cls, orig in saved.items():
            cls._prepare_torch_state = orig


# weights of the VAE's Wan conv layers prepared on host at load (False: ttnn's device prepare)
HOST_WEIGHT_PREP = True


def _decoder_blocking(in_channels, out_channels, kernel):
    """conv3d blocking for the decoder's folded float32 (1, 3, 3) convs: 96-channel input and output blocks
    and an 8 x 4 output patch per core (the fp32 table's default is 32 / 32 / 1 x 1, a single output
    position)."""
    if tuple(kernel) != (1, 3, 3):
        return None
    return (96 if in_channels % 96 == 0 else 32, 96 if out_channels % 96 == 0 else 32, 1, 8, 4)


def _encoder_blocking(in_channels, out_channels, kernel):
    """conv3d blocking for the encoder's folded bf16 (1, 3, 3) convs: the tuned blocking of the same channels'
    (3, 3, 3) conv (1/3 of its K, so it fits L1 too); the bf16 table has no (1, 3, 3) entries for these
    channels and its default is a single output position per core."""
    if tuple(kernel) != (1, 3, 3):
        return None
    c = get_conv3d_config(in_channels, out_channels, (3, 3, 3), ttnn.bfloat16, grid_size=(1, 1))
    return (c.C_in_block, c.C_out_block, c.T_out_block, c.H_out_block, c.W_out_block)


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


class TtQwenVAE:
    def __init__(self, device, hf_vae, tracker=None):
        self.device = device
        cfg = hf_vae.config
        self.z_dim = int(cfg.z_dim)
        with _host_weight_prep() if HOST_WEIGHT_PREP else contextlib.nullcontext():
            # the edit image is a single frame too: the encoder's causal convs keep only their live temporal
            # tap, so neither their T front pad nor its zero frames' MACs run
            self.encoder = qwen_image_encoder3d.build(
                device, hf_vae.encoder, batch_parallel=True, single_frame=True, blocking=_encoder_blocking, ccl_links=2
            )
            self.quant_conv = vae_pointwise.build(device, hf_vae.quant_conv)
            self.post_quant_conv = vae_pointwise.build(device, hf_vae.post_quant_conv)
            # image latents are single-frame: the decoder's causal convs keep only their live temporal tap
            self.decoder = qwen_image_decoder3d.build(
                device,
                hf_vae.decoder,
                batch_parallel=True,
                single_frame=True,
                blocking=_decoder_blocking,
                ccl_links=2,  # the decoder's gathers / halo exchanges over 2 fabric links instead of 1
            )
        mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).reshape(1, self.z_dim, 1, 1)
        std = torch.tensor(cfg.latents_std, dtype=torch.float32).reshape(1, self.z_dim, 1, 1)
        self.mean = _replicated(device, mean)
        self.std = _replicated(device, std)
        self.inv_std = _replicated(device, 1.0 / std)
        self.release_after_stage = True  # eager: free the per-shape halo buffers after each VAE stage
        if tracker is not None:
            import importlib

            tracker.track("qwen_image_encoder3d", self.encoder, stub_module=qwen_image_encoder3d)
            tracker.track("qwen_image_decoder3d", self.decoder, stub_module=qwen_image_decoder3d)
            names = {
                "TtQwenImageCausalConv3d": "qwen_image_causal_conv3d",
                "TtQwenImageRMSNorm": "qwen_image_r_m_s",
                "TtQwenImageResidualBlock": "qwen_image_residual_block",
                "TtQwenImageMidBlock": "qwen_image_mid_block",
                "TtQwenImageAttentionBlock": "qwen_image_attention_block",
                "TtQwenImageUpBlock": "qwen_image_up_block",
                "TtQwenImageResample": "qwen_image_resample",
            }
            stub_pkg = "models.tt_dit.pipelines.qwen_image_edit_vae._stubs."
            for port in list(self.encoder.ports) + list(self.decoder.ports):
                name = names[type(port).__name__]
                tracker.track(name, port, ("forward_sharded",), stub_module=importlib.import_module(stub_pkg + name))
                if name == "qwen_image_resample":
                    if port.zero_pad is not None:
                        tracker.track(
                            "zero_pad2d",
                            port.zero_pad,
                            ("forward_sharded",),
                            stub_module=importlib.import_module(stub_pkg + "zero_pad2d"),
                        )
                    if port.upsample is not None:
                        tracker.track(
                            "qwen_image_upsample",
                            port.upsample,
                            ("forward_sharded",),
                            stub_module=importlib.import_module(stub_pkg + "qwen_image_upsample"),
                        )

    def _ccl_managers(self):
        seen = []
        for m in (self.encoder, self.decoder, self.quant_conv, self.post_quant_conv):
            c = getattr(m, "ccl_manager", None)
            if c is not None and all(c is not s for s in seen):
                seen.append(c)
        return seen

    def release_buffers(self):
        """Free the halo-exchange ping-pong buffers the Wan convs cache per shape (2 per padded conv
        input shape). They are rebuilt on the next call; keeping them costs GBs at full resolution."""
        for c in self._ccl_managers():
            cache = getattr(c, "_ping_pong_buffer_cache", {})
            for key, bufs in list(cache.items()):
                if isinstance(key, tuple) and key and key[0] == "np":
                    for b in bufs:
                        ttnn.deallocate(b)
                    del cache[key]
                    getattr(c, "_ping_pong_buffer_indices", {}).pop(key, None)

    def encode(self, image):
        """image [B, 3, 1, H, W] in [-1, 1] (replicated, TILE) -> packed image latents [B, S, 64] fp32."""
        B, _, _, H, W = image.shape
        enc = self.encoder(ttnn.typecast(image, ttnn.bfloat16))  # bf16 port -> [B, 2 z, 1, h, w]
        enc = self.quant_conv(enc)
        h, w = enc.shape[3], enc.shape[4]
        z = ttnn.slice(enc, (0, 0, 0, 0, 0), (B, self.z_dim, 1, h, w))
        z = ttnn.typecast(ttnn.reshape(z, (B, self.z_dim, h, w)), ttnn.float32)
        z = ttnn.multiply(ttnn.subtract(z, self.mean), self.inv_std)
        out = pack_latents(z, B, self.z_dim, h, w)
        if self.release_after_stage:
            self.release_buffers()
        return out

    def decode(self, latents, h, w, max_batch=None):
        """packed latents [B, S, 64] fp32 -> image [B, 3, H, W] in [0, 1] fp32.

        max_batch: decode at most this many images per program (the fp32 decoder at full resolution is
        the widest activation of the pipeline); chunks run back to back and are concatenated."""
        B = latents.shape[0]
        mb = B if not max_batch else int(max_batch)
        if mb >= B:
            out = self._decode(latents, h, w)
        else:
            outs = []
            for lo in range(0, B, mb):
                hi = min(B, lo + mb)
                outs.append(
                    self._decode(ttnn.slice(latents, [lo, 0, 0], [hi, latents.shape[1], latents.shape[2]]), h, w)
                )
            out = ttnn.concat(outs, dim=0)
        if self.release_after_stage:
            self.release_buffers()
        return out

    def _decode(self, latents, h, w):
        B = latents.shape[0]
        z = unpack_latents(latents, B, self.z_dim, h, w)
        z = ttnn.add(ttnn.multiply(z, self.std), self.mean)
        z = ttnn.reshape(z, (B, self.z_dim, 1, h, w))
        x = self.post_quant_conv(ttnn.typecast(z, ttnn.bfloat16))
        out = self.decoder(x)  # [B, 3, 1, H, W]
        H, W = out.shape[3], out.shape[4]
        out = ttnn.reshape(ttnn.typecast(out, ttnn.float32), (B, 3, H, W))
        out = ttnn.clamp(out, -1.0, 1.0)
        return ttnn.clamp(ttnn.add(ttnn.multiply(out, 0.5), 0.5), 0.0, 1.0)
