# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `upsample2_d` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `decoder.up_blocks.0.upsamplers.0` of
`diffusers.AutoencoderKLFlux2` — `Upsample2D(interpolate=True, use_conv=True,
name="conv")`:

    F.interpolate(x, scale_factor=2.0, mode="nearest")  then
    Conv2d(512, 512, kernel_size=3, padding=1)

which is `models/tt_dit/models/vae/vae.py::VaeUpsampler` (`tensor.upsample` ->
`VaeConv2d`), and its single child is named `conv` in both, so the diffusers
state dict loads verbatim.

TP=8: the conv is 512->512, so `VaeConv2d` makes it row-parallel — input channels
sharded 64-wide per device, partial sums combined by `reduce_scatter`, and the
bias added on one device only (`Conv2d._prepare_torch_state` pads it with zeros
for the others) so it lands exactly once. Nearest-neighbour upsampling maps each
pixel to a 2x2 block independently of the channel axis, so it commutes with the
shard and needs no collective. `NchwAdapter` supplies the per-component contract:
replicated NCHW in, `ttnn.mesh_partition` on entry, `all_gather` on exit,
replicated NCHW out.
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block body dispatches through ttnn)
from models.tt_dit.models.vae.vae import VaeUpsampler
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import NchwAdapter, is_shardable, make_ctx, replicated_ctx


class TtUpsample2D(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "upsample2_d stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        conv = torch_module.conv
        in_channels = int(conv.in_channels)
        out_channels = int(conv.out_channels)
        shard = is_shardable(ctx, in_channels, out_channels)

        inner = VaeUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            ctx=ctx if shard else replicated_ctx(ctx),
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        return cls(inner, ctx, fracture_input=shard, gather_output=shard)


def build(device, torch_module=None):
    return TtUpsample2D.build(device, torch_module)


def upsample2_d(device, torch_module=None):
    return TtUpsample2D.build(device, torch_module)
