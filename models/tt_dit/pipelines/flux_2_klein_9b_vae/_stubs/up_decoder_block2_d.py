# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `up_decoder_block2_d` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `decoder.up_blocks.0` of
`diffusers.AutoencoderKLFlux2` — an `UpDecoderBlock2D`, the decoder stack's
repeating unit (and the same module the generic `layer` role is bound to):

    3 x ResnetBlock2D(512 -> 512)  ->  Upsample2D(512, nearest x2 + 3x3 conv)

which is `models/tt_dit/models/vae/vae.py::VaeUpBlock` (`VaeResnetBlock` +
`VaeUpsampler`). It loads the diffusers-named state dict verbatim once
`upsamplers.0` is renamed to `upsampler`, which `VaeUpBlock` already does.

TP=8: 512 channels give a 64-wide shard, so the whole block is channel-parallel —
every conv here is 512->512, i.e. row-parallel (`in_mesh_axis` +
reduce_scatter), and GroupNorm needs no collective because 32 groups over 8
devices is 4 whole groups per device. Nearest-neighbour upsampling is per-pixel,
so it commutes with the channel shard. `NchwAdapter` supplies the per-component
contract: replicated NCHW in, `ttnn.mesh_partition` on entry, `all_gather` on
exit, replicated NCHW out; inside a composite the block is driven through
`forward()` and the activation stays fractured.
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block body dispatches through ttnn)
from models.tt_dit.models.vae.vae import VaeUpBlock
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import (
    VAE_NORM,
    NchwAdapter,
    is_shardable,
    make_ctx,
    replicated_ctx,
)


class TtUpDecoderBlock2D(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "up_decoder_block2_d stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        resnets = torch_module.resnets
        in_channels = int(resnets[0].norm1.num_channels)
        out_channels = int(resnets[-1].conv2.out_channels)
        shard = is_shardable(ctx, in_channels, out_channels)

        inner = VaeUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=len(resnets),
            upsample=getattr(torch_module, "upsamplers", None) is not None,
            norm=VAE_NORM,
            ctx=ctx if shard else replicated_ctx(ctx),
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        return cls(inner, ctx, fracture_input=shard, gather_output=shard)


def build(device, torch_module=None):
    return TtUpDecoderBlock2D.build(device, torch_module)


def up_decoder_block2_d(device, torch_module=None):
    return TtUpDecoderBlock2D.build(device, torch_module)
