# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE ttnn port of `resnet_block2_d` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `encoder.down_blocks.0.resnets.0` of
`diffusers.AutoencoderKLFlux2` — a `ResnetBlock2D`:

    GroupNorm(32, 128) -> SiLU -> Conv2d(128, 128, 3x3, pad 1)
      -> GroupNorm(32, 128) -> SiLU -> Conv2d(128, 128, 3x3, pad 1)
      -> + input                  (temb_channels is None, output_scale_factor 1.0)

which is `models/tt_dit/models/vae/vae.py::VaeResnetBlock`; it loads the
diffusers-named state dict verbatim, and with `in_channels == out_channels` it
builds no `conv_shortcut`, matching the torch module's `use_in_shortcut=False`.

Parallelism: this is the encoder's first, 128-channel stage. 128/8 = 16 is HALF a
tile, and a sub-tile channel shard cannot be sliced in TILE layout —

    ttnn.mesh_partition -> slice: "Can only slice tilized tensor with width begin
    index aligned to tiles" (slice_device_operation.cpp:168)

— so at TP=8 there is no channel-parallel split of this block and
`vae_blocks.is_shardable` puts it in the REPLICATED regime, exactly as
`Flux2VaeEncoderBody` runs its 128-channel head (it fractures the activation only
once the network reaches 256 channels, where the shard is a whole tile). The
output is identical on every device, so the gathered-PCC contract holds by
construction.
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block body dispatches through ttnn)
from models.tt_dit.models.vae.vae import VaeResnetBlock
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import (
    VAE_NORM,
    NchwAdapter,
    is_shardable,
    make_ctx,
    replicated_ctx,
)


class TtResnetBlock2D(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "resnet_block2_d stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        in_channels = int(torch_module.norm1.num_channels)
        out_channels = int(torch_module.conv2.out_channels)
        shard = is_shardable(ctx, in_channels, out_channels)

        inner = VaeResnetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm=VAE_NORM,
            ctx=ctx if shard else replicated_ctx(ctx),
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        return cls(inner, ctx, fracture_input=shard, gather_output=shard)


def build(device, torch_module=None):
    return TtResnetBlock2D.build(device, torch_module)


def resnet_block2_d(device, torch_module=None):
    return TtResnetBlock2D.build(device, torch_module)
