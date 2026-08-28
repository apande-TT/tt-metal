# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE ttnn port of `down_encoder_block2_d` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `encoder.down_blocks.0` of
`diffusers.AutoencoderKLFlux2` — a `DownEncoderBlock2D`:

    2 x ResnetBlock2D(128 -> 128)  ->  Downsample2D(128, stride 2, padding 0)

There is no encoder-side down block in tt_dit, so `vae_blocks.VaeDownBlock`
(built from `VaeResnetBlock` + `vae_blocks.VaeDownsampler`) supplies it. The
downsampler matters: diffusers pads the activation `(0, 1, 0, 1)` — one column
right, one row bottom — and then runs a stride-2 padding-0 conv. A symmetric
`padding=1` would give the same output SIZE with a one-pixel-shifted window, so
the pad is done explicitly on the activation in ROW_MAJOR.

Parallelism: this block is 128 channels wide, and 128/8 = 16 is HALF a tile.
A sub-tile channel shard cannot be sliced in TILE layout —

    ttnn.mesh_partition -> slice: "Can only slice tilized tensor with width begin
    index aligned to tiles" (slice_device_operation.cpp:168)

— so at TP=8 there is no channel-parallel split of this block, and
`vae_blocks.is_shardable` puts it in the REPLICATED regime (which is also how
`Flux2VaeEncoderBody` runs its 128-channel head: it fractures the activation
only once the network reaches 256 channels). Weights stay whole, the output is
identical on every device, and the gathered-PCC contract holds by construction.
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block body dispatches through ttnn)
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import (
    NchwAdapter,
    VaeDownBlock,
    is_shardable,
    make_ctx,
    replicated_ctx,
)


class TtDownEncoderBlock2D(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "down_encoder_block2_d stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        resnets = torch_module.resnets
        in_channels = int(resnets[0].norm1.num_channels)
        out_channels = int(resnets[-1].conv2.out_channels)
        shard = is_shardable(ctx, in_channels, out_channels)

        inner = VaeDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=len(resnets),
            downsample=getattr(torch_module, "downsamplers", None) is not None,
            ctx=ctx if shard else replicated_ctx(ctx),
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        return cls(inner, ctx, fracture_input=shard, gather_output=shard)


def build(device, torch_module=None):
    return TtDownEncoderBlock2D.build(device, torch_module)


def down_encoder_block2_d(device, torch_module=None):
    return TtDownEncoderBlock2D.build(device, torch_module)
