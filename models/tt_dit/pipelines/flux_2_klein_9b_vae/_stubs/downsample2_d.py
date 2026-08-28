# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE ttnn port of `downsample2_d` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `encoder.down_blocks.0.downsamplers.0` of
`diffusers.AutoencoderKLFlux2` — `Downsample2D(use_conv=True, padding=0,
name="op")`, i.e. `Conv2d(128, 128, kernel_size=3, stride=2)` fed a
right/bottom-padded activation:

    F.pad(x, (0, 1, 0, 1))  then  stride-2, padding-0 conv

`vae_blocks.VaeDownsampler` reproduces that window exactly by giving
`ttnn.conv2d` the 4-element `padding=(pad_top, pad_bottom, pad_left, pad_right)
= (0, 1, 0, 1)` instead of padding the activation, so the output is
`H//2 x W//2` with the same one-sided window a symmetric `padding=1` would
shift. (Doing it as a separate `ttnn.pad` on a ROW_MAJOR activation is the other
route, but `writer_pad_dims_rm_interleaved_v2` does not build in this
environment — the conv's own padding avoids the op entirely.)

Parallelism: 128 channels over TP=8 is 16, half a tile, and a sub-tile channel
shard cannot be sliced in TILE layout (`ttnn.mesh_partition` -> slice: "Can only
slice tilized tensor with width begin index aligned to tiles"), so
`vae_blocks.is_shardable` runs this stage REPLICATED — the same regime
`Flux2VaeEncoderBody` uses for its 128-channel head.
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block body dispatches through ttnn)
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import (
    NchwAdapter,
    VaeDownsampler,
    is_shardable,
    make_ctx,
    replicated_ctx,
)


class TtDownsample2D(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "downsample2_d stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        conv = torch_module.conv
        channels = int(conv.in_channels)
        out_channels = int(conv.out_channels)
        shard = is_shardable(ctx, channels, out_channels)

        inner = VaeDownsampler(
            channels=channels,
            out_channels=out_channels,
            ctx=ctx if shard else replicated_ctx(ctx),
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        return cls(inner, ctx, fracture_input=shard, gather_output=shard)


def build(device, torch_module=None):
    return TtDownsample2D.build(device, torch_module)


def downsample2_d(device, torch_module=None):
    return TtDownsample2D.build(device, torch_module)
