# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `layer` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

`layer` is one of the scaffold's generic transformer roles; none of its candidate
paths exist on `diffusers.AutoencoderKLFlux2` (whose children are encoder,
decoder, quant_conv, post_quant_conv, bn) and discovery left it unbound. This
model is a convnet, and its repeating unit is a decoder up block, so the role is
bound — in `bringup_status.json` and in the test's candidate list — to

    decoder.up_blocks.0   (UpDecoderBlock2D)
        3 x ResnetBlock2D(512 -> 512)  ->  Upsample2D(512, nearest x2 + 3x3 conv)

which is `models/tt_dit/models/vae/vae.py::VaeUpBlock` (`VaeResnetBlock` +
`VaeUpsampler`); it loads the diffusers-named state dict verbatim once
`upsamplers.0` is renamed to `upsampler`, which `VaeUpBlock` already does. The
scaffold's seed (`llama_layernorm.py`) is not this module in any part.

TP=8: 512 channels give a 64-wide shard, so the whole block is channel-parallel
— `VaeConv2d` picks column-parallel where a conv widens and row-parallel +
reduce_scatter where it narrows (here every conv is 512->512, i.e. row-parallel),
and GroupNorm needs no collective because 32 groups over 8 devices is 4 whole
groups per device. Nearest-neighbour upsampling is per-pixel, so it commutes with
the channel shard. The per-component contract is replicated NCHW in / out, so
`NchwAdapter` fractures the channel axis on entry (`ttnn.mesh_partition`, a local
slice) and all-gathers it on exit; inside a composite the block is called through
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


class TtLayer(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "layer stub needs the torch module to source its weights"
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
    return TtLayer.build(device, torch_module)


def layer(device, torch_module=None):
    return TtLayer.build(device, torch_module)
