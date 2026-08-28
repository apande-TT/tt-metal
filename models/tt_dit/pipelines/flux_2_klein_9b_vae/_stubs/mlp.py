# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `mlp` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

`mlp` is one of the scaffold's generic transformer roles; none of its candidate
paths exists on `diffusers.AutoencoderKLFlux2` and discovery left it unbound.
There is no gated/SwiGLU MLP anywhere in this checkpoint — it is a convnet, and
the position-wise feed-forward unit of a convnet is its residual block. The role
is therefore bound — in `bringup_status.json` and in the test's candidate list —
to

    decoder.mid_block.resnets.0   (ResnetBlock2D)
        GroupNorm(32, 512) -> SiLU -> Conv2d(512, 512, 3x3, pad 1)
          -> GroupNorm(32, 512) -> SiLU -> Conv2d(512, 512, 3x3, pad 1)
          -> + input                      (temb_channels is None here, and
                                           output_scale_factor is 1.0)

which is `models/tt_dit/models/vae/vae.py::VaeResnetBlock`; it loads the
diffusers-named state dict verbatim, and with `in_channels == out_channels` it
builds no `conv_shortcut`, matching the torch module's `use_in_shortcut=False`.

TP=8: 512 channels give a 64-wide shard, so the block is channel-parallel
throughout — both convs are 512->512, i.e. row-parallel (`in_mesh_axis` +
reduce_scatter), and GroupNorm needs no collective because 32 groups over 8
devices is 4 whole groups per device, so every device already owns complete
groups and its statistics are exact. The per-component contract is replicated
NCHW in / out, so `NchwAdapter` fractures the channel axis on entry and
all-gathers it on exit.
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


class TtMlp(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "mlp stub needs the torch module to source its weights"
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
    return TtMlp.build(device, torch_module)


def mlp(device, torch_module=None):
    return TtMlp.build(device, torch_module)
