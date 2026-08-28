# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `u_net_mid_block2_d` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `encoder.mid_block` of
`diffusers.AutoencoderKLFlux2` — a `UNetMidBlock2D` at the encoder's bottleneck:

    ResnetBlock2D(512 -> 512) -> Attention(512, 1 head) -> ResnetBlock2D(512 -> 512)

`UNetMidBlock2D.forward` runs `resnets[0]`, then zips the attentions with
`resnets[1:]`, which for one attention is exactly the interleave
`models/tt_dit/models/vae/vae.py::VaeMidBlock` produces for `num_layers=1`, so
that class is this module — including the diffusers-named state dict
(`resnets.{0,1}`, `attentions.0`, with `to_out.0` and `group_norm` renamed
inside `VaeAttention._prepare_torch_state`).

TP=8: 512 channels give a 64-wide shard, so the whole block is channel-parallel.
Both resnets' convs are 512->512 and therefore row-parallel (`in_mesh_axis` +
reduce_scatter); the attention is column-parallel on a fused `to_qkv` whose
per-device `Q|K|V` interleaving keeps each chip's chunks contiguous, then
all-gathers q/k/v because with `heads == 1` the head_dim is all 512 channels and
single-head SDPA needs the whole of it for QK^T. GroupNorm needs no collective:
32 groups over 8 devices is 4 whole groups per device, so each device's
statistics are already exact. `NchwAdapter` supplies the per-component contract —
replicated NCHW in, `ttnn.mesh_partition` on entry, `all_gather` on exit,
replicated NCHW out.
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block body dispatches through ttnn)
from models.tt_dit.models.vae.vae import VaeMidBlock
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import (
    VAE_NORM,
    NchwAdapter,
    is_shardable,
    make_ctx,
    replicated_ctx,
)


class TtUNetMidBlock2D(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "u_net_mid_block2_d stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        num_channels = int(torch_module.resnets[0].norm1.num_channels)
        shard = is_shardable(ctx, num_channels)

        inner = VaeMidBlock(
            num_channels=num_channels,
            num_layers=len(torch_module.attentions),
            norm=VAE_NORM,
            ctx=ctx if shard else replicated_ctx(ctx),
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        return cls(inner, ctx, fracture_input=shard, gather_output=shard)


def build(device, torch_module=None):
    return TtUNetMidBlock2D.build(device, torch_module)


def u_net_mid_block2_d(device, torch_module=None):
    return TtUNetMidBlock2D.build(device, torch_module)
