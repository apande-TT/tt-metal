# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `encoder` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `encoder` of `diffusers.AutoencoderKLFlux2` — the whole
`Encoder` submodule:

    conv_in(3 -> 128)
      -> down_blocks[0..3] (128->128, 128->256, 256->512, 512->512; 2 resnets
                            each, stride-2 Downsample2D on the first three)
      -> mid_block(512, 2 resnets + 1 attention)
      -> GroupNorm(32, 512) -> SiLU -> conv_out(512 -> 64)

`vae_blocks.Flux2VaeEncoderBody` is that module. tt_dit has no encoder-side down
block, so `vae_blocks` supplies `VaeDownBlock` and `VaeDownsampler` (the latter
folds diffusers' asymmetric `(0, 1, 0, 1)` activation pad into `ttnn.conv2d`'s
4-element padding, which is the same window a symmetric `padding=1` would
shift); everything else composes the in-tree tt_dit VAE blocks, which load
diffusers-named state dicts verbatim.

TP=8 scheme (see `vae_blocks` for the details): channel-parallel — column-
parallel where a conv widens, row-parallel + reduce_scatter where it narrows, so
activations stay fractured between blocks, and GroupNorm needs no collective
because 32 groups over 8 devices is 4 whole groups per device. The exception is
the 128-channel head: 128/8 = 16 is half a tile and a sub-tile channel shard
cannot be sliced in TILE layout, so `conv_in` and `down_blocks[0..1]` run
replicated and the activation is fractured exactly once, at the 256-channel
boundary where the shard is tile-aligned. The tail mirrors it: one `all_gather`
of the 512-channel activation before the replicated `conv_out`, whose 64 outputs
are the mean|logvar pair the wrapper consumes — so this stub adds no collective
of its own (3 channels in, 64 out, both whole).
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block bodies dispatch through ttnn)
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import Flux2VaeEncoderBody, NchwAdapter, make_ctx


class TtEncoder(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "encoder stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        block_out_channels = tuple(int(b.resnets[-1].conv2.out_channels) for b in torch_module.down_blocks)
        inner = Flux2VaeEncoderBody(
            in_channels=int(torch_module.conv_in.in_channels),
            out_channels=int(torch_module.conv_out.out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=len(torch_module.down_blocks[0].resnets),
            ctx=ctx,
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        # 3 image channels in and 64 (mean|logvar) out: both ends stay whole.
        return cls(inner, ctx, fracture_input=False, gather_output=False)


def build(device, torch_module=None):
    return TtEncoder.build(device, torch_module)


def encoder(device, torch_module=None):
    return TtEncoder.build(device, torch_module)
