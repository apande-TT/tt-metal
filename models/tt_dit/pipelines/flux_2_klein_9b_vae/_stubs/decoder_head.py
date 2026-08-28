# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `decoder_head` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component name is the scaffold's generic "decoder head" role, but the capture
manifest records `submodule_path = decoder`, and that path is tried first — so this
component IS `model.decoder`, the same `diffusers.Decoder` as the `decoder`
component, and shares its implementation:

    conv_in(32 -> 512) -> mid_block(512, 2 resnets + 1 attention)
      -> up_blocks[0..3] (512->512, 512->512, 512->256, 256->128; 3 resnets each,
                          nearest-2x upsampler on the first three)
      -> GroupNorm(32, 128) -> SiLU -> conv_out(128 -> 3)

`vae_blocks.Flux2VaeDecoderBody` is that module, built out of the in-tree tt_dit
VAE blocks (which load diffusers-named state dicts verbatim). Note it is NOT
`models/tt_dit/models/vae/vae_flux2.py::Flux2VaeDecoder`: that class is the whole
decode PATH and additionally owns `post_quant_conv`, the BatchNorm
inv-normalisation and the unpatchify, none of which are children of
`model.decoder` and none of whose weights are in this component's state dict.

TP=8 scheme and its one hard limit are documented in `vae_blocks`: channel-
parallel throughout (column-parallel where a conv widens, row-parallel +
reduce_scatter where it narrows, GroupNorm collective-free because 32 groups over
8 devices is 4 whole groups per device), except that a 128-channel stage cannot
shard at TP=8 — 128/8 = 16 is half a tile and TILE slicing rejects it — so the
last up block and the tail run replicated after one `all_gather` at the
256-channel boundary. Both ends of the module are already whole (32 latent
channels in, 3 image channels out), so the stub itself adds no collective.
"""
from __future__ import annotations

import torch

import ttnn  # noqa: F401  (the block bodies dispatch through ttnn)
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import Flux2VaeDecoderBody, NchwAdapter, make_ctx


class TtDecoderHead(NchwAdapter):
    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "decoder_head stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        ctx = make_ctx(device)
        block_out_channels = tuple(reversed([int(b.resnets[-1].conv2.out_channels) for b in torch_module.up_blocks]))
        inner = Flux2VaeDecoderBody(
            in_channels=int(torch_module.conv_in.in_channels),
            out_channels=int(torch_module.conv_out.out_channels),
            block_out_channels=block_out_channels,
            layers_per_block=len(torch_module.up_blocks[0].resnets) - 1,
            ctx=ctx,
        )
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        # 32 latent channels in and 3 image channels out: both ends stay whole,
        # so no partition/gather wraps the block.
        return cls(inner, ctx, fracture_input=False, gather_output=False)


def build(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)


def decoder_head(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)
