# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared NATIVE ttnn building blocks for the FLUX.2-klein-9B VAE bring-up.

Every component of this checkpoint (`diffusers.AutoencoderKLFlux2`) is a piece of
its `Encoder` or `Decoder`, so the per-component stubs all compose the same small
set of blocks. Most of them already exist in `models/tt_dit/models/vae/vae.py`
(`VaeConv2d / VaeResnetBlock / VaeAttention / VaeMidBlock / VaeUpBlock /
VaeUpsampler / _norm`) and load *diffusers-named* state dicts verbatim. What this
module adds is what is missing there:

  * `VaeDownsampler`  — diffusers `Downsample2D(use_conv=True, padding=0)`: an
    explicit right/bottom `(0,1,0,1)` pad followed by a stride-2, padding-0 conv.
    A symmetric `padding=1` gives the same output SIZE but a one-pixel-shifted
    window, so the pad has to be done on the activation.
  * `VaeDownBlock`    — diffusers `DownEncoderBlock2D` (no encoder-side down
    block exists in tt_dit).
  * `Flux2VaeEncoderBody` / `Flux2VaeDecoderBody` — the diffusers `Encoder` /
    `Decoder` submodules themselves (tt_dit's `Flux2VaeDecoder` is the *whole*
    VAE decode path: it also owns `post_quant_conv`, the BatchNorm
    inv-normalisation and unpatchify, none of which belong to `model.decoder`).
  * `NchwAdapter`     — the per-component PCC contract: replicated NCHW in,
    replicated NCHW out, around a channel-fractured NHWC block.

Tensor-parallel scheme (channel-parallel, mesh 1xTP)
---------------------------------------------------
`VaeConv2d` picks column-parallel (`out_mesh_axis`) when a conv widens and
row-parallel (`in_mesh_axis` + reduce_scatter) when it narrows, so activations
stay channel-fractured from one block to the next and no collective is needed
per block. GroupNorm needs no collective either: 32 groups over 8 devices leaves
4 whole groups per device, so each device's statistics are already exact.

The one hard limit is TILE_WIDTH. A channel shard narrower than one 32-wide tile
cannot be sliced in TILE layout —

    ttnn.mesh_partition -> slice: "Can only slice tilized tensor with width begin
    index aligned to tiles" (slice_device_operation.cpp:168)

— so at TP=8 a 128-channel stage (128/8 = 16) is NOT shardable, while 256 (32)
and 512 (64) are. `is_shardable()` encodes exactly that, and the encoder's
128-channel head and the decoder's 128-channel tail therefore run REPLICATED,
with the switch placed at a 256-channel boundary where the activation is
tile-aligned and can be fractured/gathered exactly.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import ttnn
from models.tt_dit.layers.conv2d import Conv2d
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.models.vae.vae import (
    VaeContext,
    VaeConv2d,
    VaeMidBlock,
    VaeNormDescGroup,
    VaeResnetBlock,
    VaeUpBlock,
    _norm,
)
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.substate import rename_substate

# Every norm in this checkpoint is GroupNorm(32, eps=1e-6).
VAE_NORM = VaeNormDescGroup(num_groups=32, eps=1e-6)

# diffusers config of /tmp/tt_hw_planner_components/flux_2_klein_9b_vae
BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
LAYERS_PER_BLOCK = 2
LATENT_CHANNELS = 32
ENCODER_OUT_CHANNELS = 2 * LATENT_CHANNELS  # mean | logvar
IMAGE_CHANNELS = 3

_TILE_WIDTH = 32


# ---------------------------------------------------------------------------
# context / mesh helpers
# ---------------------------------------------------------------------------


def resolve_tp_axis(device) -> int | None:
    """The mesh axis that actually holds more than one device.

    The shard harness opens `MeshShape(1, TP)`, so the TP axis is 1; a 1x1 mesh
    has nothing to shard and every block runs replicated.
    """
    shape = tuple(device.shape)
    for axis in range(len(shape) - 1, -1, -1):
        if int(shape[axis]) > 1:
            return axis
    return None


def make_ctx(device) -> VaeContext:
    """A channel-tensor-parallel `VaeContext` for `device` (no spatial sharding)."""
    tp_axis = resolve_tp_axis(device)
    ccl_manager = CCLManager(device, num_links=1, topology=ttnn.Topology.Linear) if tp_axis is not None else None
    return VaeContext(device=device, tp_axis=tp_axis, ccl_manager=ccl_manager)


def tp_size(ctx: VaeContext) -> int:
    return int(ctx.device.shape[ctx.tp_axis]) if ctx.tp_axis is not None else 1


def replicated_ctx(ctx: VaeContext) -> VaeContext:
    """`ctx` with tensor parallelism switched off (weights and activations whole)."""
    return replace(ctx, tp_axis=None)


def is_shardable(ctx: VaeContext, *channel_counts: int) -> bool:
    """Whether every one of `channel_counts` splits into tile-aligned shards.

    See the module docstring: a sub-tile channel shard cannot be sliced in TILE
    layout, so such a stage must run replicated.
    """
    n = tp_size(ctx)
    if n == 1:
        return False
    return all(int(c) % (_TILE_WIDTH * n) == 0 for c in channel_counts)


# ---------------------------------------------------------------------------
# layout / collective helpers
# ---------------------------------------------------------------------------


def to_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    return x if x.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def to_row_major(x: ttnn.Tensor) -> ttnn.Tensor:
    return x if x.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)


def to_nhwc(x: ttnn.Tensor) -> ttnn.Tensor:
    """`[N, C, H, W]` -> `[N, H, W, C]`. Returns TILE when C is tile-aligned."""
    x = ttnn.permute(to_row_major(x), (0, 2, 3, 1))
    return to_tile(x) if int(x.shape[-1]) % _TILE_WIDTH == 0 else x


def to_nchw(x: ttnn.Tensor) -> ttnn.Tensor:
    """`[N, H, W, C]` -> `[N, C, H, W]`."""
    return ttnn.permute(to_row_major(x), (0, 3, 1, 2))


def fracture_channels(ctx: VaeContext, x: ttnn.Tensor) -> ttnn.Tensor:
    """Replicated `[..., C]` -> `[..., C/tp]`, the local slice this device owns.

    `mesh_partition` is a per-device slice (no fabric), the exact inverse of
    `all_gather`. Callers must only reach here with a tile-aligned shard width
    (see `is_shardable`).
    """
    if ctx.tp_axis is None:
        return x
    return ttnn.mesh_partition(x, len(x.shape) - 1, ctx.tp_axis)


def gather_channels(ctx: VaeContext, x: ttnn.Tensor) -> ttnn.Tensor:
    """`[..., C/tp]` -> the full `[..., C]`, identical on every device."""
    if ctx.tp_axis is None or ctx.ccl_manager is None:
        return x
    return ctx.ccl_manager.all_gather(to_tile(x), dim=len(x.shape) - 1, mesh_axis=ctx.tp_axis, use_hyperparams=True)


# ---------------------------------------------------------------------------
# blocks missing from tt_dit
# ---------------------------------------------------------------------------


class VaeDownsampler(Module):
    """diffusers `Downsample2D(use_conv=True, padding=0, name="op")`.

    diffusers pads the activation `F.pad(x, (0, 1, 0, 1))` on NCHW — one column
    on the right, one row at the bottom — then runs a stride-2 padding-0 conv.
    A symmetric `padding=1` would give the same output SIZE with a one-pixel
    shifted window, so the asymmetry has to be preserved. It is expressed as
    `ttnn.conv2d`'s 4-element `padding=(pad_top, pad_bottom, pad_left,
    pad_right)` (`sliding_window::get_pair_n4_padding`), which is exactly the
    same window and needs no separate pad op:

        H_out = (H - 3 + (0 + 1)) // 2 + 1 = H // 2   (likewise W)
    """

    def __init__(self, *, channels: int, out_channels: int | None = None, ctx: VaeContext) -> None:
        super().__init__()
        out_channels = out_channels or channels
        # in == out here, so this is the narrowing (row-parallel) case.
        shard = is_shardable(ctx, channels, out_channels)
        self.conv = Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=(0, 1, 0, 1),
            mesh_device=ctx.device,
            in_mesh_axis=ctx.tp_axis if shard else None,
            ccl_manager=ctx.ccl_manager,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return to_tile(self.conv.forward(x, use_persistent_buffer=False))


class VaeDownBlock(Module):
    """diffusers `DownEncoderBlock2D`: `num_layers` resnets, then an optional
    stride-2 downsampler."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        downsample: bool,
        norm: VaeNormDescGroup = VAE_NORM,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            VaeResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm=norm,
                ctx=ctx,
            )
            for i in range(num_layers)
        )
        self.downsampler = VaeDownsampler(channels=out_channels, ctx=ctx) if downsample else None

    def _prepare_torch_state(self, state) -> None:
        rename_substate(state, "downsamplers.0", "downsampler")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for resnet in self.resnets:
            x = resnet.forward(x)
        if self.downsampler is not None:
            x = self.downsampler.forward(x)
        return x


# ---------------------------------------------------------------------------
# the two halves of the VAE
# ---------------------------------------------------------------------------


class Flux2VaeDecoderBody(Module):
    """diffusers `Decoder` — `conv_in -> mid_block -> up_blocks -> conv_norm_out
    -> SiLU -> conv_out`.

    Channel-parallel while the per-device shard is at least one tile wide; the
    final 128-channel up block and the tail run replicated, with a single
    `all_gather` at the 256-channel boundary between the two regimes.
    """

    def __init__(
        self,
        *,
        in_channels: int = LATENT_CHANNELS,
        out_channels: int = IMAGE_CHANNELS,
        block_out_channels=BLOCK_OUT_CHANNELS,
        layers_per_block: int = LAYERS_PER_BLOCK,
        norm: VaeNormDescGroup = VAE_NORM,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        channel_counts = [block_out_channels[-1], *block_out_channels[::-1]]
        rep = replicated_ctx(ctx)

        # conv_in widens from the replicated latent, so only its output has to shard.
        self.conv_in = VaeConv2d(
            in_channels,
            channel_counts[0],
            kernel_size=3,
            padding=1,
            ctx=ctx if is_shardable(ctx, channel_counts[0]) else rep,
        )
        self.mid_block = VaeMidBlock(
            num_channels=channel_counts[0],
            norm=norm,
            ctx=ctx if is_shardable(ctx, channel_counts[0]) else rep,
        )

        pairs = list(itertools.pairwise(channel_counts))
        self._up_sharded = [is_shardable(ctx, ci, co) for ci, co in pairs]
        self.up_blocks = ModuleList(
            VaeUpBlock(
                in_channels=ci,
                out_channels=co,
                upsample=i != len(channel_counts) - 2,
                num_layers=layers_per_block + 1,
                norm=norm,
                ctx=ctx if self._up_sharded[i] else rep,
            )
            for i, (ci, co) in enumerate(pairs)
        )

        tail_sharded = is_shardable(ctx, channel_counts[-1])
        self.conv_norm_out = _norm(
            norm, num_channels=channel_counts[-1], ctx=ctx if tail_sharded else rep, activation_fn="silu"
        )
        self.conv_out = VaeConv2d(
            channel_counts[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=rep
        )
        self._tail_sharded = tail_sharded
        self._ctx = ctx

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`x`: `[N, H, W, latent_channels]` NHWC, replicated."""
        x = to_tile(self.conv_in.forward(x))
        x = self.mid_block.forward(x)

        sharded = self._up_sharded[0] if self._up_sharded else False
        for i, block in enumerate(self.up_blocks):
            if sharded and not self._up_sharded[i]:
                # leaving the sharded region: rebuild the full channel axis once
                x = gather_channels(self._ctx, x)
                sharded = False
            x = block.forward(x)

        x = self.conv_norm_out.forward(x)
        if self._tail_sharded:
            x = gather_channels(self._ctx, x)
        return self.conv_out.forward(to_tile(x))


class Flux2VaeEncoderBody(Module):
    """diffusers `Encoder` — `conv_in -> down_blocks -> mid_block ->
    conv_norm_out -> SiLU -> conv_out`.

    Mirror image of the decoder: the 128-channel head (which is also the
    highest-resolution stage) runs replicated because 128/8 = 16 is half a tile,
    and the activation is fractured once it reaches 256 channels.
    """

    def __init__(
        self,
        *,
        in_channels: int = IMAGE_CHANNELS,
        out_channels: int = ENCODER_OUT_CHANNELS,
        block_out_channels=BLOCK_OUT_CHANNELS,
        layers_per_block: int = LAYERS_PER_BLOCK,
        norm: VaeNormDescGroup = VAE_NORM,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        rep = replicated_ctx(ctx)

        self.conv_in = VaeConv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
            ctx=ctx if is_shardable(ctx, block_out_channels[0]) else rep,
        )

        block_channels = [(block_out_channels[max(i - 1, 0)], ch) for i, ch in enumerate(block_out_channels)]
        self._down_sharded = [is_shardable(ctx, ci, co) for ci, co in block_channels]
        self.down_blocks = ModuleList(
            VaeDownBlock(
                in_channels=ci,
                out_channels=co,
                num_layers=layers_per_block,
                downsample=i != len(block_out_channels) - 1,
                norm=norm,
                ctx=ctx if self._down_sharded[i] else rep,
            )
            for i, (ci, co) in enumerate(block_channels)
        )

        mid_sharded = is_shardable(ctx, block_out_channels[-1])
        self.mid_block = VaeMidBlock(num_channels=block_out_channels[-1], norm=norm, ctx=ctx if mid_sharded else rep)
        self.conv_norm_out = _norm(
            norm, num_channels=block_out_channels[-1], ctx=ctx if mid_sharded else rep, activation_fn="silu"
        )
        self.conv_out = VaeConv2d(
            block_out_channels[-1], out_channels, kernel_size=3, padding=1, tensor_parallel=False, ctx=rep
        )
        self._mid_sharded = mid_sharded
        self._ctx = ctx

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`x`: `[N, H, W, 3]` NHWC, replicated."""
        x = to_tile(self.conv_in.forward(x))

        sharded = False
        for i, block in enumerate(self.down_blocks):
            if self._down_sharded[i] and not sharded:
                # entering the sharded region: take this device's channel slice
                x = fracture_channels(self._ctx, x)
                sharded = True
            x = block.forward(x)

        x = self.mid_block.forward(x)
        x = self.conv_norm_out.forward(x)
        if self._mid_sharded:
            x = gather_channels(self._ctx, x)
        return self.conv_out.forward(to_tile(x))


# ---------------------------------------------------------------------------
# per-component PCC adapter
# ---------------------------------------------------------------------------


class NchwAdapter:
    """The per-component PCC contract around a tt_dit VAE block.

    The shard harness replicates the golden's NCHW input over the mesh and reads
    one device's copy back, so a stub must take replicated NCHW and return
    replicated, full-width NCHW. Blocks that live in the middle of the network
    speak channel-fractured NHWC, so `fracture_input` / `gather_output` add the
    `mesh_partition` / `all_gather` pair; blocks whose own input or output is
    already whole (the encoder's 3 channels, the decoder's 3 channels) leave them
    off. Composites should call `forward()` and keep the activation fractured.
    """

    def __init__(self, inner, ctx: VaeContext, *, fracture_input: bool = False, gather_output: bool = False) -> None:
        self.inner = inner
        self.ctx = ctx
        self._fracture_input = fracture_input and ctx.tp_axis is not None
        self._gather_output = gather_output and ctx.tp_axis is not None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.inner.forward(x)

    def __call__(self, x: ttnn.Tensor, *args, **kwargs) -> ttnn.Tensor:
        x = to_nhwc(x)
        if self._fracture_input:
            x = fracture_channels(self.ctx, x)
        y = self.inner.forward(x)
        if self._gather_output:
            y = gather_channels(self.ctx, y)
        return to_nchw(y)
