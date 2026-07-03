# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step AutoencoderOobleck TT decoder (Phase 4).

Port checklist (Phase 4):
  [x] OobleckSnake1d          — SnakeBeta wrapper (log-scaled α/β)
  [x] OobleckConv1d           — weight-norm fold + _AlignedOutConv1d
  [x] OobleckConvTranspose1d  — zero-stuff + torch padding=ceil(stride/2)
  [x] OobleckResidualUnit     — torch-style residual center-crop before add
  [x] OobleckDecoderBlock     — upsample + 3× ResUnit wired
  [x] OobleckDecoder.forward  — end-to-end runnable on device
  [ ] PCC ≥ 0.99 vs torch     — pending device gate (test_vae_oobleck_decoder.py)

Layer padding/crop fixes live in ``oobleck_layers.py`` (imported building blocks).
Reference torch module: ``diffusers.models.autoencoders.autoencoder_oobleck.OobleckDecoder``
(~169M params: 32 Conv1d, 5 ConvTranspose1d, 36 Snake1d, 15 ResUnit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

import ttnn

from ...layers.module import Module, ModuleList
from .oobleck_layers import OobleckConv1d, OobleckDecoderBlock, OobleckSnake1d

# Set True once component PCC test passes (see test_vae_oobleck_decoder.py).
# Phase 4 code fixes landed; device gate must confirm PCC ≥ 0.99 before enabling.
OOBLECK_DECODER_PORT_COMPLETE = False

# ACE-Step1.5 checkpoint defaults (downsampling [2,4,4,6,10] → upsampling reversed).
ACESTEP_OOBLECK_DECODER_CFG = dict(
    decoder_channels=128,
    decoder_input_channels=64,
    audio_channels=2,
    upsampling_ratios=(10, 6, 4, 4, 2),
    channel_multiples=(1, 2, 4, 8, 16),
)


@dataclass(frozen=True)
class OobleckDecoderConfig:
    decoder_channels: int
    decoder_input_channels: int
    audio_channels: int
    upsampling_ratios: tuple[int, ...]
    channel_multiples: tuple[int, ...]

    @classmethod
    def from_torch_decoder(cls, torch_decoder) -> OobleckDecoderConfig:
        """Infer block geometry from a loaded diffusers ``OobleckDecoder``."""
        blocks = list(torch_decoder.block)
        base_channels = int(torch_decoder.conv2.in_channels)
        upsampling_ratios = tuple(int(blk.conv_t1.stride[0]) for blk in blocks)
        block_mults = {int(blk.conv_t1.out_channels // base_channels) for blk in blocks}
        block_mults.add(int(torch_decoder.conv1.out_channels // base_channels))
        channel_multiples = tuple(sorted(block_mults))
        return cls(
            decoder_channels=base_channels,
            decoder_input_channels=int(torch_decoder.conv1.in_channels),
            audio_channels=int(torch_decoder.conv2.out_channels),
            upsampling_ratios=upsampling_ratios,
            channel_multiples=channel_multiples,
        )


class OobleckDecoder(Module):
    """TT port of diffusers ``OobleckDecoder``.

    ``forward`` accepts torch latents ``(B, C_latent, T)`` and returns stereo/mono
    waveform ``(B, audio_channels, T_out)``. Internally converts to ``(B, T, C)``
    ROW_MAJOR for ``Conv1dViaConv3d`` (same boundary pattern as ``Vocoder``).
    """

    def __init__(
        self,
        *,
        decoder_channels: int = ACESTEP_OOBLECK_DECODER_CFG["decoder_channels"],
        decoder_input_channels: int = ACESTEP_OOBLECK_DECODER_CFG["decoder_input_channels"],
        audio_channels: int = ACESTEP_OOBLECK_DECODER_CFG["audio_channels"],
        upsampling_ratios: Sequence[int] = ACESTEP_OOBLECK_DECODER_CFG["upsampling_ratios"],
        channel_multiples: Sequence[int] = ACESTEP_OOBLECK_DECODER_CFG["channel_multiples"],
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()

        self.decoder_channels = decoder_channels
        self.decoder_input_channels = decoder_input_channels
        self.audio_channels = audio_channels
        self.upsampling_ratios = tuple(upsampling_ratios)
        self.channel_multiples = tuple(channel_multiples)
        self.mesh_device = mesh_device
        self.dtype = dtype

        mults = (1,) + tuple(channel_multiples)
        strides = self.upsampling_ratios

        self.conv1 = OobleckConv1d(
            decoder_input_channels,
            decoder_channels * mults[-1],
            kernel_size=7,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.block = ModuleList()
        for stride_index, stride in enumerate(strides):
            in_dim = decoder_channels * mults[len(strides) - stride_index]
            out_dim = decoder_channels * mults[len(strides) - stride_index - 1]
            self.block.append(
                OobleckDecoderBlock(
                    in_dim,
                    out_dim,
                    stride=stride,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
            )

        self.snake1 = OobleckSnake1d(decoder_channels, mesh_device=mesh_device, dtype=dtype)
        self.conv2 = OobleckConv1d(
            decoder_channels,
            audio_channels,
            kernel_size=7,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    @classmethod
    def from_torch(
        cls,
        torch_decoder,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> OobleckDecoder:
        """Build from a diffusers ``OobleckDecoder`` and load its ``state_dict``."""
        cfg = OobleckDecoderConfig.from_torch_decoder(torch_decoder)
        model = cls(
            decoder_channels=cfg.decoder_channels,
            decoder_input_channels=cfg.decoder_input_channels,
            audio_channels=cfg.audio_channels,
            upsampling_ratios=cfg.upsampling_ratios,
            channel_multiples=cfg.channel_multiples,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        model.load_torch_state_dict(torch_decoder.state_dict())
        return model

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pass

    def forward(self, latents_BCT: torch.Tensor) -> torch.Tensor:
        """Decode latents ``(B, C_latent, T)`` → waveform ``(B, audio_channels, samples)``."""
        if latents_BCT.ndim != 3:
            raise ValueError(f"expected latents (B, C, T), got {tuple(latents_BCT.shape)}")

        b, channels, _t = latents_BCT.shape
        if channels != self.decoder_input_channels:
            raise ValueError(f"expected {self.decoder_input_channels} latent channels, got {channels}")

        x_BTC = latents_BCT.transpose(1, 2).float().contiguous()
        x_dev = ttnn.from_torch(x_BTC, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)

        x_dev = self.conv1(x_dev)
        for blk in self.block:
            x_dev = blk(x_dev)
        x_dev = self.snake1(x_dev)
        x_dev = self.conv2(x_dev)

        x_host = ttnn.to_torch(ttnn.get_device_tensors(x_dev)[0])
        x_host = x_host[..., : self.audio_channels]
        return x_host.transpose(1, 2).contiguous()
