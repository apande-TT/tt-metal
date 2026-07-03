# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step AutoencoderOobleck decoder building blocks (Phase B scaffold).

Reuses ``audio_ops.SnakeBeta``, ``Conv1dViaConv3d``, and ``ConvTranspose1dViaConv3d``
for the 1-D waveform decode chain on ``(B, T, C)`` ROW_MAJOR tensors.
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.audio_ops import SnakeBeta, _AlignedOutConv1d, _pad_channels_to_aligned, _zero_pad_t, _zero_stuff_t
from ...layers.module import Module


def fold_weight_norm_state(state: dict[str, torch.Tensor], *, conv_prefix: str = "conv.") -> None:
    """Fold diffusers ``weight_g`` / ``weight_v`` into ``conv.weight`` for TT load.

    Torch Oobleck uses ``weight_norm`` on Conv1d / ConvTranspose1d; the TT path
    expects a plain conv weight tensor on the inner ``_AlignedOutConv1d``.
    """
    weight_g = state.pop("weight_g", None)
    weight_v = state.pop("weight_v", None)
    if weight_g is None or weight_v is None:
        return

    norm = torch.linalg.vector_norm(weight_v, dim=tuple(range(1, weight_v.dim())), keepdim=True).clamp_min(1e-12)
    state[f"{conv_prefix}weight"] = (weight_g * weight_v / norm).contiguous()

    if "bias" in state:
        state[f"{conv_prefix}bias"] = state.pop("bias")


def _trim_t_to_length(x_BTC: ttnn.Tensor, target_len: int) -> ttnn.Tensor:
    """Center-trim ``x_BTC`` along T to ``target_len`` (handles odd excess)."""
    t = int(x_BTC.shape[1])
    if t == target_len:
        return x_BTC
    if t < target_len:
        return x_BTC

    excess = t - target_len
    pad_left = excess // 2
    pad_right = excess - pad_left
    b, _, c = x_BTC.shape
    return ttnn.slice(x_BTC, [0, pad_left, 0], [b, t - pad_right, c])


def _crop_residual_like_torch(reference: ttnn.Tensor, target_len: int) -> ttnn.Tensor:
    """Match diffusers ``OobleckResidualUnit`` residual trim before the add."""
    ref_t = int(reference.shape[1])
    if ref_t <= target_len:
        return reference

    padding = (ref_t - target_len) // 2
    if padding <= 0:
        return reference

    b, _, c = reference.shape
    # torch: hidden_state[..., padding:-padding]
    return ttnn.slice(reference, [0, padding, 0], [b, ref_t - padding, c])


class OobleckSnake1d(Module):
    """Oobleck ``Snake1d``: ``x + sin(αx)² / β`` with log-scaled α, β (SnakeBeta)."""

    def __init__(
        self,
        channels: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.snake = SnakeBeta(
            channels,
            alpha_logscale=True,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("alpha", "beta"):
            if name in state:
                state[f"snake.{name}"] = state.pop(name)

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        return self.snake(x_BTC)


class OobleckConv1d(Module):
    """Weight-normed Oobleck Conv1d → ``_AlignedOutConv1d`` (symmetric "same" pad)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.unpadded_out_channels = out_channels
        self.conv = _AlignedOutConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding_mode="zeros",
            bias=bias,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        fold_weight_norm_state(state)

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv(x_BTC)


class OobleckConvTranspose1d(Module):
    """Weight-normed Oobleck ConvTranspose1d with torch padding parity.

    Implements ``padding=ceil(stride/2), kernel_size=2*stride`` via zero-stuff +
    symmetric external pad + valid ``Conv1d`` (no causal front pad). Output is
    center-trimmed to ``T_in * stride``, matching diffusers ``OobleckDecoderBlock``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.torch_padding = math.ceil(stride / 2)
        self.external_pad_each = kernel_size - 1 - self.torch_padding
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.conv = _AlignedOutConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding_mode="zeros",
            bias=bias,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        # Zero-stuffed input already carries boundary context; inner conv is valid.
        self.conv.internal_padding = (0, 0, 0)
        self.conv.external_pad_front = 0

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        fold_weight_norm_state(state, conv_prefix="conv.")
        if "conv.weight" in state:
            w = state.pop("conv.weight")
            assert w.dim() == 3 and tuple(w.shape) == (
                self.in_channels,
                self.unpadded_out_channels,
                self.kernel_size,
            ), (
                f"expected ConvTranspose1d weight ({self.in_channels}, "
                f"{self.unpadded_out_channels}, {self.kernel_size}), got {tuple(w.shape)}"
            )
            state["conv.weight"] = torch.flip(w, dims=[-1]).permute(1, 0, 2).contiguous()
        if "conv.bias" in state:
            state["conv.bias"] = state.pop("conv.bias")

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        t_in = int(x_BTC.shape[1])
        expected_t = t_in * self.stride

        x_BTC = _pad_channels_to_aligned(x_BTC, self.mesh_device)
        x_zs = _zero_stuff_t(x_BTC, stride=self.stride, mesh_device=self.mesh_device)
        x_padded = _zero_pad_t(x_zs, self.external_pad_each, self.external_pad_each, self.mesh_device)
        y = self.conv(x_padded)

        if self.unpadded_out_channels < int(y.shape[-1]):
            b, t, _c = y.shape
            y = ttnn.slice(y, [0, 0, 0], [b, t, self.unpadded_out_channels])

        y = _trim_t_to_length(y, expected_t)
        return y


class OobleckResidualUnit(Module):
    """Snake → dilated conv7 → Snake → conv1 with torch-style residual trim/add."""

    def __init__(
        self,
        channels: int,
        *,
        dilation: int = 1,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.snake1 = OobleckSnake1d(channels, mesh_device=mesh_device, dtype=dtype)
        self.conv1 = OobleckConv1d(
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.snake2 = OobleckSnake1d(channels, mesh_device=mesh_device, dtype=dtype)
        self.conv2 = OobleckConv1d(
            channels,
            channels,
            kernel_size=1,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        residual = x_BTC
        h = self.conv1(self.snake1(x_BTC))
        h = self.conv2(self.snake2(h))
        target_len = int(h.shape[1])
        residual = _crop_residual_like_torch(residual, target_len)
        if int(residual.shape[1]) != target_len:
            h = _trim_t_to_length(h, int(residual.shape[1]))
        return ttnn.add(residual, h)


class OobleckDecoderBlock(Module):
    """One upsample stage: Snake → ConvTranspose1d → three residual units."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        kernel_size = 2 * stride
        self.snake1 = OobleckSnake1d(in_channels, mesh_device=mesh_device, dtype=dtype)
        self.conv_t1 = OobleckConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.res_unit1 = OobleckResidualUnit(out_channels, dilation=1, mesh_device=mesh_device, dtype=dtype)
        self.res_unit2 = OobleckResidualUnit(out_channels, dilation=3, mesh_device=mesh_device, dtype=dtype)
        self.res_unit3 = OobleckResidualUnit(out_channels, dilation=9, mesh_device=mesh_device, dtype=dtype)

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        x_BTC = self.snake1(x_BTC)
        x_BTC = self.conv_t1(x_BTC)
        x_BTC = self.res_unit1(x_BTC)
        x_BTC = self.res_unit2(x_BTC)
        x_BTC = self.res_unit3(x_BTC)
        return x_BTC
