# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE ttnn port of `patch_embed` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

`patch_embed` is one of the scaffold's generic transformer roles and none of its
candidate paths exists on `diffusers.AutoencoderKLFlux2`. The config's
`patch_size=[2, 2]` is NOT a module: it is a reshape of the LATENT performed by
the wrapper (`Flux2VaeDecoder.preprocess_and_unpatchify` on the way out), so
there is no patch-embedding layer to port. The real image -> channels step of
this model is the encoder stem, so the role is bound — in `bringup_status.json`
and in the test's candidate list — to

    encoder.conv_in   (torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1))

Unlike every other component here this one is rung `emit`, i.e. it runs on a
SINGLE device, not the TP mesh (a bare conv over 3 input channels has nothing to
shard: 3 does not divide by 8, and its 128 outputs give a 16-wide sub-tile shard
that TILE layout cannot slice). So it is written against `ttnn.conv2d` directly
rather than `models/tt_dit/layers/conv2d.py::Conv2d`, whose mesh bookkeeping
would ask a single `Device` for a mesh `.shape`.

`ttnn.conv2d` consumes NHWC activations and an OIHW weight (which it prepares on
first call and hands back for reuse), and returns the result flattened to
`[1, 1, N*H*W, C_out]`; the golden is NCHW, so the stub permutes in and out.
"""
from __future__ import annotations

import torch

import ttnn


class TtPatchEmbed:
    """`encoder.conv_in` as a bare `ttnn.conv2d`."""

    def __init__(self, device, *, weight, bias, out_channels, kernel_size, stride, padding) -> None:
        self.device = device
        self._weight = weight
        self._bias = bias
        self.out_channels = int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._conv_config = ttnn.Conv2dConfig(act_block_h_override=32, weights_dtype=ttnn.bfloat16)
        self._compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "patch_embed stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        # Weight staging: OIHW as ttnn.conv2d wants it, bias as [1, 1, 1, C_out].
        weight = ttnn.from_torch(
            torch_module.weight.detach().to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        bias = None
        if torch_module.bias is not None:
            bias = ttnn.from_torch(
                torch_module.bias.detach().to(torch.float32).reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        return cls(
            device,
            weight=weight,
            bias=bias,
            out_channels=torch_module.out_channels,
            kernel_size=tuple(torch_module.kernel_size),
            stride=tuple(torch_module.stride),
            padding=tuple(torch_module.padding),
        )

    def __call__(self, x, *args, **kwargs):
        # NCHW -> NHWC. permute needs ROW_MAJOR, and 3 channels are not
        # tile-aligned anyway, so the conv is fed ROW_MAJOR (it accepts either).
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.permute(x, (0, 2, 3, 1))
        batch, height, width, in_channels = (int(d) for d in x.shape)

        y, (out_height, out_width), (self._weight, self._bias) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._weight,
            bias_tensor=self._bias,
            device=self.device,
            in_channels=in_channels,
            out_channels=self.out_channels,
            batch_size=batch,
            input_height=height,
            input_width=width,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self._conv_config,
            compute_config=self._compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        y = ttnn.reshape(y, (batch, out_height, out_width, self.out_channels))
        if y.layout != ttnn.ROW_MAJOR_LAYOUT:
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.permute(y, (0, 3, 1, 2))


def build(device, torch_module=None):
    return TtPatchEmbed.build(device, torch_module)


def patch_embed(device, torch_module=None):
    return TtPatchEmbed.build(device, torch_module)
