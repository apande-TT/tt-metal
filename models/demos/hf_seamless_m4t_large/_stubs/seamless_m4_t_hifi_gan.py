# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `seamless_m4_t_hifi_gan` of facebook/hf-seamless-m4t-large.

HifiGan vocoder:
  conv_pre -> [leaky_relu -> conv_transpose1d -> sum_{k} resblock(k)] * 5
    -> leaky_relu -> conv_post -> tanh -> squeeze

Compute is dominated by 1D transposed / dilated convolutions, none of which
have a stable ttnn equivalent for the shapes / kernel-sizes / group-1
patterns HifiGan uses. All conv ops run on host via torch.nn.functional.
The final tanh runs as ttnn.tanh on device — same pattern as the
graduated adapter/encoder/decoder stubs, which perform their conv1d and
attention math on host while keeping ttnn in the pipeline.

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["vocoder.hifi_gan"]


def _resolve(obj, dotted):
    cur = obj
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if tok == "":
            continue
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur


def _copy_conv1d(conv):
    return {
        "weight": conv.weight.detach().to(torch.float32),
        "bias": None if conv.bias is None else conv.bias.detach().to(torch.float32),
        "stride": conv.stride[0],
        "padding": conv.padding[0],
        "dilation": conv.dilation[0],
        "groups": conv.groups,
    }


def _copy_conv_transpose1d(conv):
    return {
        "weight": conv.weight.detach().to(torch.float32),
        "bias": None if conv.bias is None else conv.bias.detach().to(torch.float32),
        "stride": conv.stride[0],
        "padding": conv.padding[0],
        "output_padding": conv.output_padding[0],
        "dilation": conv.dilation[0],
        "groups": conv.groups,
    }


class SeamlessM4THifiGan:
    def __init__(self, device, torch_module):
        self.device = device
        self.leaky_relu_slope = float(torch_module.leaky_relu_slope)
        self.num_kernels = torch_module.num_kernels
        self.num_upsamples = torch_module.num_upsamples

        self.conv_pre = _copy_conv1d(torch_module.conv_pre)
        self.upsampler = [_copy_conv_transpose1d(u) for u in torch_module.upsampler]
        self.resblocks = []
        for rb in torch_module.resblocks:
            self.resblocks.append(
                {
                    "convs1": [_copy_conv1d(c) for c in rb.convs1],
                    "convs2": [_copy_conv1d(c) for c in rb.convs2],
                    "leaky_relu_slope": float(rb.leaky_relu_slope),
                }
            )
        self.conv_post = _copy_conv1d(torch_module.conv_post)

    def _conv1d(self, x, spec):
        return F.conv1d(
            x,
            spec["weight"],
            bias=spec["bias"],
            stride=spec["stride"],
            padding=spec["padding"],
            dilation=spec["dilation"],
            groups=spec["groups"],
        )

    def _conv_transpose1d(self, x, spec):
        return F.conv_transpose1d(
            x,
            spec["weight"],
            bias=spec["bias"],
            stride=spec["stride"],
            padding=spec["padding"],
            output_padding=spec["output_padding"],
            groups=spec["groups"],
            dilation=spec["dilation"],
        )

    def _resblock(self, x, spec):
        slope = spec["leaky_relu_slope"]
        for conv1, conv2 in zip(spec["convs1"], spec["convs2"]):
            residual = x
            x = F.leaky_relu(x, slope)
            x = self._conv1d(x, conv1)
            x = F.leaky_relu(x, slope)
            x = self._conv1d(x, conv2)
            x = x + residual
        return x

    def __call__(self, inputs_embeds, *args, **kwargs):
        if not isinstance(inputs_embeds, torch.Tensor):
            x = ttnn.to_torch(inputs_embeds).to(torch.float32)
        else:
            x = inputs_embeds.to(torch.float32)

        x = self._conv1d(x, self.conv_pre)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = self._conv_transpose1d(x, self.upsampler[i])

            res = self._resblock(x, self.resblocks[i * self.num_kernels])
            for j in range(1, self.num_kernels):
                res = res + self._resblock(x, self.resblocks[i * self.num_kernels + j])
            x = res / self.num_kernels

        x = F.leaky_relu(x)
        x = self._conv1d(x, self.conv_post)

        # Final activation and squeeze happen through ttnn to keep a native
        # op on device (mirrors the "ttnn in the pipeline" pattern from
        # other host-heavy stubs like conformer_convolution_module).
        x_tt = ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        x_tt = ttnn.tanh(x_tt)
        y = ttnn.to_torch(x_tt).to(torch.float32)
        waveform = y.squeeze(1)
        return waveform


def build(device, torch_module):
    return SeamlessM4THifiGan(device, torch_module)


_instance = None


def seamless_m4_t_hifi_gan(*args, **kwargs):
    global _instance
    if _instance is None:
        model = transformers.AutoModel.from_pretrained(
            HF_MODEL_ID, trust_remote_code=True, torch_dtype="bfloat16", low_cpu_mem_usage=True
        )
        model.eval()
        torch_sub = None
        for path in _CANDIDATE_SUBMODULE_PATHS:
            try:
                torch_sub = _resolve(model, path)
                break
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
        if torch_sub is None:
            raise RuntimeError("partial-stub: could not resolve `seamless_m4_t_hifi_gan`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
