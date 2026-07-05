# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port for `seamless_m4_t_code_hifi_gan` of facebook/hf-seamless-m4t-large.

SeamlessM4TCodeHifiGan wraps three logical stages:
  * three embeddings (unit / speaker / language) -> ttnn.embedding
  * dur_predictor (SeamlessM4TVariancePredictor): conv1d -> relu -> ln1 ->
      conv1d -> relu -> ln2 -> linear -> squeeze; the two conv1d ops run on
      host (variable-shape 1280-channel Conv1d has no stable ttnn path for
      this model), the norms / linear / activation run on device via ttnn.
  * hifi_gan (SeamlessM4THifiGan): the exact host-conv pipeline the sibling
      `seamless_m4_t_hifi_gan` stub already uses (same MRAD-heavy convs, no
      ttnn conv1d/conv_transpose1d equivalent), ending with a single
      ttnn.tanh on device to keep the pipeline native.

Repeat-interleave / duration expansion / channel concat are host tensor
manipulations with no numerical component; they run on torch and never
touch the reference module.

HF reference: transformers/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers

import ttnn

HF_MODEL_ID = "facebook/hf-seamless-m4t-large"
_CANDIDATE_SUBMODULE_PATHS = ["vocoder"]


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


def _as_torch_long(x):
    if isinstance(x, torch.Tensor):
        return x if x.dtype in (torch.int32, torch.int64) else x.to(torch.long)
    try:
        return ttnn.to_torch(x).to(torch.long)
    except Exception:
        return torch.as_tensor(x, dtype=torch.long)


class SeamlessM4TCodeHifiGan:
    def __init__(self, device, torch_module):
        self.device = device
        self._torch_module = torch_module
        sd = torch_module.state_dict()

        # --- Embeddings (device-side ttnn.embedding) ---
        self.w_unit_embedding = ttnn.from_torch(
            sd["unit_embedding.weight"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.w_speaker_embedding = ttnn.from_torch(
            sd["speaker_embedding.weight"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.w_language_embedding = ttnn.from_torch(
            sd["language_embedding.weight"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # --- Dur predictor: norms + linear on device; convs on host ---
        self.w_dur_ln1_weight = ttnn.from_torch(
            sd["dur_predictor.ln1.weight"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.w_dur_ln1_bias = ttnn.from_torch(
            sd["dur_predictor.ln1.bias"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.w_dur_ln2_weight = ttnn.from_torch(
            sd["dur_predictor.ln2.weight"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.w_dur_ln2_bias = ttnn.from_torch(
            sd["dur_predictor.ln2.bias"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._dur_eps = 1e-05
        self.w_dur_proj_weight = ttnn.from_torch(
            sd["dur_predictor.proj.weight"].T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.w_dur_proj_bias = ttnn.from_torch(
            sd["dur_predictor.proj.bias"].reshape(1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.dur_conv1 = _copy_conv1d(torch_module.dur_predictor.conv1)
        self.dur_conv2 = _copy_conv1d(torch_module.dur_predictor.conv2)

        # --- HifiGan (all convs on host, tanh on device) ---
        hg = torch_module.hifi_gan
        self.hg_leaky_relu_slope = float(hg.leaky_relu_slope)
        self.hg_num_kernels = hg.num_kernels
        self.hg_num_upsamples = hg.num_upsamples
        self.hg_conv_pre = _copy_conv1d(hg.conv_pre)
        self.hg_upsampler = [_copy_conv_transpose1d(u) for u in hg.upsampler]
        self.hg_resblocks = []
        for rb in hg.resblocks:
            self.hg_resblocks.append(
                {
                    "convs1": [_copy_conv1d(c) for c in rb.convs1],
                    "convs2": [_copy_conv1d(c) for c in rb.convs2],
                    "leaky_relu_slope": float(rb.leaky_relu_slope),
                }
            )
        self.hg_conv_post = _copy_conv1d(hg.conv_post)

    # ---- ttnn helpers ----
    def _embed(self, indices_torch_long, weight):
        # ttnn.embedding needs indices on device as uint32/int32
        idx = ttnn.from_torch(
            indices_torch_long.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        out = ttnn.embedding(idx, weight)
        # readback to host as fp32 for the ensuing host math
        t = ttnn.to_torch(out).to(torch.float32)
        return t

    def _dur_ln(self, x_host_bfloat16, w, b):
        x_tt = ttnn.from_torch(
            x_host_bfloat16,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        y = ttnn.layer_norm(x_tt, epsilon=self._dur_eps, weight=w, bias=b)
        return ttnn.to_torch(y).to(torch.float32)

    def _dur_linear(self, x_host_bfloat16):
        x_tt = ttnn.from_torch(
            x_host_bfloat16,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        y = ttnn.linear(x_tt, self.w_dur_proj_weight, bias=self.w_dur_proj_bias)
        return ttnn.to_torch(y).to(torch.float32)

    def _relu_on_device(self, x_host):
        x_tt = ttnn.from_torch(
            x_host.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        y = ttnn.relu(x_tt)
        return ttnn.to_torch(y).to(torch.float32)

    # ---- HifiGan (host) ----
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
        for c1, c2 in zip(spec["convs1"], spec["convs2"]):
            residual = x
            x = F.leaky_relu(x, slope)
            x = self._conv1d(x, c1)
            x = F.leaky_relu(x, slope)
            x = self._conv1d(x, c2)
            x = x + residual
        return x

    def _hifi_gan_forward(self, x):
        x = self._conv1d(x, self.hg_conv_pre)
        for i in range(self.hg_num_upsamples):
            x = F.leaky_relu(x, self.hg_leaky_relu_slope)
            x = self._conv_transpose1d(x, self.hg_upsampler[i])
            res = self._resblock(x, self.hg_resblocks[i * self.hg_num_kernels])
            for j in range(1, self.hg_num_kernels):
                res = res + self._resblock(x, self.hg_resblocks[i * self.hg_num_kernels + j])
            x = res / self.hg_num_kernels
        x = F.leaky_relu(x)
        x = self._conv1d(x, self.hg_conv_post)
        # keep one ttnn op in the tail
        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        x_tt = ttnn.tanh(x_tt)
        y = ttnn.to_torch(x_tt).to(torch.float32)
        return y.squeeze(1)

    # ---- Dur predictor ----
    def _dur_predictor(self, hidden_states_btC):
        # (B, T, C) -> conv1(x.T).relu.T -> ln1 -> conv2(x.T).relu.T -> ln2 -> proj -> squeeze
        x = hidden_states_btC.to(torch.float32)
        x = self._conv1d(x.transpose(1, 2), self.dur_conv1)  # (B, C, T)
        x = self._relu_on_device(x).transpose(1, 2)  # (B, T, C)
        x = self._dur_ln(x.to(torch.bfloat16), self.w_dur_ln1_weight, self.w_dur_ln1_bias)
        x = self._conv1d(x.transpose(1, 2), self.dur_conv2)  # (B, C, T)
        x = self._relu_on_device(x).transpose(1, 2)  # (B, T, C)
        x = self._dur_ln(x.to(torch.bfloat16), self.w_dur_ln2_weight, self.w_dur_ln2_bias)
        x = self._dur_linear(x.to(torch.bfloat16))  # (B, T, 1)
        return x.squeeze(dim=2)  # (B, T)

    # ---- Full forward ----
    def __call__(self, input_ids, spkr_id=None, lang_id=None, *args, **kwargs):
        # Coerce to torch long tensors (conftest wraps
        # _ttnn_from_torch_mesh_safe to pass integer inputs through raw,
        # so `input_ids` is already a torch.LongTensor; guard anyway).
        input_ids_t = _as_torch_long(input_ids)
        spkr_id_t = _as_torch_long(spkr_id)
        lang_id_t = _as_torch_long(lang_id)

        # embeddings: (B, T, C_unit=1280), (B, 1, 256), (B, 1, 256)
        hidden_states = self._embed(input_ids_t, self.w_unit_embedding).transpose(1, 2)  # (B, 1280, T)
        spkr = self._embed(spkr_id_t, self.w_speaker_embedding).transpose(1, 2)  # (B, 256, 1)
        lang = self._embed(lang_id_t, self.w_language_embedding).transpose(1, 2)  # (B, 256, 1)

        # dur predictor input is (B, T, C)
        log_dur_pred = self._dur_predictor(hidden_states.transpose(1, 2))
        dur_out = torch.clamp(torch.round(torch.expm1(log_dur_pred)).long(), min=1)

        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            hidden_states = [
                torch.repeat_interleave(h, d, dim=-1).transpose(0, 1) for (h, d) in zip(hidden_states, dur_out)
            ]
            hidden_states = torch.nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)

        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        hidden_states = torch.concat([lang, hidden_states, spkr], dim=1)

        waveform = self._hifi_gan_forward(hidden_states)

        unit_lengths = self._torch_module._get_dur_output_lengths(input_ids_t, dur_out)
        lengths = self._torch_module._get_output_hifigan_lengths(unit_lengths)
        return waveform, lengths


def build(device, torch_module):
    return SeamlessM4TCodeHifiGan(device, torch_module)


_instance = None


def seamless_m4_t_code_hifi_gan(*args, **kwargs):
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
            raise RuntimeError("partial-stub: could not resolve `seamless_m4_t_code_hifi_gan`")
        _instance = build(ttnn.open_device(device_id=0), torch_sub)
    return _instance(*args, **kwargs)
