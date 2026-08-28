# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralEncoder (audio_tower).

Full encoder: conv1 + conv2 + positional_embedding + 32 transformer layers + layer_norm.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _to_device(t, device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class TtEncoderLayer:
    def __init__(self, device, torch_layer):
        self.device = device
        attn = torch_layer.self_attn
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.head_dim**-0.5

        self.q_weight = _to_device(attn.q_proj.weight.T.contiguous().float(), device)
        self.q_bias = _to_device(attn.q_proj.bias.unsqueeze(0).float(), device)
        self.k_weight = _to_device(attn.k_proj.weight.T.contiguous().float(), device)
        self.v_weight = _to_device(attn.v_proj.weight.T.contiguous().float(), device)
        self.v_bias = _to_device(attn.v_proj.bias.unsqueeze(0).float(), device)
        self.out_weight = _to_device(attn.out_proj.weight.T.contiguous().float(), device)
        self.out_bias = _to_device(attn.out_proj.bias.unsqueeze(0).float(), device)

        self.attn_ln_w = _to_device(torch_layer.self_attn_layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_b = _to_device(torch_layer.self_attn_layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_eps = torch_layer.self_attn_layer_norm.eps

        self.fc1_weight = _to_device(torch_layer.fc1.weight.T.contiguous().float(), device)
        self.fc1_bias = _to_device(torch_layer.fc1.bias.unsqueeze(0).float(), device)
        self.fc2_weight = _to_device(torch_layer.fc2.weight.T.contiguous().float(), device)
        self.fc2_bias = _to_device(torch_layer.fc2.bias.unsqueeze(0).float(), device)

        self.ffn_ln_w = _to_device(torch_layer.final_layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.ffn_ln_b = _to_device(torch_layer.final_layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.ffn_ln_eps = torch_layer.final_layer_norm.eps

    def __call__(self, x):
        B = x.shape[0]
        S = x.shape[1] if len(x.shape) == 3 else x.shape[-2]

        residual = x
        x = ttnn.layer_norm(
            x, weight=self.attn_ln_w, bias=self.attn_ln_b, epsilon=self.attn_ln_eps, compute_kernel_config=_HIFI4_CFG
        )

        q = ttnn.linear(x, self.q_weight, bias=self.q_bias, compute_kernel_config=_HIFI4_CFG)
        q = ttnn.multiply(q, self.scaling)
        k = ttnn.linear(x, self.k_weight, compute_kernel_config=_HIFI4_CFG)
        v = ttnn.linear(x, self.v_weight, bias=self.v_bias, compute_kernel_config=_HIFI4_CFG)

        q = ttnn.reshape(q, (B, S, self.num_heads, self.head_dim))
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.reshape(k, (B, S, self.num_heads, self.head_dim))
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.reshape(v, (B, S, self.num_heads, self.head_dim))
        v = ttnn.transpose(v, 1, 2)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0, compute_kernel_config=_HIFI4_CFG
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = ttnn.linear(attn_out, self.out_weight, bias=self.out_bias, compute_kernel_config=_HIFI4_CFG)

        x = ttnn.add(residual, attn_out)

        residual = x
        x = ttnn.layer_norm(
            x, weight=self.ffn_ln_w, bias=self.ffn_ln_b, epsilon=self.ffn_ln_eps, compute_kernel_config=_HIFI4_CFG
        )
        x = ttnn.linear(x, self.fc1_weight, bias=self.fc1_bias, compute_kernel_config=_HIFI4_CFG)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias, compute_kernel_config=_HIFI4_CFG)
        x = ttnn.add(residual, x)

        return x


class TtVoxtralEncoder:
    def __init__(self, device, torch_module):
        self.device = device
        self._prepared_w = {}
        self.max_source_positions = torch_module.config.max_source_positions

        self.conv1_weight = ttnn.from_torch(torch_module.conv1.weight.data.float(), dtype=ttnn.bfloat16)
        self.conv1_bias_tt = (
            _to_device(torch_module.conv1.bias.data.reshape(1, 1, 1, -1).float(), device)
            if torch_module.conv1.bias is not None
            else None
        )
        self.conv1_in_ch = torch_module.conv1.in_channels
        self.conv1_out_ch = torch_module.conv1.out_channels
        self.conv1_ks = torch_module.conv1.kernel_size[0]
        self.conv1_stride = torch_module.conv1.stride[0]
        self.conv1_padding = torch_module.conv1.padding[0]

        self.conv2_weight = ttnn.from_torch(torch_module.conv2.weight.data.float(), dtype=ttnn.bfloat16)
        self.conv2_bias_tt = (
            _to_device(torch_module.conv2.bias.data.reshape(1, 1, 1, -1).float(), device)
            if torch_module.conv2.bias is not None
            else None
        )
        self.conv2_in_ch = torch_module.conv2.in_channels
        self.conv2_out_ch = torch_module.conv2.out_channels
        self.conv2_ks = torch_module.conv2.kernel_size[0]
        self.conv2_stride = torch_module.conv2.stride[0]
        self.conv2_padding = torch_module.conv2.padding[0]

        self.embed_positions = _to_device(torch_module.embed_positions.weight.unsqueeze(0).float(), device)

        self.layers = [TtEncoderLayer(device, layer) for layer in torch_module.layers]

        self.ln_weight = _to_device(torch_module.layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.ln_bias = _to_device(torch_module.layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.ln_eps = torch_module.layer_norm.eps

    def _conv1d_cached(self, x, idx, weight, in_ch, out_ch, ks, stride, pad, length):
        """conv1d with the PREPROCESSED weights cached on device.

        The graduated body kept the conv weights on host and let every call
        upload/prepare them.  That host transfer is illegal inside
        ttnn.begin_trace_capture (TT_FATAL !trace_id_.has_value()), so the encode
        stage could not be traced.  Preparing once and reusing the device-resident
        weights is also strictly faster; numerics are unchanged.
        """
        prepared = self._prepared_w.get(idx)
        res = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=prepared if prepared is not None else weight,
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=1,
            input_length=length,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            dilation=1,
            groups=1,
            return_weights_and_bias=prepared is None,
        )
        if prepared is None:
            out = res[0]
            wb = res[-1]
            self._prepared_w[idx] = wb[0] if isinstance(wb, (tuple, list)) else wb
        else:
            out = res[0] if isinstance(res, tuple) else res
        return out

    def __call__(self, input_features, **kwargs):
        # input_features: ttnn tensor (1, 128, 3000) TILE_LAYOUT on device
        # conv1d expects (N, input_length, 1, C) format
        x = ttnn.to_layout(input_features, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.permute(x, (0, 2, 1))  # (1, 3000, 128)
        x = ttnn.reshape(x, (1, 3000, 1, 128))  # (N, L, 1, C)

        # conv1: (1, 3000, 1, 128) -> (1, 3000, 1, 1280)
        x = self._conv1d_cached(
            x,
            1,
            self.conv1_weight,
            self.conv1_in_ch,
            self.conv1_out_ch,
            self.conv1_ks,
            self.conv1_stride,
            self.conv1_padding,
            3000,
        )
        if self.conv1_bias_tt is not None:
            x = ttnn.add(x, self.conv1_bias_tt)
        x = ttnn.gelu(x)

        # conv2: stride=2, so 3000 -> 1500
        x = ttnn.reshape(x, (1, 3000, 1, 1280))
        x = self._conv1d_cached(
            x,
            2,
            self.conv2_weight,
            self.conv2_in_ch,
            self.conv2_out_ch,
            self.conv2_ks,
            self.conv2_stride,
            self.conv2_padding,
            3000,
        )
        if self.conv2_bias_tt is not None:
            x = ttnn.add(x, self.conv2_bias_tt)
        x = ttnn.gelu(x)

        # Reshape to (1, 1500, 1280)
        x = ttnn.reshape(x, (1, 1500, 1280))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Add positional embedding
        x = ttnn.add(x, self.embed_positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = ttnn.layer_norm(
            x, weight=self.ln_weight, bias=self.ln_bias, epsilon=self.ln_eps, compute_kernel_config=_HIFI4_CFG
        )

        return x


def build(device, torch_module=None):
    return TtVoxtralEncoder(device, torch_module)
