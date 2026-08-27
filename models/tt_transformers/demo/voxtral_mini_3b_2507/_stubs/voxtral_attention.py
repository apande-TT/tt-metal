# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralAttention (audio_tower.layers[i].self_attn).

Standard multi-head self-attention for the audio encoder.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _as_bf16(t):
    """Narrow a weight to bfloat16 ON THE HOST, before it is ever uploaded.

    ``from_torch(t, dtype=ttnn.bfloat16, ...)`` does NOT convert a float32 `t` on the host: it
    uploads the fp32 bytes, converts the LAYOUT on device in fp32, and only then emits a device
    Typecast to bf16.  Every call site here hands over a `.float()` tensor, so the layout
    conversion was moving 4 bytes per element to produce a 2-byte tensor and paying for a whole
    extra device op to do it.  Handing over bf16 makes the requested dtype the dtype that arrives:
    the conversion moves half the bytes and the typecast has nothing left to do.

    The VALUES are the same either way -- the fp32 -> bf16 rounding happens regardless, on the host
    here instead of on the device one op later.
    """
    return t.bfloat16() if hasattr(t, "bfloat16") else t


def _to_device(t, device):
    t = _as_bf16(t)
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


class TtVoxtralAttention:
    def __init__(self, device, torch_module):
        self.device = device
        self.num_heads = torch_module.num_heads
        self.head_dim = torch_module.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_weight = _to_device(torch_module.q_proj.weight.T.contiguous().float(), device)
        self.k_weight = _to_device(torch_module.k_proj.weight.T.contiguous().float(), device)
        self.v_weight = _to_device(torch_module.v_proj.weight.T.contiguous().float(), device)
        self.o_weight = _to_device(torch_module.out_proj.weight.T.contiguous().float(), device)

        self.q_bias = (
            _to_device(torch_module.q_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.q_proj.bias is not None
            else None
        )
        self.k_bias = (
            _to_device(torch_module.k_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.k_proj.bias is not None
            else None
        )
        self.v_bias = (
            _to_device(torch_module.v_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.v_proj.bias is not None
            else None
        )
        self.o_bias = (
            _to_device(torch_module.out_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.out_proj.bias is not None
            else None
        )

    def __call__(self, hidden_states, **kwargs):
        B = hidden_states.shape[0]
        S = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[-2]

        q = ttnn.linear(hidden_states, self.q_weight, bias=self.q_bias, compute_kernel_config=_HIFI4_CFG)
        k = ttnn.linear(hidden_states, self.k_weight, bias=self.k_bias, compute_kernel_config=_HIFI4_CFG)
        v = ttnn.linear(hidden_states, self.v_weight, bias=self.v_bias, compute_kernel_config=_HIFI4_CFG)

        q = ttnn.reshape(q, (B, S, self.num_heads, self.head_dim))
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.reshape(k, (B, S, self.num_heads, self.head_dim))
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.reshape(v, (B, S, self.num_heads, self.head_dim))
        v = ttnn.transpose(v, 1, 2)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scaling, compute_kernel_config=_HIFI4_CFG
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = ttnn.linear(attn_out, self.o_weight, bias=self.o_bias, compute_kernel_config=_HIFI4_CFG)

        return attn_out


def build(device, torch_module):
    return TtVoxtralAttention(device, torch_module)
