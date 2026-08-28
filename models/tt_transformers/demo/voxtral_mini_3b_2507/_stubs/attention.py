# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralAttention (audio encoder self-attention).

Maps to: audio_tower.layers[i].self_attn
Standard multi-head self-attention: Q/K/V projections -> SDPA -> output projection.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


class TtVoxtralAttention:
    def __init__(self, device, torch_module):
        self.device = device
        self.num_heads = torch_module.num_heads
        self.head_dim = torch_module.head_dim
        self.embed_dim = torch_module.embed_dim
        self.scaling = torch_module.head_dim**-0.5

        self.q_weight = ttnn.from_torch(
            torch_module.q_proj.weight.T.contiguous().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.q_bias = ttnn.from_torch(
            torch_module.q_proj.bias.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.k_weight = ttnn.from_torch(
            torch_module.k_proj.weight.T.contiguous().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.v_weight = ttnn.from_torch(
            torch_module.v_proj.weight.T.contiguous().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.v_bias = ttnn.from_torch(
            torch_module.v_proj.bias.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.out_weight = ttnn.from_torch(
            torch_module.out_proj.weight.T.contiguous().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.out_bias = ttnn.from_torch(
            torch_module.out_proj.bias.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, hidden_states, **kwargs):
        bsz = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[-2]

        q = ttnn.linear(hidden_states, self.q_weight, bias=self.q_bias, compute_kernel_config=_HIFI4_CFG)
        q = ttnn.multiply(q, self.scaling)

        k = ttnn.linear(hidden_states, self.k_weight, compute_kernel_config=_HIFI4_CFG)

        v = ttnn.linear(hidden_states, self.v_weight, bias=self.v_bias, compute_kernel_config=_HIFI4_CFG)

        q = ttnn.reshape(q, (bsz, seq_len, self.num_heads, self.head_dim))
        q = ttnn.transpose(q, 1, 2)

        k = ttnn.reshape(k, (bsz, seq_len, self.num_heads, self.head_dim))
        k = ttnn.transpose(k, 1, 2)

        v = ttnn.reshape(v, (bsz, seq_len, self.num_heads, self.head_dim))
        v = ttnn.transpose(v, 1, 2)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0, compute_kernel_config=_HIFI4_CFG
        )

        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        attn_output = ttnn.linear(attn_output, self.out_weight, bias=self.out_bias, compute_kernel_config=_HIFI4_CFG)

        return attn_output


def build(device, torch_module=None):
    return TtVoxtralAttention(device, torch_module)
