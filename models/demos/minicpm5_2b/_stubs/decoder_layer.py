# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `LlamaDecoderLayer` for `openbmb/MiniCPM5-2B`.

HF semantics reproduced (pre-norm):
  h   = x + self_attn(input_layernorm(x))
  out = h + mlp(post_attention_layernorm(h))
  mlp(x) = down(silu(gate(x)) * up(x)); RMSNorm in fp32 with the module's epsilon.
Attention reuses the graduated native `TtAttention` port.

`parts` lets a caller hand in already-built sub-blocks (keys input_layernorm, self_attn,
post_attention_layernorm, mlp) instead of building this file's own; anything not supplied is built
here as before. Extra keyword arguments (the resident KV cache and its context) go to the attention.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minicpm5_2b._stubs.attention import TtAttention


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class TtRMSNorm:
    def __init__(self, device, torch_module):
        self.eps = getattr(torch_module, "variance_epsilon", None) or getattr(torch_module, "eps", 1e-6)
        self.compute_cfg = _compute_cfg(device)
        w = torch_module.weight.detach().to(torch.float32).reshape(1, 1, -1)
        self.weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x, memory_config=None):
        return ttnn.rms_norm(
            x, epsilon=self.eps, weight=self.weight, compute_kernel_config=self.compute_cfg, memory_config=memory_config
        )


class TtMLP:
    def __init__(self, device, torch_module):
        self.compute_cfg = _compute_cfg(device)

        def _w(linear):
            return ttnn.from_torch(
                linear.weight.detach().to(torch.float32).t().contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.w_gate = _w(torch_module.gate_proj)
        self.w_up = _w(torch_module.up_proj)
        self.w_down = _w(torch_module.down_proj)

    def __call__(self, x):
        gate = ttnn.linear(x, self.w_gate, compute_kernel_config=self.compute_cfg)
        up = ttnn.linear(x, self.w_up, compute_kernel_config=self.compute_cfg)
        h = ttnn.multiply(ttnn.silu(gate), up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(h, self.w_down, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(h)
        return out


class TtDecoderLayer:
    def __init__(self, device, torch_module, parts=None):
        self.device = device
        parts = parts or {}
        build = {
            "input_layernorm": lambda: TtRMSNorm(device, torch_module.input_layernorm),
            "post_attention_layernorm": lambda: TtRMSNorm(device, torch_module.post_attention_layernorm),
            "self_attn": lambda: TtAttention(device, torch_module.self_attn),
            "mlp": lambda: TtMLP(device, torch_module.mlp),
        }
        for name, make in build.items():
            setattr(self, name, parts[name] if name in parts else make())
        bind = getattr(self.self_attn, "bind_prenorm", None)
        self._attn_prenorm = bool(bind and bind(self.input_layernorm))
        bind = getattr(self.mlp, "bind_prenorm", None)
        self._mlp_prenorm = bool(bind and bind(self.post_attention_layernorm))

    def __call__(self, hidden_states, position_embeddings=None, past_key_values=None, **kwargs):
        x = hidden_states
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[-2], x.shape[-1]))

        # A decode step's residual stream is one tile row: keep it in L1 (prefill's stays in DRAM). The norm
        # outputs feed only the next projection, so they are L1 in both stages (one 16 MB prefill tensor still
        # fits beside minimal_matmul's static CBs).
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[0] == 1 and x.padded_shape[-2] == 32 else ttnn.DRAM_MEMORY_CONFIG
        l1 = ttnn.L1_MEMORY_CONFIG
        if self._attn_prenorm:
            # The attention applies the input norm itself (inside its decode QKV program when it can).
            attn_out, _ = self.self_attn(
                x,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                prenorm=self.input_layernorm,
                **kwargs,
            )
        else:
            attn_out, _ = self.self_attn(
                self.input_layernorm(x, memory_config=l1),
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )
        # The mid-layer residual lives in L1 in both stages: in prefill it sits beside at most one other 16 MB L1
        # tensor during the MLP's minimal_matmuls, which still clears their static CBs.
        h = ttnn.add(x, attn_out, memory_config=l1)
        ttnn.deallocate(attn_out)
        if self._mlp_prenorm:
            # The MLP applies the post-attention norm itself (inside its decode gate/up program when it can).
            mlp_out = self.mlp(h, prenorm=self.post_attention_layernorm)
        else:
            mlp_out = self.mlp(self.post_attention_layernorm(h, memory_config=l1))
        out = ttnn.add(h, mlp_out, memory_config=mem)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(h)
        return out

    @classmethod
    def build(cls, device, torch_module, parts=None):
        return cls(device, torch_module, parts=parts)


def build(device, torch_module=None, parts=None):
    return TtDecoderLayer.build(device, torch_module, parts=parts)


def decoder_layer(device, torch_module=None, parts=None):
    return TtDecoderLayer.build(device, torch_module, parts=parts)
