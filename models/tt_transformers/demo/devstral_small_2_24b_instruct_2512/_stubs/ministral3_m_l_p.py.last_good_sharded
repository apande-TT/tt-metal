# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_m_l_p` (Ministral3MLP, `language_model.layers.0.mlp`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    out = down(silu(gate(x)) * up(x))            (5120 -> 32768 -> 5120)

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * gate/up    column-parallel on the intermediate dim (the SwiGLU product stays per-device).
  * down_proj  row-parallel on the intermediate dim, then all_reduce.
"""
from __future__ import annotations

import torch
import ttnn

# Prefill token rows (batch rows x sequence) one MLP chunk processes at a time.
PREFILL_ROWS_PER_CHUNK = 4096


class TtMinistral3MLP:
    # dtype of the down projection's output and its all_reduce: fp32, added unrounded into the fp32 residual
    # stream of the text decoder (and of the vision layers, whose MLPs subclass this one).
    OUT_DTYPE = ttnn.float32

    def __init__(self, device, torch_module) -> None:
        self.device = device
        try:
            self.tp = tp = int(device.get_num_devices())
        except Exception:
            self.tp = tp = 1
        assert torch_module.intermediate_size % tp == 0, (
            f"intermediate {torch_module.intermediate_size} not divisible by TP={tp}"
        )
        shard = lambda dim: ttnn.ShardTensorToMesh(device, dim=dim) if tp > 1 else None

        def _put(t, mapper):
            return ttnn.from_torch(
                t.detach().to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.w_gate = _put(torch_module.gate_proj.weight.T, shard(-1))
        self.w_up = _put(torch_module.up_proj.weight.T, shard(-1))
        self.w_down = _put(torch_module.down_proj.weight.T, shard(-2))
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, *args, **kwargs):
        shape = tuple(x.shape)
        if len(shape) != 4:
            x = ttnn.reshape(x, (1,) * (4 - len(shape)) + shape)
        b, _, s, d = x.shape
        step = max(1, PREFILL_ROWS_PER_CHUNK // max(1, s))
        if b > step:
            # Prefill of a large batch: run the 32768-wide intermediate a few batch rows at a time so its
            # peak DRAM stays small (rows are independent, so the result is identical).
            parts = [self._forward(ttnn.slice(x, [i, 0, 0, 0], [min(i + step, b), 1, s, d])) for i in range(0, b, step)]
            return ttnn.reshape(ttnn.concat(parts, dim=0), shape)
        return ttnn.reshape(self._forward(x), shape)

    def _forward(self, x):
        g = ttnn.linear(x, self.w_gate, compute_kernel_config=self.compute_cfg)
        u = ttnn.linear(x, self.w_up, compute_kernel_config=self.compute_cfg)
        m = ttnn.multiply(ttnn.silu(g), u)
        out = ttnn.linear(m, self.w_down, dtype=self.OUT_DTYPE, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, topology=ttnn.Topology.Linear)
        return out


def build(device, torch_module=None):
    return TtMinistral3MLP.build(device, torch_module)


def ministral3_m_l_p(device, torch_module=None):
    return TtMinistral3MLP.build(device, torch_module)
