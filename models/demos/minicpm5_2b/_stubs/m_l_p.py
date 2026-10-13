# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `LlamaMLP` (model.layers.0.mlp) for `openbmb/MiniCPM5-2B`.

HF semantics: down(silu(gate(x)) * up(x)), 2048 -> 6144 -> 2048, no biases.
Self-contained: all intermediates and outputs live in DRAM interleaved.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minicpm5_2b._stubs.decode_mm import decode_linear_config


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class TtMLP:
    def __init__(self, device, torch_module):
        self.device = device
        self.compute_cfg = _compute_cfg(device)
        self.mem = ttnn.DRAM_MEMORY_CONFIG

        def _w(linear):
            return ttnn.from_torch(
                linear.weight.detach().to(torch.float32).t().contiguous().unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.w_gate = _w(torch_module.gate_proj)
        self.w_up = _w(torch_module.up_proj)
        self.w_down = _w(torch_module.down_proj)
        h, f = self.w_gate.shape[-2], self.w_gate.shape[-1]
        self.cfg_up = decode_linear_config(device, h, f)
        self.cfg_down = decode_linear_config(device, f, h)

    def _mm(self, x, w, k_block=None):
        """Many-row (prefill) gate/up projection on minimal_matmul's small CBs, which span the full grid under fp32
        accumulation. K_block is all of K, so each tile is one uninterrupted K-order accumulation as in ttnn.linear."""
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=k_block or w.shape[-2] // 32,
                N_block_size=2,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            ),
            memory_config=self.mem,
            compute_kernel_config=self.compute_cfg,
        )

    def _silu_mul(self, gate, up, memory_config=None):
        # silu(gate) * up in one op: the SiLU runs on the bf16 gate inside the multiply, as the separate
        # unary did (ttnn.linear without a core grid applies its activation as a post-op on the bf16 output).
        return ttnn.multiply(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=memory_config or self.mem
        )

    def __call__(self, x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        in_shape = tuple(x.shape)
        if len(in_shape) != 4:
            # Fold every leading dim (batch included) into the matmul's M rows.
            rows = 1
            for d in in_shape[:-1]:
                rows *= d
            x = ttnn.reshape(x, (1, 1, rows, in_shape[-1]))
        if x.shape[-2] > 32 and x.shape[0] == x.shape[1] == 1:
            gate = self._mm(x, self.w_gate)
            up = self._mm(x, self.w_up)
            h = self._silu_mul(gate, up)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            out = self._mm(h, self.w_down, k_block=64)  # K = 6144 does not fit one block: three K blocks
        else:
            one_row = x.shape[0] == x.shape[1] == 1 and x.padded_shape[-2] == 32  # decode: one tile row
            cfg_up, cfg_down = (self.cfg_up, self.cfg_down) if one_row else (None, None)
            mid = ttnn.L1_MEMORY_CONFIG if one_row else self.mem  # decode intermediates fit in L1
            gate = ttnn.linear(
                x, self.w_gate, memory_config=mid, compute_kernel_config=self.compute_cfg, program_config=cfg_up
            )
            up = ttnn.linear(
                x, self.w_up, memory_config=mid, compute_kernel_config=self.compute_cfg, program_config=cfg_up
            )
            h = self._silu_mul(gate, up, memory_config=mid)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            out = ttnn.linear(
                h, self.w_down, memory_config=self.mem, compute_kernel_config=self.compute_cfg, program_config=cfg_down
            )
        ttnn.deallocate(h)
        if len(in_shape) != 4:
            out = ttnn.reshape(out, in_shape[:-1] + (out.shape[-1],))
        return out


def build(device, torch_module=None):
    return TtMLP(device, torch_module)


def m_l_p(device, torch_module=None):
    return TtMLP(device, torch_module)
