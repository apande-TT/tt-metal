# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, head-parallel port of `Glm5NextTextForgetGate` (GLM-5.3-Flash KDA forget gate).

Reference (safe lower-bound branch, the one this checkpoint uses):
    f = f_b_proj(f_a_proj(x)) + dt_bias                     # [B,S,H*dk]
    g = lower_bound * sigmoid(exp(A_log)[head] * f)         # -> view [B,S,H,dk]

Tensor parallelism over N chips (H heads -> H/N per chip):
  * f_a_proj (D -> dk, low rank) REPLICATED.
  * f_b_proj COLUMN-parallel by head (ShardTensorToMesh dim=-1); dt_bias and the per-channel
    exp(A_log) expansion are sharded the same way so every chip owns whole heads.
  * the gate is elementwise per channel, so no reduction is needed: an all_gather on the channel
    axis reassembles the full [B,S,H,dk] output on every chip.
"""
from __future__ import annotations

import torch

import ttnn


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


def _cluster_axis(device):
    try:
        shape = tuple(device.shape)
    except Exception:
        return None
    axes = [i for i, s in enumerate(shape) if s > 1]
    return axes[0] if len(axes) == 1 else None


class TtTextForgetGate:
    def __init__(self, device, m):
        self.device = device
        self.n = _num_devices(device)
        self.axis = _cluster_axis(device)
        self.H, self.dk = int(m.num_heads), int(m.head_dim)
        if self.H % self.n:
            raise ValueError(f"num_heads={self.H} not divisible by mesh size {self.n}")
        if m.safe_gate_lower_bound is None:
            raise NotImplementedError("softplus forget-gate branch is not used by this checkpoint")
        self.lb = float(m.safe_gate_lower_bound)
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        qkv = self.H * self.dk
        self.w_a = self._put(m.f_a_proj.weight.detach().t())
        self.w_b = self._put(m.f_b_proj.weight.detach().t(), shard_dim=-1)
        self.dt_bias = self._put(m.dt_bias.detach().reshape(1, 1, 1, qkv), shard_dim=-1, dtype=ttnn.float32)
        decay = torch.exp(m.A_log.detach().to(torch.float32)).repeat_interleave(self.dk)
        self.decay = self._put(decay.reshape(1, 1, 1, qkv), shard_dim=-1, dtype=ttnn.float32)

    def _put(self, t, *, shard_dim=None, dtype=ttnn.bfloat16):
        kw = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.n > 1:
            kw["mesh_mapper"] = (
                ttnn.ShardTensorToMesh(self.device, dim=shard_dim)
                if shard_dim is not None
                else ttnn.ReplicateTensorToMesh(self.device)
            )
        t = t.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32).contiguous()
        return ttnn.from_torch(t, **kw)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def local(self, x):
        """Chip-local half of the gate: x [1,1,R,D] (TILE) -> g [1,1,R,H*dk/N] fp32 for this chip's heads.
        The head-parallel KDA consumes exactly this (its heads are the same chip-local heads); __call__
        adds the all_gather that reassembles every head."""
        x = ttnn.typecast(x, ttnn.bfloat16)
        lin = lambda a, w, dt: ttnn.linear(a, w, dtype=dt, compute_kernel_config=self.ckc)
        f = lin(lin(x, self.w_a, ttnn.bfloat16), self.w_b, ttnn.float32)  # [1,1,R,H*dk/N]
        return ttnn.multiply(ttnn.sigmoid(ttnn.multiply(ttnn.add(f, self.dt_bias), self.decay)), self.lb)

    def __call__(self, hidden_states, *args, **kwargs):
        x = hidden_states
        B, S, D = x.shape
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        g = self.local(ttnn.reshape(x, [1, 1, B * S, D]))
        if self.n > 1:
            # Gather on dim 1 in bf16 -- the collective form proven on the 1x32 Galaxy line.
            # all_gather DEADLOCKS there when each chip contributes too few tiles (measured:
            # [1,1,64,256] = 16 tiles hangs, [1,1,64,1024] = 64 tiles completes), so zero-pad each
            # chip's slice to >= 64 tiles first and drop the padding after the gather.
            W = g.shape[-1]
            row_tiles = max(1, -(-(B * S) // 32))
            Wp = max(W, -(-64 // row_tiles) * 32)
            gb = ttnn.typecast(g, ttnn.bfloat16)
            if Wp > W:
                gb = ttnn.pad(gb, [(0, 0), (0, 0), (0, 0), (0, Wp - W)], 0.0)
            gb = ttnn.all_gather(gb, dim=1, cluster_axis=self.axis, topology=ttnn.Topology.Linear)
            if Wp > W:
                gb = ttnn.slice(gb, [0, 0, 0, 0], [1, self.n, B * S, W])
            # [1,N,BS,c] chip-major -> [1,BS,N,c]: each token's channels back in global head order.
            g = ttnn.permute(ttnn.typecast(gb, ttnn.float32), (0, 2, 1, 3))
        g = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(g, [B, S, self.H, self.dk])


def build(device, torch_module=None):
    return TtTextForgetGate.build(device, torch_module)


def text_forget_gate(device, torch_module=None):
    return TtTextForgetGate.build(device, torch_module)
