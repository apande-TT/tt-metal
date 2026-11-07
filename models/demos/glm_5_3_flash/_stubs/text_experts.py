# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, EXPERT-parallel port of `Glm5NextTextExperts` (GLM-5.3-Flash routed experts).

Reference: for every token t and each of its k routed experts e with weight w:
    out[t] += w * down_e( silu(clamp(gate_e x, max=L)) * clamp(up_e x, -L, L) )

Expert parallelism over the N mesh chips: the E stacked experts are split into N contiguous groups
(ShardTensorToMesh on the expert axis), E/N per chip; a single expert is never split. Each chip:
  1. turns the routing (top_k_index, top_k_weights) into per-local-expert combine weights on device:
     cw[e, t] = sum_k w[t, k] * (idx[t, k] == expert_id[e]), expert ids sharded with the weights;
  2. runs its local experts as one batched SwiGLU over the tokens ([1, E/N, T, *] matmuls);
  3. reduces sum_e cw[e] * y_e -> its partial [T, D].
The partials are summed over the mesh (all_gather + on-device sum) -> replicated output. Tokens an
expert was not routed get weight 0, so the dense batched form is mathematically identical.
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


class TtTextExperts:
    def __init__(self, device, m):
        self.device = device
        self.n = _num_devices(device)
        self.axis = _cluster_axis(device)
        E = int(m.num_experts)
        if E % self.n:
            raise ValueError(f"num_experts={E} not divisible by mesh size {self.n}")
        self.E, self.El = E, E // self.n
        self.D, self.I = int(m.hidden_dim), int(m.intermediate_dim)
        self.limit = float(m.swiglu_limit)
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        gu = m.gate_up_proj.detach()  # [E, 2I, D]
        dn = m.down_proj.detach()  # [E, D, I]
        I = self.I
        # [1, E, in, out] operands, expert axis (dim 1) sharded -> E/N whole experts per chip
        self.w_gate = self._put(gu[:, :I, :].transpose(1, 2).unsqueeze(0), shard_dim=1)
        self.w_up = self._put(gu[:, I:, :].transpose(1, 2).unsqueeze(0), shard_dim=1)
        self.w_down = self._put(dn.transpose(1, 2).unsqueeze(0), shard_dim=1)
        ids = torch.arange(E, dtype=torch.float32).reshape(1, E, 1, 1)
        self.expert_ids = self._put(ids, shard_dim=1, dtype=ttnn.float32)

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

    def _routing(self, top_k_index, top_k_weights, T):
        # Upload the router's outputs (replicated) as fp32 [1,1,T,k]; expert ids < 2^24 are exact.
        def up(x):
            if not isinstance(x, torch.Tensor):
                return ttnn.typecast(ttnn.reshape(x, [1, 1, T, x.shape[-1]]), ttnn.float32)
            return self._put(x.reshape(1, 1, T, -1).to(torch.float32), dtype=ttnn.float32)

        idx, w = up(top_k_index), up(top_k_weights)
        hit = ttnn.eq(idx, self.expert_ids)  # [1,El,T,k]
        return ttnn.sum(ttnn.multiply(hit, w), dim=-1, keepdim=True)  # [1,El,T,1]

    def __call__(self, hidden_states, top_k_index=None, top_k_weights=None, **kwargs):
        x = hidden_states
        in_shape = list(x.shape)
        T, D = in_shape[-2] if len(in_shape) > 1 else 1, in_shape[-1]
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, T, D]), ttnn.bfloat16)
        cw = self._routing(top_k_index, top_k_weights, T)

        xr = ttnn.repeat(x, ttnn.Shape([1, self.El, 1, 1]))  # [1,El,T,D]
        mm = lambda a, b, dt=ttnn.float32: ttnn.matmul(a, b, dtype=dt, compute_kernel_config=self.ckc)
        gate = ttnn.clamp(mm(xr, self.w_gate), max=self.limit)
        up = ttnn.clamp(mm(xr, self.w_up), min=-self.limit, max=self.limit)
        h = ttnn.typecast(ttnn.multiply(ttnn.silu(gate), up), ttnn.bfloat16)
        y = mm(h, self.w_down)  # [1,El,T,D] fp32
        part = ttnn.sum(ttnn.multiply(y, cw), dim=1, keepdim=True)  # [1,1,T,D]

        if self.n > 1:
            g = ttnn.all_gather(
                ttnn.typecast(part, ttnn.bfloat16), dim=1, cluster_axis=self.axis, topology=ttnn.Topology.Linear
            )
            part = ttnn.sum(ttnn.typecast(g, ttnn.float32), dim=1, keepdim=True)
        return ttnn.reshape(part, in_shape)


def build(device, torch_module=None):
    return TtTextExperts.build(device, torch_module)


def text_experts(device, torch_module=None):
    return TtTextExperts.build(device, torch_module)
