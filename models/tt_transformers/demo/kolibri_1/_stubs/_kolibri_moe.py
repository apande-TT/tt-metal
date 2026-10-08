# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN Kolibri-1 sparse MoE block (`model.layers.N.mlp`), expert-parallel. Shared by the stubs.

Reference (`Kolibri1SparseMoeBlock` in tests/pcc/_reference_loader.py):

    logits  = x.float() @ gate.weight.T                         # fp32 router, 384 experts
    ids     = topk(logits + e_score_correction_bias, 6)         # bias only SELECTS
    w       = sigmoid(logits[ids])                              # unbiased, no renormalisation
    out     = sum_k w_k * expert_{ids_k}(x) + shared_expert(x)  # SwiGLU experts, I = 512

Routing runs replicated on every chip in fp32: the 6th-largest biased score per token is the
selection threshold, so the dense weight table is sigmoid(logits) where score >= threshold, else 0.

Expert parallel over the last mesh axis (TP chips): each chip owns a contiguous 1/TP of the experts
and evaluates them on every token as two wide matmuls -- x @ [gate | up] of all local experts, then
SwiGLU -- scales each expert's 512 activation columns by its routing weight (an unrouted expert gets
0 and contributes nothing), and one down matmul over the stacked local experts sums them. The shared
expert is split the usual way (gate/up column-parallel, down row-parallel), so its partial sum joins
the routed partial sum and a single all_reduce combines both across chips.
"""
from __future__ import annotations

import torch

import ttnn

_EXPAND_CACHE = {}


class TtKolibri1MoE:
    def __init__(self, device, moe, expert_dtype=ttnn.bfloat8_b) -> None:
        self.device = device
        self.top_k = int(moe.top_k)
        self.norm_topk_prob = bool(moe.norm_topk_prob)
        n_exp = len(moe.experts)
        inter = int(moe.experts[0].gate_proj.out_features)
        s_inter = int(moe.shared_experts.gate_proj.out_features)

        self._is_mesh = isinstance(device, ttnn.MeshDevice)
        mesh_shape = list(device.shape) if self._is_mesh else [1, 1]
        tp = mesh_shape[-1]
        if n_exp % tp or s_inter % (tp * 32):
            tp = 1
        self.tp = tp
        self.tp_axis = len(mesh_shape) - 1
        self.mesh_shape = mesh_shape
        self.n_local = n_exp // tp
        self.inter = inter

        bf16 = torch.bfloat16
        # [hidden, n_exp * inter]: expert e owns columns e*inter .. (e+1)*inter, so a TP split of the
        # columns hands chip d exactly experts d*n_local .. (d+1)*n_local.
        gate = torch.cat([e.gate_proj.dequantize(bf16) for e in moe.experts], 0).t()
        up = torch.cat([e.up_proj.dequantize(bf16) for e in moe.experts], 0).t()
        down = torch.cat([e.down_proj.dequantize(bf16).t() for e in moe.experts], 0)  # [n_exp*inter, hidden]
        self.w_gate = self._upload(gate, -1, expert_dtype)
        self.w_up = self._upload(up, -1, expert_dtype)
        self.w_down = self._upload(down, -2, expert_dtype)
        del gate, up, down
        # Routing weight of expert e -> its inter activation columns (row e of a block-diagonal ones).
        # Identical for every layer, so one copy per device serves them all.
        key = (id(device), n_exp, inter, tp)
        if key not in _EXPAND_CACHE:
            expand = torch.kron(torch.eye(n_exp, dtype=bf16), torch.ones(1, inter, dtype=bf16))
            _EXPAND_CACHE[key] = self._upload(expand, -1, ttnn.bfloat16)
        self.expand = _EXPAND_CACHE[key]

        sh = moe.shared_experts
        self.ws_gate = self._upload(sh.gate_proj.dequantize(bf16).t(), -1, expert_dtype)
        self.ws_up = self._upload(sh.up_proj.dequantize(bf16).t(), -1, expert_dtype)
        self.ws_down = self._upload(sh.down_proj.dequantize(bf16).t(), -2, expert_dtype)

        self.w_router = self._upload(moe.gate.weight.float().t(), None, ttnn.bfloat16)
        self.router_bias = self._upload(moe.gate.e_score_correction_bias.float().reshape(1, -1), None, ttnn.float32)

        arch = device.arch()
        self.hifi4 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.hifi2 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _upload(self, t, shard_dim, dtype):
        mapper = None
        if self._is_mesh:
            if shard_dim is not None and self.tp > 1:
                mapper = ttnn.ShardTensor2dMesh(self.device, mesh_shape=self.mesh_shape, dims=(None, shard_dim))
            else:
                mapper = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=mapper
        )

    def routing_weights(self, x):
        """Dense [1, 1, N, n_exp] fp32 routing weights: sigmoid(logit) on the top-k experts, 0 elsewhere."""
        logits = ttnn.linear(x, self.w_router, dtype=ttnn.float32, compute_kernel_config=self.hifi4)
        scores = ttnn.add(logits, self.router_bias)
        top_vals, top_ids = ttnn.topk(scores, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(top_ids)
        n = scores.shape[-2]
        kth = ttnn.slice(top_vals, [0, 0, 0, self.top_k - 1], [1, 1, n, self.top_k])
        chosen = ttnn.ge(scores, kth)
        weights = ttnn.multiply(ttnn.sigmoid(logits), chosen)
        if self.norm_topk_prob:
            weights = ttnn.divide(weights, ttnn.add(ttnn.sum(weights, dim=-1, keepdim=True), 1e-20))
        return weights

    def __call__(self, x):
        shape = list(x.shape)
        x = ttnn.reshape(x, [1, 1, -1, shape[-1]])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)

        weights = self.routing_weights(x)
        col_scale = ttnn.linear(weights, self.expand, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)
        ttnn.deallocate(weights)
        g = ttnn.linear(x, self.w_gate, compute_kernel_config=self.hifi2)
        u = ttnn.linear(x, self.w_up, compute_kernel_config=self.hifi2)
        act = ttnn.multiply(ttnn.multiply(ttnn.silu(g), u), col_scale)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        ttnn.deallocate(col_scale)
        out = ttnn.linear(act, self.w_down, compute_kernel_config=self.hifi2)
        ttnn.deallocate(act)

        sg = ttnn.linear(x, self.ws_gate, compute_kernel_config=self.hifi2)
        su = ttnn.linear(x, self.ws_up, compute_kernel_config=self.hifi2)
        s_act = ttnn.multiply(ttnn.silu(sg), su)
        out = ttnn.add(out, ttnn.linear(s_act, self.ws_down, compute_kernel_config=self.hifi2))

        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.reshape(out, shape)
