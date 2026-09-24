# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN NemotronH expert bank for `nemotron_h_experts` (`mixer.experts`,
the un-gated MLP expert collection inside `NemotronHMoE` -- distinct from the
already-graduated `nemotron_h_mo_e` sibling, which wraps gate+experts+shared
expert as one component; this one is just the raw expert compute).

HF reference (`NemotronHExperts.forward`, modeling_nemotron_h.py):

    forward(hidden_states[T,H], top_k_index[T,K] int64, top_k_weights[T,K] fp32):
        for each expert e that has >=1 token routed to it:
            x = hidden_states[tokens routed to e]
            y = down_proj[e]( relu(up_proj[e](x)) ** 2 )   # relu2, no bias
            y *= top_k_weights[those tokens, their slot]
            scatter-add y back into the per-token output

That sparse gather/scatter is mathematically identical to the DENSE form used
here (and by the graduated `nemotron_h_mo_e` sibling): build a dense
(tokens, num_experts) routing-weight matrix from (top_k_index, top_k_weights)
(zero for experts a token didn't select), evaluate every expert on every
token, and weight-sum. `top_k_index`/`top_k_weights` never reach this stub as
device tensors (the PCC harness only converts the PRIMARY arg -- here
`hidden_states` -- via `ttnn.from_torch`; the rest stay plain torch), so the
scatter that builds the dense routing matrix runs on host, and only the
resulting (tokens, num_experts) matrix is uploaded.

Tensor-parallel (TP=2): EXPERT-parallel, mirroring `nemotron_h_mo_e` --
`up_proj`/`down_proj` are already native 3D (num_experts, ...) tensors, so
they shard directly on the expert axis (dim 0); each chip evaluates its local
E/TP experts against its own routing-matrix columns and the partial mixture
is all_reduced to recover the full sum.
"""
from __future__ import annotations

import math

import torch

import ttnn
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import _dram_mm


class TtNemotronHExperts:
    def __init__(self, device, torch_module) -> None:
        self.device = device

        self.num_experts = int(torch_module.num_experts)
        self.hidden_dim = int(torch_module.hidden_dim)
        self.intermediate_dim = int(torch_module.intermediate_dim)
        E = self.num_experts

        # nn.Linear-style weights (out, in); store pre-transposed (in, out) so
        # forward can do `x @ W` directly, matching F.linear(x, W) = x @ W.T.
        up_t = torch_module.up_proj.detach().float().transpose(-1, -2).contiguous()  # (E, hidden, inter)
        down_t = torch_module.down_proj.detach().float().transpose(-1, -2).contiguous()  # (E, inter, hidden)

        # ---- tensor-parallel (expert-parallel) config ----------------------
        import os as _os

        dev = self.device
        try:
            _is_mesh = isinstance(dev, ttnn.MeshDevice)
        except AttributeError:
            _is_mesh = False
        _mesh_shape = list(dev.shape) if _is_mesh else [1, 1]
        _shard = bool(_os.environ.get("TT_HW_PLANNER_SHARD_RUN")) and _is_mesh
        TP = _mesh_shape[-1] if _shard else 1
        _shard = _shard and TP > 1 and (E % TP == 0)
        self._shard = _shard
        self._TP = TP
        self._tp_axis = len(_mesh_shape) - 1
        self._mesh_shape = _mesh_shape

        if _shard:
            Eloc = E // TP

            # FOLDED expert bank: chip d's local experts concatenated along the
            # expert-output axis so the whole bank is ONE up and ONE down matmul
            # (was one matmul pair per expert -> Eloc dispatches per layer). Built on
            # host and sharded on dim 0 so chip d gets experts [d*Eloc, (d+1)*Eloc).
            def _folded(stack, cat_dim):
                chunk = torch.stack(
                    [torch.cat([stack[d * Eloc + j] for j in range(Eloc)], dim=cat_dim) for d in range(TP)], dim=0
                )
                t = ttnn.from_torch(
                    chunk.to(torch.bfloat16),
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                    mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=_mesh_shape, dims=(None, 0)),
                )
                return ttnn.reshape(t, list(chunk.shape[1:]))

            self._up_cat = _folded(up_t, 1)  # (hidden, Eloc*inter)
            self._down_cat = _folded(down_t, 0)  # (Eloc*inter, hidden)
            # decode copy of the down bank, DRAM-sharded (see _dram_mm)
            self._down_dram = _dram_mm.upload_weight(
                dev,
                torch.stack([torch.cat([down_t[d * Eloc + j] for j in range(Eloc)], dim=0) for d in range(TP)]),
                _mesh_shape,
            )
            # prefill sparse path: per-expert (Eloc, hidden, inter) up bank
            self._up_b = ttnn.from_torch(
                up_t.to(torch.bfloat16),  # (E, hidden, inter): chip d's dim-0 chunk is its local experts
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=_mesh_shape, dims=(None, 0)),
            )
            self._Eloc = Eloc
        else:
            self._up_cat = self._devw4(torch.cat(list(up_t), dim=1))
            self._down_cat = self._devw4(torch.cat(list(down_t), dim=0))
            self._down_dram = _dram_mm.upload_weight(dev, torch.cat(list(down_t), dim=0))
            self._up_b = self._devw4(up_t)
            self._Eloc = E

        # Per-chip expert selector for the DEVICE-SIDE routing path (see the
        # `routing_dense` kwarg on __call__). sel[d] is a one-hot (E, Eloc) that
        # projects a replicated full (tokens, E) routing matrix down to THIS
        # chip's contiguous expert window, sharded on dim 0 with the SAME
        # ShardTensor2dMesh call used for up_proj/down_proj so the columns line
        # up with the local expert weights. Identical to the pattern the
        # graduated `nemotron_h_mo_e` sibling already uses.
        self._sel = None
        if _shard:
            sel = torch.zeros(TP, E, self._Eloc)
            for d in range(TP):
                for j in range(self._Eloc):
                    sel[d, d * self._Eloc + j, j] = 1.0
            _sel_sh = ttnn.from_torch(
                sel,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=_mesh_shape, dims=(None, 0)),
            )
            # squeeze to 2-D: a rank-3 rhs with batch 1 against a batch-B lhs is
            # a PARTIAL batch broadcast, which hangs ttnn.matmul (measured
            # 2026-09-06). A 2-D rhs broadcasts safely over any batch.
            self._sel = ttnn.reshape(_sel_sh, [E, self._Eloc])

        # One-hot (Eloc, Eloc*inter) expander: row j is 1 over expert j's slice
        # of the folded activation (same on every chip -> replicated).
        self._expand = self._upload(
            torch.repeat_interleave(torch.eye(self._Eloc), self.intermediate_dim, dim=1).to(torch.bfloat16),
            ttnn.bfloat16,
        )

        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # LoFi is safe for the bf8_b expert MLP matmuls (weights already
        # lossy); the TP selector matmul above keeps self.ckc (HiFi4).
        self._expert_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        cg = dev.compute_with_storage_grid_size()
        self._core_grid = ttnn.CoreGrid(y=cg.y, x=cg.x)
        # routed experts per token, set by the owning block; enables the sparse
        # prefill path (0 = dense everywhere)
        self.top_k = 0
        self._arange = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    # ----------------------------- helpers ---------------------------- #
    def _is_mesh(self):
        try:
            if isinstance(self.device, ttnn.MeshDevice):
                return True
        except AttributeError:
            pass
        return hasattr(self.device, "get_device_ids") or hasattr(self.device, "get_devices")

    def _upload(self, torch_tensor, dtype, layout=ttnn.TILE_LAYOUT):
        if self._is_mesh():
            try:
                return ttnn.from_torch(
                    torch_tensor,
                    dtype=dtype,
                    layout=layout,
                    device=self.device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
            except Exception:
                pass
        return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=self.device)

    def _devw(self, torch_tensor, layout=ttnn.TILE_LAYOUT):
        return self._upload(torch_tensor.to(torch.bfloat16), ttnn.bfloat8_b, layout)

    def _devw4(self, torch_tensor, layout=ttnn.TILE_LAYOUT):
        return self._upload(torch_tensor.to(torch.bfloat16), ttnn.bfloat4_b, layout)

    def _dev(self, torch_tensor, layout=ttnn.TILE_LAYOUT):
        return self._upload(torch_tensor.float(), ttnn.float32, layout)

    def _fp32(self, t):
        if isinstance(t, ttnn.Tensor):
            if t.dtype != ttnn.float32:
                return ttnn.typecast(t, ttnn.float32)
            return t
        return self._dev(t.float())

    # ----------------------------- forward ---------------------------- #
    def __call__(self, hidden_states, top_k_index=None, top_k_weights=None, routing_dense=None, **kwargs):
        hs = self._fp32(hidden_states)
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT)
        num_tokens = list(hs.shape)[0]
        E = self.num_experts

        # DEVICE-SIDE routing path: `routing_dense` is an already-on-device
        # (tokens, E) fp32 routing matrix produced by the chained pipeline's
        # on-device router. It replaces the host torch scatter_add_ below, which
        # would otherwise put host compute in the pipeline's hot path. The
        # expert loop, the expert-parallel weight split and the all_reduce are
        # unchanged either way.
        if routing_dense is not None:
            W_dev = routing_dense
            if W_dev.layout != ttnn.TILE_LAYOUT:
                W_dev = ttnn.to_layout(W_dev, ttnn.TILE_LAYOUT)
            if W_dev.dtype != ttnn.float32:
                W_dev = ttnn.typecast(W_dev, ttnn.float32)
            # self._sel is 2-D (E, Eloc), so this stays a (tokens, Eloc) matrix.
            W_sh = ttnn.matmul(W_dev, self._sel, compute_kernel_config=self.ckc) if self._shard else W_dev
            return self._mix(hs, W_sh, num_tokens)

        # Dense (tokens, E) routing-weight matrix, built on host from the
        # (index, weight) pairs -- top_k_index/top_k_weights arrive as plain
        # torch tensors (never converted to ttnn by the harness).
        idx = top_k_index if isinstance(top_k_index, torch.Tensor) else torch.zeros(num_tokens, 1, dtype=torch.long)
        wts = top_k_weights if isinstance(top_k_weights, torch.Tensor) else torch.zeros(num_tokens, 1)
        W_full = torch.zeros(num_tokens, E, dtype=torch.float32)
        W_full.scatter_add_(1, idx.long(), wts.float())

        out = None
        Eloc = self._Eloc
        if self._shard:
            # Shard the (tokens, E) routing matrix on the SAME expert axis
            # (dim 1 here == the weights' dim 0) with the SAME ShardTensor2dMesh
            # call used for up_proj/down_proj, so chip d's local W_sh columns
            # line up with chip d's local expert weights (both are the
            # contiguous window [d*Eloc, (d+1)*Eloc)).
            W_sh = ttnn.from_torch(
                W_full,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.device, mesh_shape=self._mesh_shape, dims=(None, 1)),
            )
        else:
            W_sh = self._dev(W_full)

        return self._mix(hs, W_sh, num_tokens)

    def _capacity(self, num_tokens):
        """Tokens each local expert processes on the sparse path: 4x the mean
        load (num_tokens * top_k / num_experts), tile-aligned, capped at T."""
        mean = num_tokens * self.top_k / self.num_experts
        return min(num_tokens, _dram_mm.TILE * max(1, math.ceil(4 * mean / _dram_mm.TILE)))

    def _mix_sparse(self, hs, W_sh, num_tokens):
        """Prefill MoE computing only routed (token, expert) pairs.

        Each local expert takes its top-C tokens by routing weight (unrouted
        slots carry weight 0 and contribute nothing), gathers their rows,
        runs its own up/relu2/down, and a one-hot matmul adds every row back
        to its token. The dense form evaluates all Eloc experts on all T
        tokens; this does Eloc*C rows, C ~ 4x the mean expert load."""
        Eloc, I, H, T = self._Eloc, self.intermediate_dim, self.hidden_dim, num_tokens
        C = self._capacity(T)
        # per-expert top-C tokens: values = routing weights, indices = token ids
        vals, idx = ttnn.topk(ttnn.typecast(ttnn.transpose(W_sh, -2, -1), ttnn.bfloat16), C, dim=-1)  # (Eloc, C)
        ttnn.deallocate(W_sh)
        idx_rm = ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        idx_rm = ttnn.reshape(idx_rm, [1, Eloc * C])
        table = ttnn.to_layout(ttnn.typecast(hs, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)  # (T, H)
        xe = ttnn.embedding(idx_rm, table, layout=ttnn.TILE_LAYOUT)  # (1, Eloc*C, H)
        ttnn.deallocate(table)
        xe = ttnn.reshape(xe, [Eloc, C, H])
        act = ttnn.matmul(xe, self._up_b, compute_kernel_config=self._expert_ckc, dtype=ttnn.bfloat8_b)  # (Eloc,C,I)
        ttnn.deallocate(xe)
        w3 = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT), [Eloc, C, 1]), ttnn.TILE_LAYOUT
        )  # (Eloc, C, 1)
        act = ttnn.multiply(
            ttnn.relu(act), w3, dtype=ttnn.bfloat8_b, input_tensor_a_activations=[ttnn.UnaryOpType.SQUARE]
        )  # relu2 * routing weight
        ye = ttnn.matmul(
            act, ttnn.reshape(self._down_cat, [Eloc, I, H]), compute_kernel_config=self._expert_ckc, dtype=ttnn.bfloat16
        )  # (Eloc, C, H)
        ttnn.deallocate(act)
        # combine: out[t] = sum over (e, c) with idx[e, c] == t of ye[e, c]
        ar = self._arange.get(T)
        if ar is None:
            ar = self._arange[T] = self._upload(torch.arange(T, dtype=torch.float32).reshape(T, 1), ttnn.float32)
        idx_f = ttnn.reshape(ttnn.typecast(idx, ttnn.float32), [1, Eloc * C])
        onehot = ttnn.eq(ar, idx_f, dtype=ttnn.bfloat8_b)  # (T, Eloc*C); 0/1 is exact in bf8_b
        out = ttnn.matmul(
            onehot, ttnn.reshape(ye, [Eloc * C, H]), compute_kernel_config=self._expert_ckc, dtype=ttnn.float32
        )  # (T, H)
        ttnn.deallocate(onehot)
        ttnn.deallocate(ye)
        if self._shard:
            out = ttnn.all_reduce(out, cluster_axis=self._tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.typecast(out, ttnn.bfloat16)

    def _mix(self, hs, W_sh, num_tokens):
        """The expert-parallel mixture itself: pure ttnn, identical for the
        host-routing and device-routing entry paths above."""
        if self.top_k and num_tokens > _dram_mm.TILE:
            return self._mix_sparse(hs, W_sh, num_tokens)
        Eloc, I = self._Eloc, self.intermediate_dim
        hs_bf = ttnn.typecast(hs, ttnn.bfloat16)
        # Folded bank: one up matmul over all local experts, relu2, scale each
        # expert's slice by its routing weight, then one down matmul whose K
        # reduction performs the weighted sum over experts.
        act = ttnn.linear(
            hs_bf,
            self._up_cat,
            compute_kernel_config=self._expert_ckc,
            core_grid=self._core_grid,
            activation="relu",
            dtype=ttnn.bfloat8_b,
        )  # (T, Eloc*inter)
        ttnn.deallocate(hs_bf)
        # (T, Eloc) routing weights -> (T, Eloc*inter) via a one-hot expander
        # matmul; avoids tile-layout reshapes of the wide activation.
        w_wide = ttnn.matmul(W_sh, self._expand, compute_kernel_config=self.ckc, dtype=ttnn.bfloat8_b)
        # relu2 = square fused into the routing-weight multiply (one pass over act)
        act = ttnn.multiply(
            act, w_wide, dtype=ttnn.bfloat8_b, input_tensor_a_activations=[ttnn.UnaryOpType.SQUARE]
        )  # bf8_b halves the down matmul's activation read
        ttnn.deallocate(w_wide)
        if num_tokens <= _dram_mm.TILE:  # decode: stream the DRAM-sharded bank
            out = _dram_mm.matmul(
                self.device,
                ttnn.reshape(act, [1, num_tokens, Eloc * I]),
                self._down_dram,
                self.hidden_dim,
                self._expert_ckc,
            )
            out = ttnn.reshape(out, [num_tokens, self.hidden_dim])
        else:
            out = ttnn.matmul(
                act,
                self._down_cat,
                compute_kernel_config=self._expert_ckc,
                core_grid=self._core_grid,
                dtype=ttnn.float32,
            )  # (T, hidden)
        ttnn.deallocate(act)
        ttnn.deallocate(W_sh)

        if self._shard:
            out = ttnn.all_reduce(out, cluster_axis=self._tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.typecast(out, ttnn.bfloat16)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtNemotronHExperts.build(device, torch_module)


# Backward-compatible slug shim.
def nemotron_h_experts(device, torch_module=None):
    return TtNemotronHExperts.build(device, torch_module)
