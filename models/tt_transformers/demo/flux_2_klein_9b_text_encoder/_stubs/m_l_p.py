# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the Qwen3 SwiGLU MLP.

Component `m_l_p` of `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
(`model.layers.0.mlp`, `Qwen3MLP`):

    y = down_proj( silu(gate_proj(x)) * up_proj(x) )

Shapes: hidden=4096, intermediate=12288, no biases.

Tensor-parallel scheme (TP = number of mesh devices; 8 here)
------------------------------------------------------------
The textbook column-then-row pair. gate_proj and up_proj are
COLUMN-parallel: their outputs meet only in an ELEMENTWISE product through
the SiLU gate, so a chip that owns intermediate columns [1536i, 1536i+1536)
of both can compute that product for its own columns with no communication
(12288/8 = 1536 = 48 tiles, so the split is tile-clean). down_proj is
ROW-parallel over that same intermediate axis: each chip multiplies its
1536 rows and produces a PARTIAL sum over the full hidden dim, so one
`ttnn.all_reduce` adds the partials and leaves the full-width result
replicated on every chip -- which is what the harness reads back.

Nothing here is a norm, lookup or bias, so nothing stays replicated except
the activations entering gate_proj/up_proj.

Placement changes; the math does not.
"""
from __future__ import annotations

import torch

import ttnn

TILE = 32

_ACT = {
    "silu": ttnn.silu,
    "swish": ttnn.silu,
    "gelu": ttnn.gelu,
    "relu": ttnn.relu,
}


def _num_devices(device) -> int:
    fn = getattr(device, "get_num_devices", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    ids = getattr(device, "get_device_ids", None)
    if callable(ids):
        try:
            return max(1, len(ids()))
        except Exception:
            pass
    return 1


def _is_mesh(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


class TtMLP:
    """Native ttnn SwiGLU MLP, column-then-row parallel over the mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)
        self.tp = _num_devices(device) if self.mesh else 1

        sd = torch_module.state_dict()
        wg = sd["gate_proj.weight"]
        wu = sd["up_proj.weight"]
        wd = sd["down_proj.weight"]
        self.intermediate = int(wg.shape[0])
        self.hidden = int(wg.shape[1])

        if self.intermediate % (self.tp * TILE):
            # The intermediate axis must cut on tile boundaries on every chip.
            self.tp = 1

        cfg = getattr(torch_module, "config", None)
        self.act = _ACT.get(str(getattr(cfg, "hidden_act", "silu") or "silu").lower(), ttnn.silu)

        # nn.Linear stores [out, in]; ttnn.linear wants [in, out].
        self.w_gate = self._shard(wg, -1)  # column-parallel
        self.w_up = self._shard(wu, -1)  # column-parallel
        self.w_down = self._shard(wd, 0)  # row-parallel

        self.b_gate = self._shard(sd["gate_proj.bias"], -1, vector=True) if "gate_proj.bias" in sd else None
        self.b_up = self._shard(sd["up_proj.bias"], -1, vector=True) if "up_proj.bias" in sd else None
        # down_proj's bias is added AFTER the reduction, so it must not be
        # sharded (and must be added once, not once per chip).
        self.b_down = self._replicate(sd["down_proj.bias"]) if "down_proj.bias" in sd else None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _shard(self, t, dim, vector=False):
        t = t.detach().to(torch.bfloat16)
        t = t.reshape(1, -1) if vector else t.t().contiguous()
        mapper = ttnn.ShardTensorToMesh(self.device, dim=dim) if (self.mesh and self.tp > 1) else None
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _replicate(self, t, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.detach().reshape(1, -1).to(torch.bfloat16) if t.ndim == 1 else t.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, *args, **kwargs):
        x = kwargs.pop("x", None)
        for name in ("hidden_states", "input", "inputs"):
            if x is None:
                x = kwargs.pop(name, None)
        if x is None:
            for a in args:
                if a is not None:
                    x = a
                    break
        if x is None:
            raise ValueError("m_l_p stub: no input tensor supplied")
        if torch.is_tensor(x):
            x = self._replicate(x.to(torch.bfloat16))

        gate = ttnn.linear(x, self.w_gate, bias=self.b_gate, compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(x, self.w_up, bias=self.b_up, compute_kernel_config=self.compute_kernel_config)
        h = ttnn.mul(self.act(gate), up)
        out = ttnn.linear(h, self.w_down, compute_kernel_config=self.compute_kernel_config)
        if self.tp > 1:
            out = ttnn.all_reduce(out)
        if self.b_down is not None:
            out = ttnn.add(out, self.b_down)
        return out

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("m_l_p stub needs the torch reference module to source its weights")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtMLP.build(device, torch_module)


def m_l_p(device, torch_module=None):
    return TtMLP.build(device, torch_module)
