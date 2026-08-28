# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the LM head of
`/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder` (`lm_head`).

The component is a single bias-free projection, hidden=4096 -> vocab=151936.

Tensor-parallel scheme (TP = number of mesh devices; 8 here)
------------------------------------------------------------
The head's output is the final logit vector -- it feeds per-element work
(argmax / sampling), not a reduction back to model dim -- so this is the
textbook COLUMN-parallel projection: split the OUTPUT (vocab) axis, give
each chip padded_vocab/TP columns, and `all_gather` on that axis at the end
so every chip carries the full-width logits the harness reads back.

Vocab padding is what makes the split tile-clean: 151936 / 8 = 18992
columns per chip, which is NOT a multiple of the 32-wide tile, so each
shard would carry padded tiles and fall onto the slow composite all-gather.
Rounding the vocab up to 152064 (= 8 x 19008, and 19008 = 32 x 594) with
ZERO columns splits evenly on tile boundaries; the pad columns are sliced
off after the gather, so they never reach the output. This mirrors
`models/tt_transformers/tt/lm_head.py`, which pads to `padded_vocab_size`
and splits `padded_vocab_size // num_devices` columns per device.

Nothing is reduced, so there is no all_reduce here -- an all_gather on the
sharded axis is the correct collective for a column-parallel tail.

Placement changes; the math does not.
"""
from __future__ import annotations

import torch

import ttnn

TILE = 32


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


class TtDecoderHead:
    """Native ttnn LM head, column-parallel over the mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)
        self.tp = _num_devices(device) if self.mesh else 1

        sd = torch_module.state_dict()
        w = sd.get("weight")
        if w is None:  # a wrapper around the projection rather than the projection itself
            for k, v in sd.items():
                if k.endswith("weight") and v.ndim == 2:
                    w = v
                    break
        if w is None:
            raise RuntimeError("decoder_head stub: no 2-D projection weight in the reference state_dict")
        bias = sd.get("bias")

        # nn.Linear stores [out, in]; ttnn.linear wants [in, out].
        w = w.detach().to(torch.bfloat16).t().contiguous()
        self.in_features, self.vocab_size = int(w.shape[0]), int(w.shape[1])

        # Round the vocab up so every chip's column slice is a whole number of
        # tiles; the extra columns are zeros and get sliced off after the gather.
        block = TILE * self.tp
        self.padded_vocab = ((self.vocab_size + block - 1) // block) * block
        if self.padded_vocab != self.vocab_size:
            w = torch.cat(
                [w, torch.zeros(self.in_features, self.padded_vocab - self.vocab_size, dtype=w.dtype)], dim=-1
            )

        self.weight = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if (self.mesh and self.tp > 1) else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        del w

        self.bias = None
        if bias is not None:
            b = bias.detach().to(torch.bfloat16).reshape(1, -1)
            if self.padded_vocab != self.vocab_size:
                b = torch.cat([b, torch.zeros(1, self.padded_vocab - self.vocab_size, dtype=b.dtype)], dim=-1)
            self.bias = ttnn.from_torch(
                b,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1) if (self.mesh and self.tp > 1) else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, *args, **kwargs):
        x = kwargs.pop("input", None)
        if x is None:
            x = kwargs.pop("hidden_states", None)
        if x is None:
            for a in args:
                if a is not None:
                    x = a
                    break
        if x is None:
            raise ValueError("decoder_head stub: no input tensor supplied")

        if torch.is_tensor(x):
            x = ttnn.from_torch(
                x.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        shape = list(x.shape)
        out = ttnn.linear(x, self.weight, bias=self.bias, compute_kernel_config=self.compute_kernel_config)
        if self.tp > 1:
            # Column-parallel tail: gather the vocab shards so every chip holds
            # the full-width logits.
            out = ttnn.all_gather(out, len(shape) - 1)
        if self.padded_vocab != self.vocab_size:
            out = ttnn.slice(out, [0] * len(shape), [int(d) for d in shape[:-1]] + [self.vocab_size])
        return out

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("decoder_head stub needs the torch reference module to source its weights")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)


def decoder_head(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)
