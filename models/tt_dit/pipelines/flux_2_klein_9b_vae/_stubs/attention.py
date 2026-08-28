# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE tensor-parallel ttnn port of `attention` for
`/tmp/tt_hw_planner_components/flux_2_klein_9b_vae`.

The component resolves to `encoder.mid_block.attentions.0` of
`diffusers.AutoencoderKLFlux2` — a *spatial* self-attention, not a transformer
attention:

    Attention(query_dim=512, heads=1, group_norm=GroupNorm(32, 512, eps=1e-6),
              residual_connection=True, rescale_output_factor=1,
              processor=AttnProcessor2_0)

so `heads == 1` and `head_dim == 512`: the whole channel axis is one head.
`AttnProcessor2_0` computes, for a rank-3 `(B, L, C)` input,

    h = group_norm(x^T)^T ; q,k,v = to_q/k/v(h)
    y = to_out[0](sdpa(q, k, v)) + x        (rescale_output_factor == 1)

which is exactly `models/tt_dit/models/vae/vae.py::VaeAttention` — the in-tree
VAE block that `Flux2VaeDecoder` already composes. This stub builds that block
directly (the scaffold's seed pointed at `models/tt_transformers/tt/attention.py`,
a GQA/RoPE decoder attention with none of this structure).

TP=8 scheme (channel-parallel; there is only one head, so heads cannot split):
  * `group_norm` REPLICATED per shard and needs NO collective — 32 groups over 8
    devices leaves 4 whole groups (64 channels) per device, so each device owns
    complete groups and its statistics are exact.
  * fused `to_qkv` is COLUMN-parallel with per-device `Q|K|V` interleaving
    (`VaeAttention._merge_qkv`), so one even shard of the output dim hands chip
    *i* a contiguous Q, K and V chunk. Each is then all-gathered back to the full
    512 channels because single-head SDPA needs the whole head_dim for the
    QK^T dot product.
  * `to_out` is COLUMN-parallel, so the block's output stays channel-fractured
    and lines up with the channel-fractured residual — activations stay
    fractured end to end, which is what lets the mid-block/encoder composites
    chain these blocks without a collective per block.
Norms, the group-norm gamma/beta and the input mask stay replicated per the
TP principles.

Standalone (per-component PCC) contract: the harness replicates the input over
the mesh and reads back one device's copy, so `__call__` fractures the channel
axis on entry with `ttnn.mesh_partition` (a local slice, no fabric) and
all-gathers it back on exit. Composites should call `forward()` instead and keep
the activation fractured.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_dit.models.vae.vae import VaeAttention, VaeContext, VaeNormDescGroup
from models.tt_dit.parallel.manager import CCLManager


def _resolve_tp_axis(device) -> int | None:
    """The mesh axis that actually holds more than one device.

    The shard harness opens `MeshShape(1, TP)`, so the TP axis is 1; on a 1x1
    mesh there is no axis to shard and the block runs replicated.
    """
    shape = tuple(device.shape)
    for axis in range(len(shape) - 1, -1, -1):
        if int(shape[axis]) > 1:
            return axis
    return None


class TtAttention:
    """`VaeAttention` wired for this checkpoint's mid-block attention."""

    def __init__(self, inner, ctx, num_channels: int) -> None:
        self.inner = inner
        self.ctx = ctx
        self.num_channels = int(num_channels)

    @classmethod
    def build(cls, device, torch_module=None):
        if torch_module is None:
            msg = "attention stub needs the torch module to source its weights"
            raise RuntimeError(msg)

        group_norm = torch_module.group_norm
        num_channels = int(group_norm.num_channels)

        tp_axis = _resolve_tp_axis(device)
        ccl_manager = CCLManager(device, num_links=1, topology=ttnn.Topology.Linear) if tp_axis is not None else None
        ctx = VaeContext(device=device, tp_axis=tp_axis, ccl_manager=ccl_manager)

        inner = VaeAttention(
            num_channels=num_channels,
            norm=VaeNormDescGroup(num_groups=int(group_norm.num_groups), eps=float(group_norm.eps)),
            ctx=ctx,
        )
        # Weight staging only: `_prepare_torch_state` renames `to_out.0`->`to_out`
        # and `group_norm`->`norm`, and fuses to_q/to_k/to_v into the interleaved
        # `to_qkv`; `Parameter.load_torch_tensor` shards each one onto the mesh.
        state = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}
        inner.load_torch_state_dict(state)

        return cls(inner, ctx, num_channels)

    def forward(self, x):
        """`x`: `[N, H, W, C/tp]` channel-fractured NHWC (tt_dit VAE convention)."""
        return self.inner.forward(x)

    def __call__(self, x, *args, **kwargs):
        """Per-component entry point: replicated in, replicated full-width out.

        Accepts the flattened-spatial rank-3 `(B, L, C)` the PCC harness builds,
        NHWC `(N, H, W, C)`, or NCHW `(N, C, H, W)`.
        """
        c = self.num_channels
        rank = len(x.shape)
        is_nchw = rank == 4 and int(x.shape[1]) == c and int(x.shape[-1]) != c
        if is_nchw:
            x = ttnn.permute(x, (0, 2, 3, 1))
        elif rank == 3:
            # (1, L, C) -> (N=1, H=1, W=L, C): a leading-1 insert, so every
            # reshape inside VaeAttention degenerates to a no-op view.
            x = ttnn.unsqueeze(x, 1)

        tp_axis = self.ctx.tp_axis
        replicated = int(x.shape[-1]) == c and tp_axis is not None
        if replicated:
            x = ttnn.mesh_partition(x, len(x.shape) - 1, tp_axis)

        y = self.inner.forward(x)

        if replicated:
            y = self.ctx.ccl_manager.all_gather(y, dim=len(y.shape) - 1, mesh_axis=tp_axis, use_hyperparams=True)

        if is_nchw:
            y = ttnn.permute(y, (0, 3, 1, 2))
        elif rank == 3:
            y = ttnn.squeeze(y, 1)
        return y


def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)
