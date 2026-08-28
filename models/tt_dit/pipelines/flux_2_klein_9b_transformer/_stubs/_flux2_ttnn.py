# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared weight-staging / mesh helpers for the FLUX.2-klein-9B transformer ports.

Only setup and marshalling live here — every port keeps its own forward maths
in its own stub, so each stub reads as a complete description of its module.

Two build-time facts drive most of this file:

* **Packed weights.** Flux2 fuses projections into single matrices
  (`to_qkv_mlp_proj` = q|k|v|gate|up, `ff.linear_in` = gate|up,
  `norm_out.linear` = scale|shift).  Column-sharding a packed matrix directly
  hands one chip the tail of one block and the head of the next, so the blocks
  are re-interleaved on the HOST first — see `pack_col_blocks`.
* **Replicated activations.** The PCC harness replicates the input across the
  mesh, so a row-parallel matmul takes its own slice of that replicated
  activation with `ttnn.mesh_partition` (a local per-device slice, no fabric
  traffic) and finishes with `ttnn.all_reduce`.
"""

from __future__ import annotations

import math

import torch

import ttnn


def compute_config():
    """HiFi3 + fp32 accumulate.

    Every port here is dominated by 4096-wide matmuls feeding a normalization
    or a softmax, where LoFi rounding shows up undamped in the output -- so the
    fidelity has to be high.  It is HiFi3 rather than HiFi4 because ttnn itself
    says so on every op that pairs the two ("On Wormhole with fp32
    accumulation, output accuracy can be worse with HiFi4 than HiFi3 due to a
    hardware bug"), and the 32-block PCC ladder confirms it: at B=4 the stage's
    per-sample minimum against the bf16 golden is

        HiFi4  0.98418     HiFi3  0.98744     HiFi2  0.98223

    so HiFi3 is both the most accurate of the three here AND one mantissa pass
    cheaper than the HiFi4 it replaces.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def mesh_width(device):
    """Number of chips this port is spread over (1 for a plain device)."""
    for attr in ("get_num_devices", "get_device_ids"):
        fn = getattr(device, attr, None)
        if fn is None:
            continue
        try:
            value = fn()
        except Exception:  # noqa: BLE001
            continue
        n = value if isinstance(value, int) else len(value)
        if n:
            return int(n)
    return 1


def as_device(x, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """`x` as a device tensor, whichever side of the harness it came from.

    The single-device PCC test stages non-primary kwargs itself (it has to: the
    native probe counts a `ttnn.from_torch` inside the forward as torch
    compute).  The generated sharded test hands them over as raw host tensors.
    Ports accept either.
    """
    if x is None or isinstance(x, ttnn.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return type(x)(as_device(e, device, dtype=dtype, layout=layout) for e in x)
    if isinstance(x, torch.Tensor):
        t = x.to(torch.bfloat16) if dtype == ttnn.bfloat16 else x
        if mesh_width(device) > 1:
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=layout,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)
    return x


def stage(t, device, *, shard_dim=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """A host tensor on the mesh, optionally sharded along `shard_dim`."""
    t = t.detach().float().contiguous()
    if shard_dim is None:
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=shard_dim),
    )


def matmul_weight(linear, device, *, tp=1, mode="replicate"):
    """An `nn.Linear`'s weight as a `(in, out)` matmul operand on the mesh.

    ``mode`` is ``"col"`` (split OUTPUT features; follow with `all_gather` or
    keep the result local), ``"row"`` (split INPUT features; follow with
    `all_reduce`), or ``"replicate"``.  Falls back to replicated whenever the
    split axis is not divisible by ``tp``.
    """
    w = linear.weight.detach().float().t().contiguous()  # (in, out)
    if tp > 1 and mode == "col" and w.shape[1] % tp == 0:
        return stage(w, device, shard_dim=1)
    if tp > 1 and mode == "row" and w.shape[0] % tp == 0:
        return stage(w, device, shard_dim=0)
    return stage(w, device)


def bias_vector(linear, device):
    b = getattr(linear, "bias", None)
    if b is None:
        return None
    return stage(b.reshape(1, -1), device)


def pack_col_blocks(blocks, tp):
    """Re-interleave packed COLUMN blocks so an even `tp`-way column shard is
    `[b0_i | b1_i | ...]` on chip *i*.

    `blocks` are `(in, out_k)` matmul operands.  A plain `cat` + even split
    would give chip *i* a contiguous window of the concatenation, which crosses
    block boundaries — the classic packed-sharding bug.
    """
    if tp <= 1:
        return torch.cat(list(blocks), dim=1)
    out = []
    for i in range(tp):
        for w in blocks:
            n = w.shape[1] // tp
            out.append(w[:, i * n : (i + 1) * n])
    return torch.cat(out, dim=1)


def pack_row_blocks(blocks, tp):
    """The row-parallel mirror of `pack_col_blocks`.

    `blocks` are `(in_k, out)` matmul operands whose inputs are the
    concatenation of several column-parallel results.  Chip *i* holds
    `[b0_i ; b1_i ; ...]`, matching the `[b0_i | b1_i | ...]` activation it
    already has locally.
    """
    if tp <= 1:
        return torch.cat(list(blocks), dim=0)
    out = []
    for i in range(tp):
        for w in blocks:
            n = w.shape[0] // tp
            out.append(w[i * n : (i + 1) * n, :])
    return torch.cat(out, dim=0)


def rotate_matrix(head_dim, device):
    """The `(head_dim, head_dim)` matrix implementing diffusers' interleaved
    RoPE rotation `(x0, x1) -> (-x1, x0)`.

    `ttnn.reshape` cannot split a tiled last dimension in this build, so the
    pair-swap is done as a matmul by a fixed 0/±1 block-diagonal matrix — exact
    in bfloat16, and negligible next to the 4096-wide projections.
    """
    r = torch.zeros(head_dim, head_dim)
    even = torch.arange(0, head_dim, 2)
    r[even, even + 1] = 1.0
    r[even + 1, even] = -1.0
    return stage(r, device)


def rope_pair(image_rotary_emb, device):
    """`(cos, sin)` broadcastable over `(1, heads, seq, head_dim)`."""
    if image_rotary_emb is None:
        return None
    cos, sin = image_rotary_emb
    cos = as_device(cos, device)
    sin = as_device(sin, device)
    while len(cos.shape) < 4:
        cos = ttnn.unsqueeze(cos, 0)
        sin = ttnn.unsqueeze(sin, 0)
    return cos, sin


def apply_rope(x, rope, rot):
    """diffusers `apply_rotary_emb(..., sequence_dim=1)` on a `(1, H, S, D)` tensor."""
    if rope is None:
        return x
    cos, sin = rope
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(ttnn.matmul(x, rot), sin))


def timestep_freqs(time_proj, device):
    """The `(1, num_channels // 2)` frequency row of a diffusers `Timesteps`, in FLOAT32.

    `get_timestep_embedding` multiplies the (scalar) timestep by this row and
    takes sin/cos of the product, so `scale` folds into the row:
    `scale * (t * f) == t * (scale * f)`.

    fp32 is not optional here. The product is an absolute angle that reaches
    the timestep's own magnitude (~1e3 on the pipeline's `timestep * 1000`
    scale), where one bfloat16 ulp is several radians.
    """
    half = int(time_proj.num_channels) // 2
    shift = float(time_proj.downscale_freq_shift)
    scale = float(getattr(time_proj, "scale", 1))
    exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / (half - shift)
    row = (scale * torch.exp(exponent)).reshape(1, -1)
    return stage(row, device, dtype=ttnn.float32)


def timestep_column(t):
    """A 1-D timestep tensor as an `(N, 1)` column that broadcasts against a
    frequency row.

    `unsqueeze(0)` is a free view; `unsqueeze(-1)` would go through the tiled
    reshape kernel, which this build cannot compile — so the N > 1 case
    transposes the `(1, N)` view instead.
    """
    if len(t.shape) == 1:
        t = ttnn.unsqueeze(t, 0)
        if t.shape[-1] > 1:
            t = ttnn.transpose(t, -1, -2)
    return t


def mod_chunks(mod, count):
    """`Flux2Modulation.split`: a packed `(B, count * dim)` vector as `count`
    `(1, B, dim)` tensors that broadcast over the token axis."""
    if len(mod.shape) == 2:
        mod = ttnn.unsqueeze(mod, 0)
    width = mod.shape[-1] // count
    rows = mod.shape[-2]
    return [ttnn.slice(mod, [0, 0, i * width], [1, rows, (i + 1) * width]) for i in range(count)]
