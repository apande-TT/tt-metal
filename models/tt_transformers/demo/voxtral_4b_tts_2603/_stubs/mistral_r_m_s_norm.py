# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mistral_r_m_s_norm` (`model.layers.0.input_layernorm`).

`x * rsqrt(mean(x^2) + eps) * weight`, over dim 3072 with `eps = 1e-5`. `MistralRMSNorm` does the
reduction in float32 and casts back, so the activation is widened here to match rather than
normalising in bfloat16.

SPELLED OUT IN FOUR OPS, NOT `ttnn.rms_norm`. Measured against the reference on the real layer-0
input, the stock op lands at 9.65e-4 relative error and the four ops below at 6.6e-8 -- four
orders of magnitude apart. It matters because the stack runs 52 of these and the error is
RELATIVE, so it rescales the whole branch output that follows: at 9.65e-4 apiece the final hidden
state came back at PCC 0.9971, and a 21-level acoustic quantiser (code edges 0.1 apart in x, with
a fifth of all values landing within 0.01 of one) turned that into ~26% wrong audio codes. Every
other term is already smaller -- the float32-activation x bfloat16-weight matmul sits at a 4.9e-4
hardware floor per linear, and the bfloat16 Q/K/V cast SDPA forces is 5.7e-4 end to end.
`mistral_model.py::_rms_norm` is the same four ops, for the same reason.

Tile padding on an off-tile sequence is safe: a padded row is all zeros, so `mean(x^2)` is 0 and
`0 * rsqrt(eps)` stays 0 -- no NaN, and nothing leaks into a real row.

Gamma therefore goes up as `[1, 1, 1, dim]` float32 TILE, the form the final multiply broadcasts
against, rather than the `[1, 1, dim // 32, 32]` ROW_MAJOR that only existed to satisfy
`ttnn.rms_norm`'s `gamma.padded_shape[-1] == TILE_WIDTH` assert
(`layernorm_device_operation.cpp:106`).

THE LEADING BOUND IS READ OFF THE TENSOR. This was `ttnn.reshape(hidden_states, [1, 1, seq, dim])`
-- correct at the batch of 1 the per-component harness feeds, and wrong for every batched caller:
at B=32 that reshape either raises on volume or, once a leading 1 is folded in elsewhere, keeps
row 0 and silently drops samples 1..31. An RMS norm reduces over the LAST dim only, so collapsing
every leading axis into one bound is exact for `[B, S, D]`, `[B, 1, S, D]` and the decode stream's
`[1, 1, B, D]` alike, and the output goes back in the rank it arrived in.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import cpp_sqmean


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if layout == ttnn.TILE_LAYOUT and t.numel() <= 262144:
        # A small tensor (a gamma, a bias, a layer scale) tilizes on the HOST: tilized on the device it is
        # a whole single-core op (~80 us for a 1 x 1024 float32 row) at load time.
        mapper = ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None
        return ttnn.to_device(ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper), device)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


_STATS_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
)


def _sq_mean(x):
    """`mean(x^2, -1)` of a float32 `x`, as a square then a row-mean reduce at every height.

    A tall `x` used to take `rms_norm_pre_all_gather` (one read of `x`, float32 row sums); measured
    on the 640-row tail it ran ~71 us a norm (65 us on its 20 cores, one per tile row, plus the
    slice and scale) where the square into L1 and the one-core-per-tile-row mean take ~45 us."""
    shape = [int(d) for d in x.shape]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    # x^2 and its row means live in L1 -- the mean is a one-core-per-tile-row reduce, which reads x^2
    # back faster from L1 -- and x^2 is freed before any other op runs (~72 KB a core at 640 rows).
    l1 = ttnn.L1_MEMORY_CONFIG
    # A one-tile-row (decode-step) mean takes the FPU reduce path (x^2 truncated to TF32 on the way in)
    # instead of the accurate SFPU one; the prefill's taller norms keep the accurate path.
    if rows > 32 and cpp_sqmean.supports(x):
        # cpp: the square and the row mean as ONE generic_op on two cores a tile row (tt/cpp_sqmean) -- the same
        # float32 SFPU steps in the same order, so every mean is the stock one, bit for bit.
        return cpp_sqmean.sq_mean(x, memory_config=l1)
    return ttnn.mean(
        ttnn.square(x, memory_config=l1),
        dim=-1,
        keepdim=True,
        memory_config=l1,
        fast_and_approximate_mode=rows <= 32,
    )


def build(device, torch_module):
    norm = torch_module
    dim = int(norm.weight.shape[-1])
    eps = float(norm.variance_epsilon)
    gamma = _from_torch(norm.weight.detach().reshape(1, 1, 1, dim), device, dtype=ttnn.float32)
    # A unit gamma (one the caller folded into the consuming weights) costs a full pass for nothing.
    unit = bool(torch.all(norm.weight.detach() == 1))

    def mistral_r_m_s_norm(hidden_states, dtype=None, memory_config=None, **kwargs):
        """`dtype` narrows only the OUTPUT (e.g. bf16 for a norm that feeds a matmul); the
        statistics and the scaling are float32 either way. `memory_config` places the output (L1
        for a norm whose only reader is the next matmul)."""
        shape = [int(s) for s in hidden_states.shape]
        seq = shape[-2]
        lead = 1
        for size in shape[:-2]:
            lead *= size
        x = ttnn.reshape(hidden_states, [lead, 1, seq, dim])
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        scale = ttnn.add(_sq_mean(x), eps, activations=[ttnn.UnaryOpType.RSQRT])
        if unit:
            out = ttnn.multiply(x, scale, dtype=dtype or ttnn.float32, memory_config=memory_config)
        else:
            out = ttnn.multiply(
                ttnn.multiply(x, scale), gamma, dtype=dtype or ttnn.float32, memory_config=memory_config
            )
        return ttnn.reshape(out, [lead, seq, dim] if len(shape) == 3 else [lead, 1, seq, dim])

    return mistral_r_m_s_norm
