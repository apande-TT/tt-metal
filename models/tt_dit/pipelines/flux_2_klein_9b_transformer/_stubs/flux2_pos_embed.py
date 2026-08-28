# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flux2_pos_embed` of FLUX.2-klein-9B.

HF reference: `Flux2PosEmbed`, reached as `Flux2Transformer2DModel.pos_embed`
(theta 2000, `axes_dim = [32, 32, 32, 32]`). It has no parameters: for each of
the four rope axes it takes that column of the `(S, 4)` id table, forms the
outer product with `1 / theta**(2j/32)`, and emits cos/sin repeat-interleaved
to the axis width; the four axes are concatenated into `(S, 128)`.

    cos_i, sin_i = get_1d_rotary_pos_embed(32, ids[:, i], theta, repeat_interleave_real=True)
    return cat(cos_i, dim=-1), cat(sin_i, dim=-1)

Implementation
--------------
The per-axis outer product, the repeat-interleave and the concatenation are all
linear in `ids`, so they collapse into a SINGLE constant `(4, 128)` frequency
matrix built once at load time: column `32i + 2j` and column `32i + 2j + 1`
both hold axis *i*'s frequency `j`, every other entry is zero. The forward is
then one matmul and two SFPU calls — no per-axis loop, no `repeat_interleave`,
no `concat`, and (usefully in this build) no `ttnn.reshape`.

The matmul runs in FLOAT32. The rope argument is an absolute angle, so its
precision has to be absolute too: at a position value of 7 a bfloat16 argument
carries a ~0.06 rad error, which lands directly on cos/sin. In fp32 the SFPU
trig is exact against the torch reference over this range.

This module is a rotary table with no matmul weights, so it is replicate-only
under tensor parallelism — every consumer needs all 128 columns.
"""

from __future__ import annotations

import torch

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["pos_embed"]


class TtFlux2PosEmbed:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        theta = float(torch_module.theta)
        axes_dim = list(torch_module.axes_dim)
        total = sum(axes_dim)

        # float64 to match the reference's `freqs_dtype`; the table is tiny and
        # is rounded to fp32 exactly once, here.
        freq = torch.zeros(len(axes_dim), total, dtype=torch.float64)
        offset = 0
        for axis, dim in enumerate(axes_dim):
            f = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            # Interleaved layout: `repeat_interleave(2)` of the reference's
            # per-axis frequency vector, folded into the matrix.
            freq[axis, offset : offset + dim : 2] = f
            freq[axis, offset + 1 : offset + dim : 2] = f
            offset += dim

        self.freq = ttnn.from_torch(
            freq.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, ids, **kwargs):
        angle = ttnn.matmul(ttnn.typecast(ids, ttnn.float32), self.freq, compute_kernel_config=self.cfg)
        return ttnn.cos(angle), ttnn.sin(angle)


def build(device, torch_module):
    return TtFlux2PosEmbed(device, torch_module)


def flux2_pos_embed(device, torch_module, ids, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(ids, **kwargs)
