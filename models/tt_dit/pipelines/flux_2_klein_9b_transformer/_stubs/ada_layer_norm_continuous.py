# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_layer_norm_continuous` of FLUX.2-klein-9B.

HF reference: `diffusers.models.normalization.AdaLayerNormContinuous`, reached
as `Flux2Transformer2DModel.norm_out` (embedding_dim = conditioning_dim = 4096,
`elementwise_affine=False`, `bias=False`, eps 1e-6)::

    emb          = linear(silu(conditioning_embedding))   # (B, 2 * dim)
    scale, shift = chunk(emb, 2, dim=1)
    x            = norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

The affine-free LayerNorm followed by an outer scale/shift is exactly
`ttnn.layer_norm(x, weight=1 + scale, bias=shift)`, so the whole module is one
SiLU, two matmuls and one fused normalization.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["norm_out"]


def _compute_config():
    """HiFi4 + fp32 accumulate: the modulation vector multiplies every channel
    of the normalized activation, so matmul error here shows up undamped in the
    output."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _weight(t, device):
    """A torch `nn.Linear.weight` (out, in) staged as a TILE matmul operand (in, out)."""
    return ttnn.from_torch(
        t.detach().float().t().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _bias(t, device):
    if t is None:
        return None
    return ttnn.from_torch(
        t.detach().float().reshape(1, -1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


class TtAdaLayerNormContinuous:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = _compute_config()
        self.eps = float(getattr(torch_module.norm, "eps", 1e-6))

        w = torch_module.linear.weight
        dim = w.shape[0] // 2
        # `chunk(emb, 2, dim=1)` yields scale first, then shift. Split the
        # packed projection on the HOST: slicing the packed result on device
        # would cost an extra op every call, and under tensor parallelism a
        # column-shard of the packed matrix hands one chip half of each block.
        self.w_scale = _weight(w[:dim], device)
        self.w_shift = _weight(w[dim:], device)

        b = torch_module.linear.bias
        self.b_scale = _bias(b[:dim] if b is not None else None, device)
        self.b_shift = _bias(b[dim:] if b is not None else None, device)

    def __call__(self, x, conditioning_embedding=None, **kwargs):
        cond = ttnn.silu(conditioning_embedding)

        scale = ttnn.linear(cond, self.w_scale, bias=self.b_scale, compute_kernel_config=self.cfg)
        shift = ttnn.linear(cond, self.w_shift, bias=self.b_shift, compute_kernel_config=self.cfg)

        # LayerNorm's gamma/beta broadcast over the token axis, which is what
        # the reference's `[:, None, :]` does.
        gamma = ttnn.reshape(ttnn.add(scale, 1.0), (1, 1, scale.shape[-1]))
        beta = ttnn.reshape(shift, (1, 1, shift.shape[-1]))

        return ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=gamma,
            bias=beta,
            compute_kernel_config=self.cfg,
        )


def build(device, torch_module):
    return TtAdaLayerNormContinuous(device, torch_module)


def ada_layer_norm_continuous(device, torch_module, x, conditioning_embedding=None, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(x, conditioning_embedding=conditioning_embedding, **kwargs)
