# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flux2_feed_forward` of FLUX.2-klein-9B.

HF reference: `Flux2FeedForward`, reached as
`Flux2Transformer2DModel.transformer_blocks[0].ff` (dim 4096, mult 3.0 →
inner 12288, no bias)::

    x = linear_in(x)          # 4096 -> 2 * 12288, packed [gate | up]
    x = Flux2SwiGLU(x)        # silu(gate) * up      -> 12288
    x = linear_out(x)         # 12288 -> 4096

The SwiGLU projection is fused into `linear_in`, so `act_fn` holds no weights.

Tensor-parallel scheme (TP=8)
-----------------------------
Column-then-row with a single collective:

* `linear_in` is COLUMN-parallel PER PACKED BLOCK. An even 8-way split of the
  24576-wide packed matrix would hand one chip the tail of `gate` and the head
  of `up`; `_flux2_ttnn.pack_col_blocks` re-interleaves the two halves on the
  host first, so chip *i* holds `[gate_i | up_i]` (1536 + 1536) and its local
  SwiGLU is the true SwiGLU of inner features `[1536i, 1536i + 1536)`.
* `linear_out` is ROW-parallel over exactly those inner features, so each chip
  matmuls its own rows and one `all_reduce` sums the partials — a matmul over a
  concatenated contraction axis IS the sum of the per-block matmuls.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["transformer_blocks[0].ff"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2FeedForward:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()

        wi = torch_module.linear_in.weight.detach().float().t().contiguous()  # (4096, 2 * inner)
        inner = wi.shape[1] // 2

        tp = H.mesh_width(device)
        # Both packed halves must split evenly, and the row-parallel partner
        # splits on that same inner axis.
        if tp > 1 and inner % tp:
            tp = 1
        self.tp = tp

        self.w_in = H.stage(
            H.pack_col_blocks([wi[:, :inner], wi[:, inner:]], tp),
            device,
            shard_dim=1 if tp > 1 else None,
        )
        self.b_in = H.bias_vector(torch_module.linear_in, device)
        self.w_out = H.matmul_weight(torch_module.linear_out, device, tp=tp, mode="row")
        self.b_out = H.bias_vector(torch_module.linear_out, device)

    def __call__(self, x, **kwargs):
        x = H.as_device(x, self.device)

        packed = ttnn.linear(x, self.w_in, bias=self.b_in, compute_kernel_config=self.cfg)
        rows = packed.shape[-2]
        half = packed.shape[-1] // 2
        gate = ttnn.slice(packed, [0, 0, 0], [packed.shape[0], rows, half])
        up = ttnn.slice(packed, [0, 0, half], [packed.shape[0], rows, packed.shape[-1]])

        hidden = ttnn.multiply(ttnn.silu(gate), up)

        out = ttnn.linear(hidden, self.w_out, compute_kernel_config=self.cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out)
        # A row-parallel bias is added ONCE, after the reduction.
        if self.b_out is not None:
            out = ttnn.add(out, self.b_out)
        return out


def build(device, torch_module):
    return TtFlux2FeedForward(device, torch_module)


def flux2_feed_forward(device, torch_module, x, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(x, **kwargs)
