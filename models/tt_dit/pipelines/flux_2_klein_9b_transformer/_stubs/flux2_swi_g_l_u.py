# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flux2_swi_g_l_u` of FLUX.2-klein-9B.

HF reference: `Flux2SwiGLU`, reached as
`Flux2Transformer2DModel.transformer_blocks[0].ff.act_fn`::

    half = x.shape[-1] // 2
    x    = silu(x[..., :half]) * x[..., half:]

It has no trainable parameters — Flux2 fuses the SwiGLU gate projection into
the preceding `linear_in`, so this module only consumes the packed
`[gate | up]` activation.

Tensor-parallel placement (TP=8)
--------------------------------
There is no weight to shard, so the parallelism here is over the ACTIVATION's
feature axis: the op is elementwise in the inner dimension, and chip *i* owns
inner features `[1536i, 1536i + 1536)`.

The two halves are partitioned SEPARATELY. Partitioning the packed 24576-wide
tensor directly would give chip *i* a contiguous 3072-wide window that straddles
the `gate`/`up` boundary and pairs the wrong channels; slicing `gate` and `up`
apart first and partitioning each keeps chip *i*'s pair aligned.

The closing `all_gather` exists only because this component is being validated
standalone against a full-width golden. In the real model the consumer is
`ff.linear_out`, which is row-parallel over exactly these inner features, so
the sharded result feeds it directly and the gather is absent — see the
`flux2_feed_forward` port.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["transformer_blocks[0].ff.act_fn"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2SwiGLU:
    def __init__(self, device, torch_module):
        self.device = device
        self.tp = H.mesh_width(device)

    def __call__(self, x, **kwargs):
        x = H.as_device(x, self.device)
        rows = x.shape[-2]
        half = x.shape[-1] // 2

        gate = ttnn.slice(x, [0, 0, 0], [x.shape[0], rows, half])
        up = ttnn.slice(x, [0, 0, half], [x.shape[0], rows, 2 * half])

        sharded = self.tp > 1 and half % self.tp == 0
        if sharded:
            gate = ttnn.mesh_partition(gate, dim=-1)
            up = ttnn.mesh_partition(up, dim=-1)

        out = ttnn.multiply(ttnn.silu(gate), up)
        if sharded:
            out = ttnn.all_gather(out, dim=-1)
        return out


def build(device, torch_module):
    return TtFlux2SwiGLU(device, torch_module)


def flux2_swi_g_l_u(device, torch_module, x, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(x, **kwargs)
