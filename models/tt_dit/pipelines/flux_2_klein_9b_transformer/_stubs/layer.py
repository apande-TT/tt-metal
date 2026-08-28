# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `layer` of FLUX.2-klein-9B.

HF reference: `Flux2Transformer2DModel.transformer_blocks[0].norm1` — the
image stream's pre-attention `nn.LayerNorm(4096, elementwise_affine=False,
eps=1e-6)`. That is the module the capture step hooked for this generic role,
and it is the norm every double-stream block applies before its AdaLN
modulation scales and shifts the result.

Affine-free, so the whole port is one `ttnn.layer_norm`. The `elementwise_affine
= True` case is handled too, for a checkpoint that enables it.

(The scaffold seeded this file with a copy of the Llama vision LayerNorm; the
maths is the same, but that class is wired to `ModelArgs` / weight-cache paths
that do not exist here.)

Tensor-parallel placement (TP=8)
--------------------------------
REPLICATED — and deliberately so. LayerNorm reduces over the full hidden
dimension, so a width-sharded activation would need a distributed norm
(`layer_norm_pre_all_gather` → gather stats → `layer_norm_post_all_gather`)
plus a gather to get back to full width. In this port family the residual
stream is replicated across the mesh and only the projections shard (see the
`encoder_stack` / `flux2_transformer_block` ports, whose four LayerNorms are
replicated for the same reason), so every chip can compute the identical
normalization locally in one pass. Two collectives would buy nothing for a
module with no weights.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["transformer_blocks[0].norm1"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2LayerNorm:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()
        self.eps = float(torch_module.eps)

        def affine(p):
            if p is None:
                return None
            return H.stage(p.detach().float().reshape(1, 1, -1), device)

        self.weight = affine(getattr(torch_module, "weight", None))
        self.bias = affine(getattr(torch_module, "bias", None))

    def __call__(self, x, **kwargs):
        return ttnn.layer_norm(
            H.as_device(x, self.device),
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias,
            compute_kernel_config=self.cfg,
        )


def build(device, torch_module):
    return TtFlux2LayerNorm(device, torch_module)


def layer(device, torch_module, x, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(x, **kwargs)
