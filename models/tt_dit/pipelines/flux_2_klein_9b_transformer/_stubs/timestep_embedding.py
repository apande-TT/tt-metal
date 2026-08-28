# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timestep_embedding` of FLUX.2-klein-9B.

HF reference: `diffusers.models.embeddings.TimestepEmbedding`, reached as
`Flux2Transformer2DModel.time_guidance_embed.timestep_embedder`
(256 -> 4096 -> 4096, SiLU, no bias)::

    if condition is not None:
        sample = sample + cond_proj(condition)
    sample = linear_2(act(linear_1(sample)))

`cond_proj` and `post_act` are both None in this checkpoint; the `cond_proj`
branch is kept so the port covers the conditioned variant.

The port is replicated across the mesh: it consumes a single timestep vector
and produces the value every block's modulation reads, so there is nothing per
chip to split — the gate classifies this role as replicate-only.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["time_guidance_embed.timestep_embedder"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtTimestepEmbedding:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()

        self.w1 = H.matmul_weight(torch_module.linear_1, device)
        self.b1 = H.bias_vector(torch_module.linear_1, device)
        self.w2 = H.matmul_weight(torch_module.linear_2, device)
        self.b2 = H.bias_vector(torch_module.linear_2, device)

        cond = getattr(torch_module, "cond_proj", None)
        self.w_cond = None if cond is None else H.matmul_weight(cond, device)
        self.b_cond = None if cond is None else H.bias_vector(cond, device)

    def __call__(self, sample, condition=None, **kwargs):
        x = H.as_device(sample, self.device)

        condition = H.as_device(condition, self.device)
        if condition is not None and self.w_cond is not None:
            x = ttnn.add(x, ttnn.linear(condition, self.w_cond, bias=self.b_cond, compute_kernel_config=self.cfg))

        h = ttnn.linear(x, self.w1, bias=self.b1, compute_kernel_config=self.cfg)
        h = ttnn.silu(h)
        return ttnn.linear(h, self.w2, bias=self.b2, compute_kernel_config=self.cfg)


def build(device, torch_module):
    return TtTimestepEmbedding(device, torch_module)


def timestep_embedding(device, torch_module, sample, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(sample, **kwargs)
