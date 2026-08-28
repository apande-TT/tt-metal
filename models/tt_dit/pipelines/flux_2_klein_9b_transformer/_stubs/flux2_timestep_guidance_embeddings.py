# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flux2_timestep_guidance_embeddings` of FLUX.2-klein-9B.

HF reference: `Flux2TimestepGuidanceEmbeddings`, reached as
`Flux2Transformer2DModel.time_guidance_embed`::

    timesteps_proj = time_proj(timestep)                  # sinusoid, 256 ch
    timesteps_emb  = timestep_embedder(timesteps_proj)    # 256 -> 4096 -> 4096
    if guidance is not None and guidance_embedder is not None:
        return timesteps_emb + guidance_embedder(time_proj(guidance))
    return timesteps_emb

This checkpoint sets `guidance_embeds: false`, so `guidance_embedder` is None
and the `guidance` argument is accepted and ignored. The branch is still
implemented, so the port also covers a guidance-distilled Flux2 checkpoint.

The sinusoid
------------
`get_timestep_embedding` is `cat([sin(t * f), cos(t * f)])`, flipped to
`cat([cos, sin])` because `flip_sin_to_cos=True`. Here it is one broadcast
multiply of the constant frequency row by the timestep column, then `ttnn.cos`
and `ttnn.sin`, then a concat — no per-channel loop.

`t * f` is an absolute angle reaching the timestep's own magnitude (the
pipeline feeds `timestep * 1000`), so it is formed in FLOAT32: one bfloat16 ulp
at that scale is several radians, and the error would land straight on the
sinusoid. The broadcast multiply is exact; a matmul against a `(1, half)` row
is not (it rounds the operands to bfloat16 per pass).

This module holds two small projections and feeds a value every block's
modulation reads, so it stays replicated across the mesh.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["time_guidance_embed"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2TimestepGuidanceEmbeddings:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()

        time_proj = torch_module.time_proj
        self.freqs = H.timestep_freqs(time_proj, device)
        self.flip_sin_to_cos = bool(time_proj.flip_sin_to_cos)

        def embedder(module):
            if module is None:
                return None
            return (
                H.matmul_weight(module.linear_1, device),
                H.bias_vector(module.linear_1, device),
                H.matmul_weight(module.linear_2, device),
                H.bias_vector(module.linear_2, device),
            )

        self.timestep_embedder = embedder(torch_module.timestep_embedder)
        self.guidance_embedder = embedder(getattr(torch_module, "guidance_embedder", None))

    def _sinusoid(self, t):
        angle = ttnn.multiply(self.freqs, ttnn.typecast(H.timestep_column(t), ttnn.float32))
        cos, sin = ttnn.cos(angle), ttnn.sin(angle)
        parts = [cos, sin] if self.flip_sin_to_cos else [sin, cos]
        return ttnn.typecast(ttnn.concat(parts, dim=-1), ttnn.bfloat16)

    def _embed(self, t, weights):
        w1, b1, w2, b2 = weights
        h = ttnn.linear(self._sinusoid(t), w1, bias=b1, compute_kernel_config=self.cfg)
        h = ttnn.silu(h)
        return ttnn.linear(h, w2, bias=b2, compute_kernel_config=self.cfg)

    def __call__(self, timestep, guidance=None, **kwargs):
        timestep = H.as_device(timestep, self.device)
        emb = self._embed(timestep, self.timestep_embedder)

        guidance = H.as_device(guidance, self.device)
        if guidance is not None and self.guidance_embedder is not None:
            emb = ttnn.add(emb, self._embed(guidance, self.guidance_embedder))
        return emb


def build(device, torch_module):
    return TtFlux2TimestepGuidanceEmbeddings(device, torch_module)


def flux2_timestep_guidance_embeddings(device, torch_module, timestep, guidance=None, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(timestep, guidance=guidance, **kwargs)
