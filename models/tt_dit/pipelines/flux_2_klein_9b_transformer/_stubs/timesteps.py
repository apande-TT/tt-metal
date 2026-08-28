# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `timesteps` of FLUX.2-klein-9B.

HF reference: `diffusers.models.embeddings.Timesteps`, reached as
`Flux2Transformer2DModel.time_guidance_embed.time_proj` (256 channels,
`flip_sin_to_cos=True`, `downscale_freq_shift=0`). It wraps
`get_timestep_embedding`::

    f   = exp(-log(10000) * arange(128) / (128 - downscale_freq_shift))
    emb = scale * (timesteps[:, None] * f[None, :])
    emb = cat([sin(emb), cos(emb)], -1)     # flipped to [cos, sin] here

Parameter-free. The port is a broadcast multiply of the constant frequency row
by the timestep column, then `ttnn.cos` / `ttnn.sin`, then one concat — the
`scale` factor folds into the frequency row, since `scale * (t * f)` is
`t * (scale * f)`.

Precision
---------
The product is an ABSOLUTE angle: the pipeline feeds `timestep * 1000`, so it
reaches ~1e3, where a single bfloat16 ulp is several radians and the sinusoid
would be noise. So the row is FLOAT32 and the multiply is done in fp32, where
the SFPU trig matches the torch reference exactly over this range. A matmul
against a `(1, 128)` row would be the natural way to write the outer product
but is NOT accurate enough — it rounds its operands to bfloat16 per pass, which
measured ~0.24 rad of argument error at t = 256. The broadcast multiply is
exact.

Tensor-parallel placement (TP=8)
--------------------------------
REPLICATED. This is a sinusoidal position table with no weights, and the
principle for tables is to keep them whole: every chip needs all 256 channels,
and each computes the identical row in a handful of SFPU ops. Splitting the
channel axis would only add a gather.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["time_guidance_embed.time_proj"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtTimesteps:
    def __init__(self, device, torch_module):
        self.device = device
        self.freqs = H.timestep_freqs(torch_module, device)
        self.flip_sin_to_cos = bool(torch_module.flip_sin_to_cos)

    def __call__(self, timesteps, **kwargs):
        t = H.timestep_column(H.as_device(timesteps, self.device))
        angle = ttnn.multiply(self.freqs, ttnn.typecast(t, ttnn.float32))
        cos, sin = ttnn.cos(angle), ttnn.sin(angle)
        parts = [cos, sin] if self.flip_sin_to_cos else [sin, cos]
        return ttnn.concat(parts, dim=-1)


def build(device, torch_module):
    return TtTimesteps(device, torch_module)


def timesteps(device, torch_module, timesteps_in, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(timesteps_in, **kwargs)
