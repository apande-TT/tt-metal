# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `patch_embed` of FLUX.2-klein-9B.

HF reference: `Flux2Transformer2DModel.x_embedder` —
`nn.Linear(in_channels=128, inner_dim=4096, bias=False)`, the projection that
lifts each latent patch into the transformer's hidden dim.

There is NO convolutional patchifier here: the config sets `patch_size = 1`, so
the latent is already a token sequence when it reaches the transformer and the
"patch embedding" is a plain linear map. (The scaffold seeded this file with a
copy of the Llama Conv2d patch embedder; that op does not appear in this
model.)

The port is one matmul, replicated across the mesh — the gate classifies this
role as replicate-only, which matches the port family: `x_embedder`'s output IS
the residual stream, which stays replicated here, and its 128-wide input is far
too small to be worth splitting.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["x_embedder"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2PatchEmbed:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()
        self.w = H.matmul_weight(torch_module, device)
        self.b = H.bias_vector(torch_module, device)

    def __call__(self, x, **kwargs):
        return ttnn.linear(H.as_device(x, self.device), self.w, bias=self.b, compute_kernel_config=self.cfg)


def build(device, torch_module):
    return TtFlux2PatchEmbed(device, torch_module)


def patch_embed(device, torch_module, x, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(x, **kwargs)
