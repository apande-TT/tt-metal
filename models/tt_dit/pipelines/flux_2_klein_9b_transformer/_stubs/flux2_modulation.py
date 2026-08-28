# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flux2_modulation` of FLUX.2-klein-9B.

HF reference: `Flux2Modulation`, reached as
`Flux2Transformer2DModel.double_stream_modulation_img` (dim 4096,
`mod_param_sets=2`, no bias)::

    mod = linear(silu(temb))     # 4096 -> 3 * 2 * 4096 = 24576

`Flux2Modulation.split` is a `@staticmethod` the consuming blocks call on this
output, so the module itself is exactly a SiLU and one projection.

Tensor-parallel scheme (TP=8)
-----------------------------
Plain COLUMN-parallel plus `all_gather(dim=-1)`: 24576 / 8 = 3072 output
features per chip, and the SiLU is elementwise on the replicated input.

Note the packed layout needs NO host re-interleaving here, unlike
`ff.linear_in` or `to_qkv_mlp_proj`. Those feed a LOCAL per-block consumer
(a SwiGLU half, an attention head), so each chip must hold whole blocks. This
output is immediately gathered back to full width, and `all_gather`
concatenates the shards in chip order — so a contiguous column split
reassembles the six modulation chunks in exactly their original order.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["double_stream_modulation_img"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2Modulation:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()

        tp = H.mesh_width(device)
        out_features = int(torch_module.linear.out_features)
        if tp > 1 and out_features % tp:
            tp = 1
        self.tp = tp

        self.w = H.matmul_weight(torch_module.linear, device, tp=tp, mode="col")
        # A column-parallel bias splits with its columns.
        b = torch_module.linear.bias
        self.b = (
            None if b is None else H.stage(b.detach().float().reshape(1, -1), device, shard_dim=1 if tp > 1 else None)
        )

    def __call__(self, temb, **kwargs):
        x = ttnn.silu(H.as_device(temb, self.device))
        out = ttnn.linear(x, self.w, bias=self.b, compute_kernel_config=self.cfg)
        if self.tp > 1:
            out = ttnn.all_gather(out, dim=-1)
        return out


def build(device, torch_module):
    return TtFlux2Modulation(device, torch_module)


def flux2_modulation(device, torch_module, temb, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(temb, **kwargs)
