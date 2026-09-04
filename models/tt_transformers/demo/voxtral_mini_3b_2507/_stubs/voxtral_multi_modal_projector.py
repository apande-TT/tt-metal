# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralMultiModalProjector (multi_modal_projector).

Linear → GELU → Linear projection from audio features to LLM hidden dim.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# THE ONE COMPONENT THAT NEVER GOT NARROWED.  linear_1 (5120 -> 3072) and linear_2 (3072 -> 3072)
# are the whole parameter mass of this projector -- 31.5 MB and 18.9 MB at bf16 -- and they were
# still the only bf16 matmuls left in the encode stack, at 184.4 us and 115.7 us per call for
# 5.1 ms of the stage.  Every other projection of this shape class in the model measured about 2x
# faster once its weight went bf8_b and its kernel dropped to LoFi to match: the audio tower's fc2
# went 216.8 -> 105.4 us on an IDENTICAL 1504 x 5120 x 1280 shape.  There is no reason this
# projector is different -- it sits between the tower's own bf8_b output and the LM's bf8_b
# embedding stream, so it is the only wide tensor in the chain still carried at double width.
_PROJ_DTYPE = ttnn.bfloat8_b

# LoFi IS THE PAIRING FOR bf8_b, not a further gamble on top of it.  8-bit operands through a
# HiFi2 kernel make the math engine take two passes over one pass worth of mantissa, which cancels
# the bandwidth the narrower weight just bought (GUIDELINES/01 section 12).  fp32_dest_acc_en stays
# False as the matmul preference, which also unlocks wider subblocks.
_PROJ_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _dram_sharded():
    """Load the shared decode-layout helper that sits next to this stub.

    The stubs are imported standalone BY PATH (tt/pipeline._load_stub_module), so they have no
    package context and a relative import is not available to them.
    """
    import importlib.util
    import pathlib
    import sys

    key = "_voxtral_stub__dram_sharded"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, pathlib.Path(__file__).with_name("_dram_sharded.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


_DS = _dram_sharded()


def _to_device(t, device, dtype=ttnn.bfloat16):
    # BLOCK-FLOAT TARGETS SKIP THE HOST NARROWING.  bf8_b/bf4_b derive their mantissa from a
    # per-block shared exponent, so rounding to bf16 first can change the packed result; only the
    # bf16 path below is a pure round-trip removal.
    if dtype != ttnn.bfloat16:
        try:
            if isinstance(device, ttnn.MeshDevice):
                return ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
        except (AttributeError, TypeError):
            pass
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    # NARROW TO bf16 ON THE HOST.  Callers hand this `.float()` tensors, but the target dtype is
    # bf16, so ttnn used to upload fp32 and fix it up on DEVICE -- the profile showed 42 ms of
    # fp32 Tilize plus 24 ms of fp32->bf16 Typecast doing exactly that.  Narrowing first halves
    # the bytes tilized and removes the typecast entirely.  It is EXACT, not an approximation:
    # both host and device round fp32->bf16 round-to-nearest-even, and these weights came from a
    # bf16 checkpoint that `.float()` had merely widened, so this restores the original values.
    t = t.bfloat16()
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class TtVoxtralMultiModalProjector:
    def __init__(self, device, torch_module):
        self.device = device
        self.linear_1_weight = _to_device(torch_module.linear_1.weight.T.contiguous().float(), device, _PROJ_DTYPE)
        self.linear_2_weight = _to_device(torch_module.linear_2.weight.T.contiguous().float(), device, _PROJ_DTYPE)

    def __call__(self, x, **kwargs):
        # ASK FOR THE GRID BY NAME.  `_DS.mm` only names it above a row threshold meant to keep the
        # DECODE shape off it, and this projector runs 384 rows -- below that threshold, so it was
        # routed on 66 of 110 cores.  The threshold is about there being enough OUTPUT tiles to
        # spread, not rows: 384 rows against a 3072-wide weight is 12 x 96 = 1152 output tiles,
        # which is ten per core.  Routing through `linear` with an explicit grid gets those cores
        # without moving the shared threshold and disturbing the call sites that depend on it.
        g = self.device.compute_with_storage_grid_size()
        grid = ttnn.CoreGrid(y=g.y, x=g.x)
        x = _DS.linear(x, self.linear_1_weight, None, _PROJ_CFG, grid)
        x = ttnn.gelu(x)
        x = _DS.linear(x, self.linear_2_weight, None, _PROJ_CFG, grid)
        return x


def build(device, torch_module):
    return TtVoxtralMultiModalProjector(device, torch_module)
