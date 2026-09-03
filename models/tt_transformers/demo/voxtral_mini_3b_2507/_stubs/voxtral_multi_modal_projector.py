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


# AUDIO-TOWER PROJECTION FIDELITY, MATCHED TO THE bf16 WEIGHTS.  These projections keep bf16
# weights, and HiFi4 makes the math engine take FOUR passes over operands that hold TWO passes
# worth of mantissa -- the profiler tags every one of them compute-bound ("SLOW", not DRAM) on a
# full 110-core grid, so the math is the critical path and the extra passes are pure waste.
# HiFi2 is the documented pairing for bf16 (GUIDELINES/01 section 12; LoFi rarely wins at bf16,
# so this stops at HiFi2 rather than dropping all the way).  The layer_norms and SDPA stay at
# HiFi4 + fp32_dest_acc_en=True: this tower's own repair history records it losing PCC when its
# reductions ran at a lower fidelity, and softmax/variance accumulation is where that compounds.
_PROJ_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
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


def _to_device(t, device):
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
        self.linear_1_weight = _to_device(torch_module.linear_1.weight.T.contiguous().float(), device)
        self.linear_2_weight = _to_device(torch_module.linear_2.weight.T.contiguous().float(), device)

    def __call__(self, x, **kwargs):
        x = _DS.mm(self.device, x, self.linear_1_weight, _PROJ_CFG)
        x = ttnn.gelu(x)
        x = _DS.mm(self.device, x, self.linear_2_weight, _PROJ_CFG)
        return x


def build(device, torch_module):
    return TtVoxtralMultiModalProjector(device, torch_module)
