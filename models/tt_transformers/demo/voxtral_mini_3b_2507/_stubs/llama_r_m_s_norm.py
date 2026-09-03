# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for LlamaRMSNorm (language_model.layers[i].input_layernorm).

RMSNorm: x * weight / sqrt(mean(x^2) + eps).
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
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


# NORM FIDELITY, SPLIT FROM THE ACCUMULATOR WIDTH.  The reduction is the one place in the stack
# where low fidelity compounds over depth, but that risk lives in the ACCUMULATOR, not in the
# multiplier: fp32_dest_acc_en is what keeps the sum of 3072 squares exact, while math_fidelity
# only sets how many passes the math engine takes over each operand's mantissa.  The inputs here
# are bf16 -- one pass worth of mantissa -- so HiFi4's four passes buy nothing and the norm
# profiled as the largest reduction cost in the decode step (2 per layer, 60 per token).  HiFi2
# with fp32_dest_acc_en STILL held is the pairing; SDPA's softmax stays at HiFi4.
_NORM_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


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


class TtLlamaRMSNorm:
    def __init__(self, device, torch_module):
        self.device = device
        self.eps = torch_module.variance_epsilon
        self.weight = _to_device(torch_module.weight.unsqueeze(0).unsqueeze(0).float(), device)

    def __call__(self, x, **kwargs):
        return _DS.rms_norm(self.device, x, self.weight, self.eps, _NORM_CFG)


def build(device, torch_module):
    return TtLlamaRMSNorm(device, torch_module)
