# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for MLP (language_model.layers[i].mlp).

SwiGLU MLP: gate_proj -> silu * up_proj -> down_proj.
"""
from __future__ import annotations

import ttnn

# THE SwiGLU BRANCH THAT TOLERATES THE NARROWEST WEIGHT.  gate/up/down are 3/4 of every decode
# token's DRAM read, so halving one of them is the largest byte lever left -- but all three at
# bfloat4_b was measured at PCC 0.8724, far under the 0.95 gate.  `up` is the gentlest of the three
# to narrow: `down` writes straight into the residual stream and `gate` is perturbed BEFORE the
# nonlinearity, whereas `up` enters as a plain linear factor of silu(gate).
_UP_DTYPE = ttnn.bfloat4_b

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _dram_sharded():
    """Load the shared DRAM-bank-sharded projection helper that sits next to this stub.

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


# MATCHED TO bf8_b WEIGHTS.  8-bit operands through a HiFi4 kernel make the math engine take four
# passes over one pass worth of precision, which cancels the bandwidth saving; LoFi is the pairing.
_LOFI_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _to_device(t, device, dtype=ttnn.bfloat16):
    # NARROW TO bf16 ON THE HOST.  Callers hand this `.float()` tensors, but the target dtype is
    # bf16, so ttnn used to upload fp32 and fix it up on DEVICE -- the profile showed 42 ms of
    # fp32 Tilize plus 24 ms of fp32->bf16 Typecast doing exactly that.  Narrowing first halves
    # the bytes tilized and removes the typecast entirely.  It is EXACT, not an approximation:
    # both host and device round fp32->bf16 round-to-nearest-even, and these weights came from a
    # bf16 checkpoint that `.float()` had merely widened, so this restores the original values.
    # Block-float targets (bf8_b / bf4_b) are left in fp32 on purpose: their mantissa is
    # derived from a per-block shared exponent, so inserting a bf16 rounding step first can
    # change the packed result.  Only the bf16 path is a pure round-trip removal.
    if dtype == ttnn.bfloat16:
        t = t.bfloat16()
    """Upload a weight.  dtype is a PARAMETER so the projections can go bf8_b on their own."""
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


class TtMlp:
    def __init__(self, device, torch_module):
        self.device = device
        # bf8_b + LoFi + full grid: this is the FOURTH body of the same SwiGLU MLP (llama_m_l_p, the
        # bulk stack in llama_model, llama_decoder_layer, and this one, routed for LM layer 2), and
        # it was the one the earlier precision and grid levers never reached -- it still carried
        # bf16 weights on a HiFi4 kernel with no grid request while the other three had moved.  The
        # projections are DRAM-bandwidth-bound at the decode shape, so the stored width IS the cost.
        self.gate_weight = _to_device(torch_module.gate_proj.weight.T.contiguous().float(), device, ttnn.bfloat8_b)
        self.up_weight = _to_device(torch_module.up_proj.weight.T.contiguous().float(), device, _UP_DTYPE)
        self.down_weight = _to_device(torch_module.down_proj.weight.T.contiguous().float(), device, ttnn.bfloat8_b)
        # Decode-only DRAM-bank-sharded mirrors -- see _dram_sharded.py.
        self.gate_ds = _DS.attach(device, self.gate_weight)
        self.up_ds = _DS.attach(device, self.up_weight)
        self.down_ds = _DS.attach(device, self.down_weight)

    def __call__(self, x, **kwargs):
        g = self.device.compute_with_storage_grid_size()
        grid = ttnn.CoreGrid(y=g.y, x=g.x)
        x = _DS.swiglu(
            x,
            self.gate_weight,
            self.gate_ds,
            self.up_weight,
            self.up_ds,
            self.down_weight,
            self.down_ds,
            _LOFI_CFG,
            grid,
        )
        return x


def build(device, torch_module):
    return TtMlp(device, torch_module)
