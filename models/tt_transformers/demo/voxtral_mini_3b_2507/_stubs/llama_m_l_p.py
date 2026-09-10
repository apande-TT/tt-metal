# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for LlamaMLP (language_model.layers[i].mlp).

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
# GATE IS bfloat4_b ON THE DECODE MIRROR ONLY, AND THE SPLIT IS WHAT MAKES IT AFFORDABLE.  The
# note below stands on its numbers -- narrowing this weight EVERYWHERE costs e2e PCC ~0.02 and
# lands under the 0.95 gate -- but it charges that accuracy to prefill as well as decode, and
# prefill does not want the narrower weight: at 3328 rows the projection is compute-bound and
# reads its weight once for the whole batch (measured: gate at bfloat8_b 362.3 us against up at
# bfloat4_b 360.0 us, i.e. the width buys prefill NOTHING).  Decode is the opposite -- 12 cores
# pulling the whole 26.7 MB out of DRAM at 85% of peak, where the width IS the time: 57.4 us per
# layer against up's 38.3.  So the resident weight stays bfloat8_b for prefill and only the
# DRAM-sharded decode mirror is requantised (_DS.attach(..., dtype=_GATE_DECODE_DTYPE)); the
# mirror's serves() is decode-only, so the two never mix.  Half the error surface, all the bytes:
# measured decode 10.0014 -> 9.4768 ms/token (-5.15%) with prefill unchanged at 105.28 ms, e2e
# PCC 0.9635 -> 0.9511.  That leaves only 0.0011 over the gate, so this weight is now the
# model's accuracy ceiling -- the next narrowing anywhere has to buy its budget first.
_GATE_DECODE_DTYPE = ttnn.bfloat4_b
_DOWN_DECODE_DTYPE = ttnn.bfloat4_b

# GATE STAYS bf8_b, AND THAT IS MEASURED.  It is the largest byte lever left -- gate and up are read
# in full on every decode token, so narrowing 3072x8192 from 26.7 MB to 14.2 MB is ~0.9 ms/token
# across 32 layers, ~8% of the token -- but bfloat4_b on gate ALONE drops e2e PCC 0.9598 -> 0.9382,
# under the 0.95 gate (measured 2026-09-05, all four MLP bodies at once).  So `up` is not merely the
# gentlest of the three, it is the ONLY one of the three this model can afford: the earlier
# all-three-bf4_b result (0.8724) was not dominated by up.  Do not retry gate or down at bf4_b
# without a PCC budget won back somewhere else first.

# THE DOWN PROJECTION STAYS bf8_b, AND THAT IS NOW MEASURED ON ITS OWN.  gate and up had each been
# measured separately at bfloat4_b (gate 0.9382, all-three 0.8724) but `down` never had, and it is
# 3072x8192 = 25.2 M parameters per layer, 806 M across 32 layers, worth ~400 MB a decode token.
# The structural argument said it should be the gentlest of the three -- its output is ADDED into
# the residual stream, so the error enters linearly and once, where gate's is perturbed BEFORE the
# nonlinearity.  The argument is wrong.  Measured 2026-09-05 across all four MLP bodies: e2e PCC
# fails at stream 2 with 0.9193 against the 0.95 gate (streams 0/1 held at 0.963/0.962), and the
# same-prefix per-step trace shows it decaying with depth rather than failing at one token -- the
# residual stream is the one tensor every later layer AND every later token reads, so "enters
# linearly and once" understates it.  All three SwiGLU branches are now measured at bf4_b and only
# `up` survives; see _UP_DTYPE.  Do not retry without a PCC budget won back elsewhere.
_DOWN_DTYPE = ttnn.bfloat8_b

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
#
# AND THE APPROX FLAG IS NOT THE FUSED SILU's LEVER, WHICH IS MEASURED.  math_approx_mode selects
# which SFPU implementation a fused activation compiles to, so it looked like the answer to gate
# being the one prefill matmul off its floor: gate and up are the SAME 3584x3072x8192 shape and
# gate runs 644.4 us/call against up's 433.5 (280 TFLOP/s against 416), where the 12 MB of extra
# weight bytes bf8_b reads over up's bf4_b account for under 20 us of the 211.  Setting it True
# moved NOTHING (prefill 130.40 -> 130.94 ms, decode 10.670 -> 10.663, both inside tolerance), so
# the cost is not which sigmoid the SFPU runs -- it is the activation being in the pack loop at
# all, which is why the fix is to defer it to the multiply (see `swiglu`).  Left False.
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
    """Upload a weight.  dtype is a PARAMETER so the MLP projections can go bf8_b without
    dragging the norm weights down with them -- norms must stay wide."""
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


class TtLlamaMLP:
    def __init__(self, device, torch_module):
        self.device = device
        # bf8_b: these three projections are the bulk of the LM's parameters, and at the decode
        # shape they are DRAM-bandwidth-bound, so halving the stored width halves the bytes read.
        self.gate_weight = _to_device(torch_module.gate_proj.weight.T.contiguous().float(), device, ttnn.bfloat8_b)
        self.up_weight = _to_device(torch_module.up_proj.weight.T.contiguous().float(), device, _UP_DTYPE)
        self.down_weight = _to_device(torch_module.down_proj.weight.T.contiguous().float(), device, _DOWN_DTYPE)
        # DECODE-ONLY MIRRORS, width-sharded across the DRAM banks and derived from the interleaved
        # weights ON DEVICE (no second host upload).  See _dram_sharded.py for why prefill keeps the
        # interleaved path.
        # QUANTISE THE MIRROR FROM THE HOST WEIGHT, NOT FROM THE bf8_b COPY.  The other mirrors
        # are derived on device from the interleaved weight because they keep its dtype, but
        # gate's is narrower, and a bfloat8_b -> bfloat4_b typecast is a SECOND requantisation:
        # the 4-bit mantissa is re-derived from values a 16-wide block exponent had already
        # rounded, not from the original ones.  Measured across all four MLP bodies -- via the
        # bf8_b copy e2e PCC 0.9069, straight from the host weight 0.9511, same bytes, same
        # kernel.  One extra interleaved upload per layer, freed as soon as the mirror owns its
        # width-sharded buffers.
        _gate_narrow = _to_device(torch_module.gate_proj.weight.T.contiguous().float(), device, _GATE_DECODE_DTYPE)
        self.gate_ds = _DS.attach(device, _gate_narrow)
        ttnn.deallocate(_gate_narrow)
        self.up_ds = _DS.attach(device, self.up_weight)
        # DOWN'S MIRROR IS NARROWER TOO -- same regime split as gate, funded the same way.
        _down_narrow = _to_device(torch_module.down_proj.weight.T.contiguous().float(), device, _DOWN_DECODE_DTYPE)
        self.down_ds = _DS.attach(device, _down_narrow)
        ttnn.deallocate(_down_narrow)

    def __call__(self, x, **kwargs):
        # Ask for the WHOLE compute grid by name.  These projections profile grid=partial, and at
        # the decode shape (one position x 3072 <-> 8192) a hand-sized program_config cannot be
        # expressed inside L1 -- core_grid lets the op pick a full-grid split that fits.
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


def build(device, torch_module=None):
    return TtLlamaMLP(device, torch_module)
