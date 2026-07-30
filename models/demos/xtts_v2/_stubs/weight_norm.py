# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `weight_norm` for coqui/XTTS-v2.

HF submodule: ``hifigan_decoder.waveform_decoder.ups.0.parametrizations.weight.0``
— a ``torch.nn.utils.parametrizations._WeightNorm`` parametrization (``dim=0``).
``forward(weight_g, weight_v)`` recomputes the weight-normed weight::

    torch._weight_norm(v, g, dim=0)
      == v * (g / ||v||)          where ||v|| = sqrt(sum_{c != 0} v^2), keepdim

Here ``g`` is ``[512, 1, 1]`` and ``v`` is ``[512, 256, 16]`` (a ConvTranspose1d
weight); the norm is taken over the in-channel and kernel axes for each of the
512 output channels, giving a per-output-channel rescale.

Native strategy
---------------
Pure ttnn elementwise + reduction: square ``v``, sum over the non-``dim`` axes
(keepdim) for the L2 norm, then ``v * g / norm`` with the ``[512,1,1]`` factor
broadcast over ``v``. ``weight_v`` arrives as a raw torch tensor (the PCC harness
only marshals the primary arg onto the device), so it is staged onto the mesh
with ``ttnn.from_torch`` — input marshalling, not host compute. float32 + fp32
accumulation holds PCC.

This is a per-channel rescale with no large matmul weight to split (it has no
parameters of its own — g and v are the inputs), so it is a replicate-only role:
every tensor is staged REPLICATED across the mesh and the result matches the
single-device golden bit-for-bit.
"""

from __future__ import annotations

import torch

import ttnn


def build(device, torch_module):
    dim = int(getattr(torch_module, "dim", 0))

    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )

    def _coerce(t):
        if isinstance(t, ttnn.Tensor):
            return ttnn.typecast(t, ttnn.float32) if t.get_dtype() != ttnn.float32 else t
        return ttnn.from_torch(
            t.contiguous().to(torch.float32), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def forward(weight_g, weight_v=None, **_):
        g = _coerce(weight_g)
        v = _coerce(weight_v)
        # L2 norm of v over every axis except `dim` (=0 here), keepdim -> [512,1,1].
        rank = len(v.shape)
        reduce_dims = [d for d in range(rank) if d != dim]
        vsq = ttnn.multiply(v, v)
        nrm = ttnn.sqrt(ttnn.sum(vsq, dim=reduce_dims, keepdim=True))
        # w = v * (g / ||v||), the [512,1,1] factor broadcast over v.
        factor = ttnn.multiply(g, ttnn.reciprocal(nrm))
        return ttnn.multiply(v, factor)

    return forward
