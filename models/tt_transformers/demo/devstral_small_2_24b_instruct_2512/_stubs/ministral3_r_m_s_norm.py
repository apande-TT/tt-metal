# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# >>> MACHINE-GENERATED stub (ADAPT — canonical-wrapper path) <<<
"""ADAPT-status stub for `ministral3_r_m_s_norm` of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

The reuse_registry mapped this component to the canonical TT impl at:

    models/common/rmsnorm.py

Status was REUSE; the global PCC gate then failed → force_adapt_all
demoted to ADAPT. This stub IS the per-component starting point:
it imports the canonical class and delegates forward to it.

ADAPT semantics:
  Iter 0 (no LLM): run this stub as-is. If per-component PCC >= 0.99,
    GRADUATE — the canonical impl was already correct for this model.
  Iter 1+ (LLM only enters if iter 0 PCC < 0.99):
    LLM REFINES this stub. Allowed edits: change ModelArgs config,
    adjust constructor args, add small adapter logic in __call__.
    FORBIDDEN: rewriting the canonical class, replacing the import,
    writing your own ttnn ops from scratch, delegating to torch.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.rmsnorm import RMSNorm


class TtMinistral3RMSNorm:
    """Adapter that wraps the canonical TT impl from `models/common/rmsnorm.py`.

    Delegates __call__ to the canonical instance. The LLM may refine
    __init__ / build / __call__ here on PCC failure, but MUST keep the
    canonical class as the underlying implementation.
    """

    # Apply the checkpoint weight as a separate multiply after a unit-gamma canonical norm (see build).
    SEPARATE_GAMMA = True

    def __init__(self, canonical_instance, gamma=None) -> None:
        self._impl = canonical_instance
        self.gamma = gamma

    @classmethod
    def build(cls, device, torch_module):
        # Ministral3RMSNorm: weight * x * rsqrt(mean(x^2) + eps), weight replicated on a mesh.
        weight = torch_module.weight.detach().to(torch.bfloat16)
        # The kernel's fused gamma multiply biases the output ~0.06% low (measured on layer-8 residuals:
        # |tt| / |fp32| = 0.9994 fused, 1.0001 with gamma applied as a separate multiply). The text model
        # compounds that over 80 norms, so the canonical norm runs with a unit gamma and the checkpoint's
        # weight is applied after it.
        gamma = ttnn.from_torch(
            weight.reshape(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None,
        )
        canonical = RMSNorm(
            device=device,
            dim=weight.shape[0],
            state_dict={"norm.weight": torch.ones_like(weight) if cls.SEPARATE_GAMMA else weight},
            weight_key="norm",
            weight_dtype=ttnn.bfloat16,
            eps=float(torch_module.variance_epsilon),
        )
        # The canonical norm computes at HiFi2, whose truncated products bias every output ~0.5% toward
        # zero (measured: |tt| / |fp32 norm of the same input| = 0.995 on the text model). That level error
        # compounds through the residual stream, so the norm runs at HiFi4 with fp32 accumulation.
        canonical.compute_kernel_config_hifi2 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        return cls(canonical, gamma if cls.SEPARATE_GAMMA else None)

    def __call__(self, x, *args, **kwargs):
        # Canonical RMSNorm takes the 4D [1, 1, S, H] prefill layout; restore the caller's rank.
        shape = tuple(x.shape)
        if len(shape) != 4:
            x = ttnn.reshape(x, (1,) * (4 - len(shape)) + shape)
        out = self._impl(x, mode="prefill")
        if self.gamma is not None:
            out = ttnn.multiply(out, self.gamma)
        return ttnn.reshape(out, shape)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtMinistral3RMSNorm.build(device, torch_module)


# Module-level shim with the component's lowercase slug name. Kept for
# backward compatibility with legacy SMOKE/PCC tests that import the
# slug directly.
def ministral3_r_m_s_norm(device, torch_module=None):
    return TtMinistral3RMSNorm.build(device, torch_module)
