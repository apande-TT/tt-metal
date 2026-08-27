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


def _as_bf16(t):
    """Narrow a weight to bfloat16 ON THE HOST, before it is ever uploaded.

    ``from_torch(t, dtype=ttnn.bfloat16, ...)`` does NOT convert a float32 `t` on the host: it
    uploads the fp32 bytes, converts the LAYOUT on device in fp32, and only then emits a device
    Typecast to bf16.  Every call site here hands over a `.float()` tensor, so the layout
    conversion was moving 4 bytes per element to produce a 2-byte tensor and paying for a whole
    extra device op to do it.  Handing over bf16 makes the requested dtype the dtype that arrives:
    the conversion moves half the bytes and the typecast has nothing left to do.

    The VALUES are the same either way -- the fp32 -> bf16 rounding happens regardless, on the host
    here instead of on the device one op later.
    """
    return t.bfloat16() if hasattr(t, "bfloat16") else t


def _to_device(t, device):
    """Upload the weight ALREADY TILED, so the device emits no layout-conversion op at all.

    Passing ``device=`` to from_torch is what puts the conversion on the device: the ROW_MAJOR
    bytes go up and a Tilize (or TilizeWithValPadding, for a bias that is not a whole tile) runs
    there.  Building the tensor with NO device argument tilizes on the host instead, and
    ``ttnn.to_device`` is then a plain DMA of bytes that are already in the layout the consumer
    wants -- the conversion does not move to a cheaper kernel, it stops existing.

    This is a WEIGHT path, so the host cost is paid once at build and never in a forward, whereas
    the device op it replaces was on the critical path of the measured region.  Values are
    untouched: the same host-side bf16 tensor, the same tiling, just assembled before the copy
    rather than after it.
    """
    t = _as_bf16(t)
    kw = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT}
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except (AttributeError, TypeError):
        pass
    # NO `device=` HERE, and that is the whole point. `ttnn.open_device()` returns a MeshDevice on
    # this build, so a `isinstance(device, MeshDevice)` branch that kept `device=` was the branch
    # ALWAYS taken -- the host-tilize path below it was dead code. The mapper does not need the
    # tensor placed to describe the replication, so it composes with a host build.
    return ttnn.to_device(ttnn.from_torch(t, **kw), device)


class TtVoxtralMultiModalProjector:
    def __init__(self, device, torch_module):
        self.device = device
        self.linear_1_weight = _to_device(torch_module.linear_1.weight.T.contiguous().float(), device)
        self.linear_2_weight = _to_device(torch_module.linear_2.weight.T.contiguous().float(), device)

    def __call__(self, x, **kwargs):
        x = ttnn.linear(x, self.linear_1_weight, compute_kernel_config=_HIFI4_CFG)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.linear_2_weight, compute_kernel_config=_HIFI4_CFG)
        return x


def build(device, torch_module):
    return TtVoxtralMultiModalProjector(device, torch_module)
