# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for token embedding (language_model.embed_tokens).

Simple embedding lookup: indices -> weight table -> output.
"""
from __future__ import annotations

import ttnn


def _as_bf16(t):
    """Narrow a weight to bfloat16 ON THE HOST, before it is ever uploaded.

    ``from_torch(t, dtype=ttnn.bfloat16, ...)`` does NOT convert a float32 `t` on the host: it
    uploads the fp32 bytes, converts the LAYOUT on device in fp32, and only then emits a device
    Typecast to bf16.  A ROW_MAJOR upload of a big fp32 table is worse still -- it tilizes,
    typecasts, then UNtilizes back.  Handing over bf16 makes the requested dtype the dtype that
    arrives, so none of those device ops exist.

    The VALUES are the same either way -- the fp32 -> bf16 rounding happens regardless, on the host
    here instead of on the device one op later.
    """
    return t.bfloat16() if hasattr(t, "bfloat16") else t


def _to_device_rm(t, device):
    t = _as_bf16(t)
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


class TtTokenEmbed:
    def __init__(self, device, torch_module):
        self.device = device
        self.weight = _to_device_rm(torch_module.weight.float(), device)

    def __call__(self, x, **kwargs):
        return ttnn.embedding(x, self.weight, layout=ttnn.TILE_LAYOUT)


def build(device, torch_module):
    return TtTokenEmbed(device, torch_module)
