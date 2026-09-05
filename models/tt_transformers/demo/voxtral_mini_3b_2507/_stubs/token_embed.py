# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for token embedding (language_model.embed_tokens).

Simple embedding lookup: indices -> weight table -> output.
"""
from __future__ import annotations

import ttnn


def _to_device_rm(t, device):
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

    def __call__(self, x, layout=ttnn.TILE_LAYOUT, **kwargs):
        # THE OUTPUT LAYOUT IS THE CALLER'S TO CHOOSE.  The gather itself is row-major -- one row of
        # the table per id -- and `layout=TILE` makes the op tilize the result on the way out.  That
        # is what the decode step wants (its consumer is a norm), but the prefill path immediately
        # slices the result on the ROW dim and concatenates the audio embeddings into the gap, and
        # neither the slice bounds nor the concat seam are tile-aligned -- so on a tiled tensor ttnn
        # has to untilize every piece again to do the join.  Letting that caller ask for ROW_MAJOR
        # means the [B, C, hidden] embedding is tilized ONCE, after the join, instead of tilized
        # here and untilized piecewise straight afterwards.
        return ttnn.embedding(x, self.weight, layout=layout)


def build(device, torch_module):
    return TtTokenEmbed(device, torch_module)
