# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight upload with the tilize (and dtype cast) done on the host.

ttnn.from_torch(float32 t, device=..., mesh_mapper=...) copies the row-major float32 shards to every device and
tilizes + typecasts them there: load-time device work that is the same tiles a host tilize makes. Uploading the
finished tiles drops it, but the traced denoise is sensitive to where later DRAM allocations land, so the device
path's transient buffers are still allocated and freed in its order (no kernels run): row-major float32, then the
float32 tiles beside it, the row-major copy freed, then the final tiles (after a cast), then the float32 tiles
freed. Every later allocation lands where it did.
"""
from __future__ import annotations

import torch

import ttnn


def host_tiled_from_torch(t, dtype, layout, device, mesh_mapper):
    """ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=mesh_mapper), tilized on the host
    (float32 sources in TILE layout; anything else takes the stock path)."""
    if t.dtype != torch.float32 or layout != ttnn.TILE_LAYOUT:
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=mesh_mapper)
    host = ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    shape, dram = host.shape, ttnn.DRAM_MEMORY_CONFIG
    src_rm = ttnn.allocate_tensor_on_device(shape, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, device, dram)
    if dtype == ttnn.float32:  # RM -> tilize: the tiles land beside the row-major copy, which is then freed
        out = ttnn.to_device(host, device)
        ttnn.deallocate(src_rm)
        return out
    src_tile = ttnn.allocate_tensor_on_device(shape, ttnn.float32, ttnn.TILE_LAYOUT, device, dram)
    ttnn.deallocate(src_rm)  # RM -> tilize -> typecast: the cast tiles land after the row-major copy is freed
    out = ttnn.to_device(host, device)
    ttnn.deallocate(src_tile)
    return out
