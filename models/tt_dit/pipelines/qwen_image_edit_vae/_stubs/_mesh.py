# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh layout shared by the VAE ports.

The sharded harness opens the Galaxy as a 1x32 logical line, but CCLs (all_gather, both the async and
the synchronous op) along a 32-chip logical line deadlock on this system, while the same collectives
along either axis of the physical Galaxy grid complete. So every port runs on an 8x4 grid: the mesh
is reshaped in place (same 32 chips, same device order for the harness readback), and the VAE's
spatial split becomes 2D -- H over the 8-axis, W over the 4-axis. 8x4 rather than 4x8 keeps each
chip's W slice >= 2 on the 8x8 latent: a 1-wide W slice (4x8) drops the decode to PCC ~0.969.
"""

from __future__ import annotations

import ttnn

# logical 1xN line -> physical grid it is laid out on
_LINE_TO_GRID = {32: (8, 4)}


def mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


def physical_grid(device):
    """Reshape a 1xN line mesh onto its physical 2D grid (in place); return the device."""
    shape = mesh_shape(device)
    grid = _LINE_TO_GRID.get(shape[1]) if shape[0] == 1 else None
    if grid is not None and hasattr(device, "reshape"):
        device.reshape(ttnn.MeshShape(*grid))
    return device


def pad_to_multiple(x, dim, factor):
    """Zero-pad `x` along `dim` up to the next multiple of `factor`; returns (x, logical_size)."""
    size = x.shape[dim]
    pad = (factor - size % factor) % factor
    if pad:
        padding = [(0, 0)] * len(x.shape)
        padding[dim] = (0, pad)
        x = ttnn.pad(x, padding, value=0.0)
    return x, size
