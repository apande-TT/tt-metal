# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open / close the Kolibri-1 mesh for the standalone entry points (the demo and the zero-argument
selftests). Tests get the same mesh from the repo's `mesh_device` fixture instead."""
from __future__ import annotations

import ttnn

# TP=4 x DP=1 on the QB2's four Blackhole chips (tensor parallel over the columns).
MESH_SHAPE = (1, 4)
L1_SMALL_SIZE = 24576
# Trace buffer per chip, sized from the larger stage trace (prefill: ~45 ops x 50 layers + sampler).
TRACE_REGION_SIZE = 256 * 1024 * 1024


def device_params(trace_region_size: int = TRACE_REGION_SIZE) -> dict:
    return {
        "l1_small_size": L1_SMALL_SIZE,
        "trace_region_size": trace_region_size,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    }


def open_mesh(trace_region_size: int = TRACE_REGION_SIZE):
    """FABRIC_1D first (the all_gather / all_reduce need it; tt-metal auto-discovers the topology), then a
    1x4 mesh, or a single chip if fewer than four are present (noted in the output)."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    shape = MESH_SHAPE
    if ttnn.get_num_devices() < MESH_SHAPE[0] * MESH_SHAPE[1]:
        print(f"[mesh] only {ttnn.get_num_devices()} device(s) present: falling back to a single chip (TP=1)")
        shape = (1, 1)
    return ttnn.open_mesh_device(
        ttnn.MeshShape(*shape), l1_small_size=L1_SMALL_SIZE, trace_region_size=trace_region_size, num_command_queues=1
    )


def close_mesh(mesh) -> None:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
