# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opening / closing the mesh this demo runs on (QB2: 4 Blackhole chips as a 1x4 mesh, TP=4 x DP=1).

The demos, the e2e test fixture and the pipeline's standalone self-tests open the device through here, ONCE,
and hand it to tt.pipeline.build_pipeline -- the pipeline itself never opens a device.
"""
from __future__ import annotations

import ttnn

DEFAULT_MESH = (1, 4)


def open_mesh(trace_region_size=0, l1_small_size=24576, mesh_shape=None):
    """Enable the 1D fabric (CCLs need it), then open the planned mesh. Falls back to one device if fewer
    chips are present, and says so."""
    try:
        from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

        rows, cols = resolve_mesh_shape(*(mesh_shape or DEFAULT_MESH))
    except Exception:  # noqa: BLE001 -- the perf package is optional for a plain demo run
        rows, cols = mesh_shape or DEFAULT_MESH
    available = ttnn.get_num_devices()
    if rows * cols > available:
        # Widest power-of-two TP the chips allow (TP must divide the 8 kv heads); bf16 weights need TP >= 2.
        tp = 1
        while tp * 2 <= available and tp * 2 <= cols:
            tp *= 2
        print(f"[devstral] only {available} device(s) available; falling back to a 1x{tp} mesh (TP={tp})", flush=True)
        rows, cols = 1, tp
    if rows * cols > 1:
        _set_fabric(rows * cols)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(rows, cols),
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=1,
    )
    print(f"[devstral] opened mesh {rows}x{cols} (TP={cols} x DP={rows})", flush=True)
    return mesh


def _set_fabric(num_devices):
    """QB2 (P300_X2): the four chips close a ring, so the TP collectives can run Ring over two links (the measured
    QB2 TP4 setup, 8 KB router payload); anything else gets the plain 1D fabric. The stubs read the active
    fabric (ttnn.get_fabric_config) to pick their CCL topology, so the two always agree."""
    ring = False
    if num_devices == 4:
        try:
            ring = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.P300_X2
        except Exception:  # noqa: BLE001 -- unknown cluster: stay on the 1D fabric
            ring = False
    if ring:
        router = ttnn.FabricRouterConfig()
        router.max_packet_payload_size_bytes = 8192
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)
    else:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)


def close_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:  # noqa: BLE001
        pass
