# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The mesh this pipeline runs on (2x4 over the run's 8 chips), opened the way the e2e test's
mesh_device fixture opens it. Used by the demo entrypoint and by the pipeline's zero-arg self-tests;
the pipeline itself only ever runs on the device it is handed."""

from __future__ import annotations

import ttnn

MESH_SHAPE = (2, 4)
L1_SMALL_SIZE = 24576  # the conv3d ports' L1 small region (as in their bring-up tests)
# Trace region: sized from the largest per-stage trace. See README "Trace".
TRACE_REGION_SIZE = 1 << 30
DEVICE_PARAMS = {
    "l1_small_size": L1_SMALL_SIZE,
    "trace_region_size": TRACE_REGION_SIZE,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
}


def open_mesh(shape=MESH_SHAPE, trace_region_size=TRACE_REGION_SIZE):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(
        ttnn.MeshShape(*shape),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=trace_region_size,
        num_command_queues=1,
    )


def close_mesh(device):
    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
