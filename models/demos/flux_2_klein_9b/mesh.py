# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Opening the 1x8 mesh the bring-up graduated on, for RUNNERS -- not for the pipeline.

This module sits OUTSIDE ``tt/`` on purpose.  ``tt/`` is the pipeline: every stage in
it runs on the single device handed to ``build_pipeline``, and it must never open one
of its own.  A second, ad-hoc open would create a competing device with a different
command-queue count, which is exactly what breaks trace capture with
``id < mesh_command_queues_.size()``.

Device ownership therefore belongs to whoever *drives* the pipeline -- the ``demo/``
scripts and the ``tests/e2e/`` fixtures -- and this is the one place that policy is
written down, so all of them open the same mesh:

* the bring-up PCC tests got theirs from the repo's ``mesh_device`` fixture with
  ``device_params={"l1_small_size": 24576, "fabric_config": FABRIC_1D}``, and
* fabric has to be configured BEFORE the mesh is opened and reset after it is closed,
  which a plain script has no fixture to do for it.
"""

from __future__ import annotations

import contextlib

import ttnn

#: what the graduated stubs were verified on (see every RUN_REPORT.md)
MESH_SHAPE = (1, 8)
L1_SMALL_SIZE = 24576

#: The halo any runner that drives the VAE ENCODER AND DECODER in one pass needs.
#:
#: `ttnn.conv2d` keeps its halo/reader buffers in L1_SMALL and the PROGRAM CACHE owns
#: them, so the encoder's are still held when the decoder asks for its own: measured
#: on this mesh at 24576, the encode routes reach 8224 B/bank and the decoder's tail
#: conv at (256, 256, 256, 256) then asks for 2048 B/bank against 1184 B free.  It is
#: a halo size, not a batch cap -- the chunk widths, and so the halo, are the same at
#: B=1 and B=32.
#:
#: It is NOT the default, deliberately.  Raising it for every runner pushes the
#: L1-resident activations of the heads that need no halo into the circular-buffer
#: region, which broke the text head at B=32 even though it has no VAE.  So only the
#: runners that decode ask for it.
VAE_L1_SMALL = 65536


@contextlib.contextmanager
def open_flux_mesh(mesh_shape=MESH_SHAPE, *, l1_small_size=L1_SMALL_SIZE, trace_region_size=None):
    """A 1x8 mesh with FABRIC_1D, closed and reset on the way out.

    One command queue (ttnn's default) plus an optional ``trace_region_size``: the
    trace lever needs the region, and a second queue would put the captured trace on
    a queue the pipeline never replays it on.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
    params = {"l1_small_size": l1_small_size}
    if trace_region_size is not None:
        params["trace_region_size"] = trace_region_size
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **params)
    try:
        yield device
    finally:
        for submesh in device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
