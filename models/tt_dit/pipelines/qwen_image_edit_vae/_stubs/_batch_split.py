# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How a Wan VAE stack (encoder / decoder port) lays a batch of BTHWC activations over a 2-D mesh.

batch_parallel:
  False   H over mesh axis 0, W over axis 1 (spatial: every conv exchanges halos)
  True    batch over axis 0, W over axis 1
  "full"  batch over axis 0, then axis 1: whole images per chip, no H / W partition, so no conv needs
          a halo exchange or width masking; a batch that does not divide the mesh is padded with zero
          images, dropped again on the way back
"""

from __future__ import annotations

import ttnn
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig

H_AXIS = 0
W_AXIS = 1


def mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class BatchSplit:
    def __init__(self, device, batch_parallel=False):
        shape = mesh_shape(device)
        # mesh axes the batch is split over, in partition order (gathered back in reverse)
        self.axes = [H_AXIS] if (batch_parallel and shape[H_AXIS] > 1) else []
        if batch_parallel == "full" and shape[W_AXIS] > 1:
            self.axes.append(W_AXIS)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1 if H_AXIS in self.axes else shape[H_AXIS], mesh_axis=H_AXIS),
            width_parallel=ParallelFactor(factor=1 if W_AXIS in self.axes else shape[W_AXIS], mesh_axis=W_AXIS),
        )
        self.factor = 1
        for a in self.axes:
            self.factor *= shape[a]

    def scatter(self, x):
        """Replicated row-major BTHWC -> this chip's part. Returns (part, pad_b)."""
        pc = self.parallel_config
        H, W = x.shape[2], x.shape[3]
        assert (
            H % pc.height_parallel.factor == 0 and W % pc.width_parallel.factor == 0
        ), f"{H}x{W} must divide the {pc.height_parallel.factor}x{pc.width_parallel.factor} spatial mesh"
        pad_b = -x.shape[0] % self.factor
        if pad_b:
            x = ttnn.pad(x, [(0, pad_b), (0, 0), (0, 0), (0, 0), (0, 0)], value=0.0)
        for a in self.axes:
            x = ttnn.mesh_partition(x, dim=0, cluster_axis=a)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        return x, pad_b

    def gather(self, out, ccl_manager, B, pad_b):
        """This chip's BTHWC part -> the replicated [B, ...] whole."""
        pc = self.parallel_config
        if pc.width_parallel.factor > 1:
            out = ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        if pc.height_parallel.factor > 1:
            out = ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
        for a in reversed(self.axes):
            out = ccl_manager.all_gather(out, dim=0, mesh_axis=a, use_hyperparams=False)
        if pad_b:
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (B,) + tuple(out.shape)[1:])
        return out
