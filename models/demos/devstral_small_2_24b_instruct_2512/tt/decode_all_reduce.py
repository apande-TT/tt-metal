# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The decode all_reduce of one tile row of bf16 partials as ONE collective (ttnn.experimental.all_reduce_async).

The reduce-scatter + all_gather pair it replaces costs two collective launches (~10 us each on QB2's ring). Here every
device multicasts its whole width-sharded partial row into a scratch buffer on each device's output cores, and those
cores add the TP blocks in an fp32 DEST and round the sum to bf16 once (in a bf16 DEST, like the reduce-scatter's
kernel, the sum rounds after each add: text logits PCC 0.995078 -> 0.994993 when the order changed).

The partials and the sum share one layout: the streamed projections' own output ownership (tt/down_decode.py: core c
of the 8 x 10 rectangle holds output tiles 2c, 2c + 1), so each stream core writes its partial tiles locally and the
scratch buffer is 2 x TP tiles a core (16 KB). The residual add reads the sum from there into its own width shards.

ONE scratch buffer and ONE global semaphore serve both call sites (o_proj, MLP down) of every layer: L1 at the top of
the bank has room for only one (the prefill final norm's static buffers end ~27 KB below it), and two uses are a whole
sub-layer apart -- no device starts the next all_reduce before every device has sent its blocks of this one, and the
output cores reset the semaphore as soon as all TP blocks have landed."""

from __future__ import annotations

import ttnn

_SHARED = {}


class DecodeAllReduce:
    @classmethod
    def shared(cls, device, cores, tiles_per_core, tp, topology, num_links):
        key = id(device)
        if key not in _SHARED or _SHARED[key].device is not device:
            _SHARED[key] = cls(device, cores, tiles_per_core, tp, topology, num_links)
        return _SHARED[key]

    def __init__(self, device, cores, tiles_per_core, tp, topology, num_links):
        self.device, self.tp, self.topology, self.num_links = device, tp, topology, num_links
        h, w = ttnn.TILE_SIZE, tiles_per_core * ttnn.TILE_SIZE
        shard = lambda width: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [h, width], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.mem = shard(w)
        self.buffer = ttnn.empty(
            [1, 1, h, w * tp * cores.num_cores()],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=shard(w * tp),
        )
        g = device.compute_with_storage_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        self.sem = ttnn.create_global_semaphore(device, grid, 0)

    def __call__(self, t):
        """t: (1, 1, 32, D) bf16 partials in self.mem; returns their sum over the TP mesh, bf16 in self.mem."""
        return ttnn.experimental.all_reduce_async(
            t,
            self.buffer,
            cluster_axis=1,
            mesh_device=self.device,
            multi_device_global_semaphore=self.sem,
            dtype=ttnn.bfloat16,
            memory_config=self.mem,
            topology=self.topology,
            num_links=self.num_links,
            fp32_dest_acc=True,
        )
