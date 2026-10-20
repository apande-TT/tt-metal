# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The decode all_reduce of one tile row of bf16 partials as ONE collective (ttnn.experimental.all_reduce_async).

The reduce-scatter + all_gather pair it replaces costs two collective launches (~10 us each on QB2's ring). Here every
device multicasts its whole width-sharded partial row into a scratch buffer on each device's output cores, and those
cores add the TP blocks in an fp32 DEST and round the sum to bf16 once (in a bf16 DEST, like the reduce-scatter's
kernel, the sum rounds after each add: text logits PCC 0.995078 -> 0.994993 when the order changed).

The partials and the sum share the residual add's own width shards (the decode norms' 40-core layout): the streamed
projections write their partial tiles there and the add reads the sum natively (non-native shards cost it ~2.5 us).
That makes the scratch TP x 4 tiles a core (32 KB) -- it fits under the L1 top only because the prefill final norm runs
without its all-ones weight row (ministral3_r_m_s_norm UNIT_PREFILL_NO_WEIGHT).

ONE scratch buffer and ONE global semaphore serve both call sites (o_proj, MLP down) of every layer: two uses are a
whole sub-layer apart -- no device starts the next all_reduce before every device has sent its blocks of this one, and
the output cores reset the semaphore as soon as all TP blocks have landed."""

from __future__ import annotations

import ttnn

_SHARED = {}


class DecodeAllReduce:
    @classmethod
    def shared(cls, device, mem, tp, topology, num_links):
        key = id(device)
        if key not in _SHARED or _SHARED[key].device is not device:
            _SHARED[key] = cls(device, mem, tp, topology, num_links)
        return _SHARED[key]

    def __init__(self, device, mem, tp, topology, num_links):
        """mem: the WIDTH_SHARDED L1 layout of the partials and of the sum."""
        assert mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED, mem
        self.device, self.tp, self.topology, self.num_links = device, tp, topology, num_links
        self.mem = mem
        spec = mem.shard_spec
        h, w = int(spec.shape[0]), int(spec.shape[1])
        self.buffer = ttnn.empty(
            [1, 1, h, w * tp * spec.grid.num_cores()],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(spec.grid, [h, w * tp], spec.orientation),
            ),
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
