# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM-sharded decode matmul for the folded expert-down bank.

At decode the down matmul is (32 tokens) x (Eloc*inter) x hidden: one tile
row of activation against ~160 MB of bf4_b weight per chip, i.e. pure weight
streaming. Width-sharding the weight across the DRAM banks lets each bank's
reader core stream only its own slice (MatmulMultiCoreReuseMultiCastDRAMSharded),
instead of every core pulling interleaved pages over the whole NoC.

The weight's N is padded to a multiple of (banks * 32) and the padded output
columns are sliced off.
"""
from __future__ import annotations

import math

import torch

import ttnn

TILE = 32


def _banks(device):
    g = device.dram_grid_size()
    assert g.y == 1, "DRAM sharding assumes a 1-row DRAM grid"
    return g.x


def padded_n(device, n):
    step = TILE * _banks(device)
    return math.ceil(n / step) * step


def weight_memcfg(device, k, n):
    """(k, n) weight WIDTH-sharded over every DRAM bank; n must be padded_n()."""
    banks = _banks(device)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    spec = ttnn.ShardSpec(grid, (k, n // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def pad_n(w, n_pad):
    """Zero-pad the last dim of a torch weight to n_pad."""
    if w.shape[-1] == n_pad:
        return w
    return torch.nn.functional.pad(w, (0, n_pad - w.shape[-1]))


def upload_weight(device, w, mesh_shape=None, dtype=ttnn.bfloat4_b):
    """Upload a torch weight DRAM-sharded for matmul(). w is (k, n) replicated,
    or (TP, k, n) with chip d of the TP (last) mesh axis getting w[d]."""
    k, n = int(w.shape[-2]), int(w.shape[-1])
    n_pad = padded_n(device, n)
    w = pad_n(w.to(torch.bfloat16), n_pad)
    if w.dim() == 3:
        mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=(None, 0))
        w = w.reshape(-1, n_pad)  # chip d's dim-0 chunk is exactly its (k, n_pad) slice
    else:
        mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
    return ttnn.from_torch(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weight_memcfg(device, k, n_pad),
        **({"mesh_mapper": mapper} if mapper is not None else {}),
    )


def mcast1d_config(device, k, n):
    """Full-grid 1-D in0-multicast config for a one-tile-row (M<=32) matmul:
    N spread over every core, whole K per block step of up to 8 tiles."""
    grid = device.compute_with_storage_grid_size()
    k_tiles, n_tiles = k // TILE, math.ceil(n / TILE)
    per_core_n = math.ceil(n_tiles / (grid.x * grid.y))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=max(d for d in range(1, 9) if k_tiles % d == 0),
        out_subblock_h=1,
        out_subblock_w=max(d for d in range(1, min(4, per_core_n) + 1) if per_core_n % d == 0),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _cores(k_tiles, n_tiles, max_cores=64):
    c = max(c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0)
    return c


def matmul(device, x, w, n_out, ckc, dtype=ttnn.bfloat16):
    """x (1, M<=32, k) interleaved @ DRAM-sharded w (k, n_pad) -> (1, M, n_out) interleaved.

    Mirrors the op's unit-test contract: bf16 rank-4 activation width-sharded
    in L1, op-chosen width-sharded L1 output."""
    k = int(x.shape[-1])
    M = int(x.shape[-2])
    n_pad = int(w.shape[-1])
    k_tiles, n_tiles = k // TILE, n_pad // TILE
    cores = _cores(k_tiles, n_tiles)
    grid = device.compute_with_storage_grid_size()
    cols = min(cores, grid.x)
    while cores % cols:
        cols -= 1
    core_grid = ttnn.CoreGrid(y=cores // cols, x=cols)
    x4 = ttnn.reshape(x, [1, 1, M, k])
    if x4.dtype != ttnn.bfloat16:
        x4 = ttnn.typecast(x4, ttnn.bfloat16)
    in_mem = ttnn.create_sharded_memory_config(
        (1, 1, TILE, k),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    xs = ttnn.to_memory_config(x4, in_mem)
    k_per_core = k_tiles // cores
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=max(d for d in range(1, 9) if k_per_core % d == 0),
        per_core_M=1,
        per_core_N=n_tiles // cores,
        fused_activation=None,
    )
    out = ttnn.matmul(
        xs,
        w,
        program_config=pc,
        memory_config=ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1
        ),
        compute_kernel_config=ckc,
        dtype=dtype,
    )
    ttnn.deallocate(xs)
    out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.slice(out, [0, 0, 0, 0], [1, 1, M, n_out]) if n_out != n_pad else out
