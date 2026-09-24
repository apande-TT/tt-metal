# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM-sharded decode matmuls for the folded expert banks.

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


def as_rows(x):
    """(B, T, K) -> (1, B*T, K) token rows, so a projection runs as one 2-D
    matmul rather than a B-way batched one. A view when T is tile-aligned."""
    B, T, K = [int(v) for v in x.shape]
    if B == 1:
        return x
    if T % TILE == 0:
        return ttnn.reshape(x, [1, B * T, K])
    rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, B * T, K])
    return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)


def from_rows(y, B, T):
    """Inverse of as_rows: (1, B*T, N) -> (B, T, N)."""
    if B == 1:
        return y
    N = int(y.shape[-1])
    if T % TILE == 0:
        return ttnn.reshape(y, [B, T, N])
    rm = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), [B, T, N])
    return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)


def _banks(device):
    g = device.dram_grid_size()
    assert g.y == 1, "DRAM sharding assumes a 1-row DRAM grid"
    return g.x


def _core_grid(grid, cores):
    """A cores-sized rectangle inside the compute grid (widest first), or None."""
    for cols in range(min(cores, grid.x), 0, -1):
        if cores % cols == 0 and cores // cols <= grid.y:
            return ttnn.CoreGrid(y=cores // cols, x=cols)
    return None


def plan(device, k, n, max_pad=0.05):
    """(in0 cores, padded n) for a (k, n) weight. The activation is width-sharded
    over `cores`, which must divide k's tiles and fit the grid; n is padded to a
    multiple of both `cores` and the DRAM bank count. Take the most cores whose
    padding stays within max_pad of n (else the least-padded option)."""
    banks = _banks(device)
    grid = device.compute_with_storage_grid_size()
    k_tiles, n_tiles = k // TILE, math.ceil(n / TILE)
    options = []
    for c in range(1, grid.x * grid.y + 1):
        if k_tiles % c or _core_grid(grid, c) is None:
            continue
        step = c * banks // math.gcd(c, banks)
        options.append((c, math.ceil(n_tiles / step) * step))
    ok = [o for o in options if o[1] - n_tiles <= max_pad * n_tiles]
    c, n_pad_tiles = max(ok) if ok else min(options, key=lambda o: (o[1], -o[0]))
    return c, n_pad_tiles * TILE


def _workers_per_bank(device, n_tiles, max_worker_n=96):
    """Compute runs on reader workers beside each DRAM bank, each owning its
    bank's n/banks output tiles; split a wide bank shard over up to 3 workers
    (the op's limit) so the per-worker output fits L1."""
    shard = n_tiles // _banks(device)
    for w in (1, 2, 3):
        if shard % w == 0 and shard // w <= max_worker_n:
            return w
    return max(w for w in (1, 2, 3) if shard % w == 0)


def weight_memcfg(device, k, n):
    """(k, n) weight WIDTH-sharded over every DRAM bank; n must be padded per plan()."""
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
    _, n_pad = plan(device, k, n)
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


def matmul(device, x, w, n_out, ckc, dtype=ttnn.bfloat16, fused_activation=None):
    """x (1, M<=32, k) interleaved @ DRAM-sharded w (k, n_pad) -> (1, M, n_out) interleaved.

    Mirrors the op's unit-test contract: bf16 rank-4 activation width-sharded
    in L1, op-chosen width-sharded L1 output."""
    k = int(x.shape[-1])
    M = int(x.shape[-2])
    n_pad = int(w.shape[-1])
    k_tiles, n_tiles = k // TILE, n_pad // TILE
    cores, planned = plan(device, k, n_out)
    assert planned == n_pad, f"weight padded to {n_pad}, plan says {planned}"
    core_grid = _core_grid(device.compute_with_storage_grid_size(), cores)
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
        fused_activation=fused_activation,
        num_workers_per_dram_bank=_workers_per_bank(device, n_tiles),
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


def upload_chunks(device, w, n_chunks, mesh_shape=None, dtype=ttnn.bfloat4_b):
    """upload_weight for a weight too wide for one DRAM-sharded call (each bank
    worker's output block must fit L1): split N into n_chunks column blocks."""
    n = int(w.shape[-1])
    assert n % n_chunks == 0
    step = n // n_chunks
    return [upload_weight(device, w[..., i * step : (i + 1) * step], mesh_shape, dtype) for i in range(n_chunks)]


def matmul_chunks(device, x, ws, n_out, ckc, dtype=ttnn.bfloat16, fused_activation=None):
    """matmul() over column-block chunks from upload_chunks, concatenated along N."""
    step = n_out // len(ws)
    outs = [matmul(device, x, w, step, ckc, dtype, fused_activation) for w in ws]
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=-1)
