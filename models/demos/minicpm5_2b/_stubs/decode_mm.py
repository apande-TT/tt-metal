# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program config for a one-tile-row (decode) projection on the 1-D in0-multicast matmul.

ttnn.linear's default for these shapes is the same 1-D multicast but with a two-tile K block, so it
multicasts the activation 32-96 times per projection. The 1-D factory reloads fp32 partials straight
into DEST (UnpackToDestFp32), so a wider K block gives a bit-identical result with far fewer multicasts.
"""
from __future__ import annotations

import ttnn

_MAX_IN0_BLOCK_W = 4  # decode sweep on BH (ms/token): 2 -> 26.86, 4 -> 23.46, 8 -> 23.55, 16 -> 24.00, 64 -> 24.85
_MAX_SUBBLOCK_W = 4  # fp32 dest accumulation, half-sync DEST


def decode_linear_config(device, k, n, per_core_n=None):
    grid = device.compute_with_storage_grid_size()
    k_t, n_t = k // 32, n // 32
    cores = grid.x * grid.y
    per_core_n = per_core_n or -(-n_t // cores)
    in0_block_w = max(d for d in range(1, min(_MAX_IN0_BLOCK_W, k_t) + 1) if k_t % d == 0)
    sub_w = max(d for d in range(1, min(_MAX_SUBBLOCK_W, per_core_n) + 1) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
