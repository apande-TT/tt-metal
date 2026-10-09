# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The transformer's exact-lane QK^T (all three lane patterns and their median) as one C++ Metalium kernel
(ttnn.generic_op, kernels/exact_qk.cpp on the lane_bmm reader / writer).

The folded path (_precise.exact_matmul_bt, FOLD_LANES) runs a concat of the query limbs into the 24-block
folded K, one mask multiply and one matmul over 24 * K per lane pattern, then the median of the three
float32 [Sq, Sk] estimates as four min / max passes. The kernel makes the lane copies per row block,
accumulates each pattern's (term, lane) blocks in its own float32 DEST tile in the folded K order and takes
the median in DEST. Plugged into _precise as its EXACT_QK_FN.
"""
from __future__ import annotations

import ttnn
from models.demos.qwen_image_edit.tt.lane_bmm import _LANES, _TILE, _diag_tiles, lane_program

_DIAGS = {}


def _pattern_diags(device, lanes_per_pattern, kt):
    """The patterns' diagonal lane tiles back to back: tile (p * 8 + r) * kt + t."""
    key = (id(device), tuple(tuple(l) for l in lanes_per_pattern), kt)
    if key not in _DIAGS:
        _DIAGS[key] = ttnn.concat([_diag_tiles(device, l, kt) for l in lanes_per_pattern], dim=0)
    return _DIAGS[key]


def fused_exact_qk(pa, pb, lanes_per_pattern):
    """pa: (hi, lo) bf16 limbs of a [..., M, K]; pb: of b [..., N, K]; lanes_per_pattern: per lane pattern,
    the lane id of each K entry. -> float32 median over the patterns of the exact-lane a @ b^T [..., M, N],
    or None when the operands do not fit the kernel."""
    if len(pa) != 2:
        return None
    k_t = int(pa[0].padded_shape[-1]) // _TILE
    n_p = len(lanes_per_pattern)
    if n_p != 3:
        return None
    diag = lambda device: _pattern_diags(device, lanes_per_pattern, k_t)  # noqa: E731
    return lane_program(
        pa, pb, True, ttnn.DRAM_MEMORY_CONFIG, diag, n_p * _LANES * k_t, n_p * 2 * _LANES * k_t, "exact_qk.cpp"
    )
