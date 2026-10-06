# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precise mode for the transformer ports (off by default: the graduated numerics stay as they were).

Why: a denoising run feeds every step's output into the next. Over 50 true-CFG steps (scale 4.0) the
per-forward error of the graduated ports (CFG noise PCC 0.99994, ~1% relative) drifts some samples'
trajectories apart (measured: final image PCC down to 0.52 while every stage passed per forward).
Two measured Wormhole floors drive it:
  * matmul inputs are rounded to bf16 before every projection (2^-9 relative per element);
  * a tile matmul accumulates its 32-term dot products at ~2^-12 of the largest term, whatever the
    input splitting. The joint-attention logits reach ~4e4, so QK^T carries absolute errors of whole
    units into a near-argmax softmax.
Precise mode carries matmul inputs as bf16 hi + lo (exact products against bf16 weights). For QK^T it
also splits K into 8 lanes (<= 4 nonzeros per 32-group, 8 apart: exact accumulation, measured 1.2e-6)
and takes the median of 3 rotated lane partitions (a rare power-of-two tile glitch is outvoted).
"""
from __future__ import annotations

import ttnn

ENABLED = False  # the pipeline switches this on; the per-component PCC tests keep the graduated path
SHARE_SPLIT = False  # projections of one activation share its hi/lo split (the pipeline switches this on)
FF_HIDDEN_L1 = False  # the feed-forward's GELU output (the down projection's input) is written to L1
# optional fused GELU + split for the feed-forward: a callable pre-activation x32 -> (hi, lo) of gelu_tanh(x);
# None: ttnn.gelu then split_bf16
GELU_SPLIT_FN = None
# optional program config picker for linear's limb matmuls: a callable (x_limb, w) -> a program_config, a
# ttnn.CoreGrid (ttnn.linear's own pick on that grid) or None (ttnn.linear's own pick)
LINEAR_PC_FN = None
EXACT_QK = True  # within precise mode: exact-lane QK^T (else the dense 3-term split)
FOLD_LANES = False  # exact QK^T as one K-folded matmul per lane pattern (the pipeline switches this on)
EXACT_LANES = 8
PATTERNS = ("strided", "rot1", "rot3")
_MASKS = {}


def precise_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


# optional replacement for the hi/lo split of a float32 tensor: a callable x32 -> (hi, lo), e.g. a fused
# kernel; None: typecast / typecast / subtract / typecast
SPLIT_FN = None


def split_bf16(x):
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    if SPLIT_FN is not None and x32.layout == ttnn.TILE_LAYOUT and x32.is_sharded() is False:
        return SPLIT_FN(x32)
    hi = ttnn.typecast(x32, ttnn.bfloat16)
    lo = ttnn.typecast(ttnn.subtract(x32, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
    return hi, lo


def linear(x, w, bias=None, compute_kernel_config=None, parts=None):
    """float32 x @ bf16 w -> float32, x as bf16 hi + lo; bias (float32) added separately. parts: x's
    (hi, lo) from split_bf16 when several projections share x (split once, not once per projection)."""
    cfg = precise_config()
    hi, lo = split_bf16(x) if parts is None else parts
    pc = LINEAR_PC_FN(hi, w) if LINEAR_PC_FN is not None else None
    kw = {} if pc is None else {"core_grid": pc} if isinstance(pc, ttnn.CoreGrid) else {"program_config": pc}
    y = ttnn.add(
        ttnn.linear(hi, w, compute_kernel_config=cfg, dtype=ttnn.float32, **kw),
        ttnn.linear(lo, w, compute_kernel_config=cfg, dtype=ttnn.float32, **kw),
    )
    if bias is not None:
        y = ttnn.add(y, bias if bias.dtype == ttnn.float32 else ttnn.typecast(bias, ttnn.float32))
    return y


def _lane_of(k, pattern):
    j, g = k % 32, k // 32
    if pattern == "strided":
        return j % 8
    if pattern == "rot1":
        return (j % 8 + g) % 8
    return (j % 8 + 3 * g) % 8


def _masks(device, k, pattern):
    key = (id(device), k, pattern)
    if key not in _MASKS:
        lane = [_lane_of(i, pattern) for i in range(k)]
        ms = []
        for r in range(EXACT_LANES):
            vals = [1.0 if lane[i] == r else 0.0 for i in range(k)]
            ms.append(ttnn.typecast(ttnn.Tensor(vals, [1, k], ttnn.float32, ttnn.TILE_LAYOUT, device), ttnn.bfloat16))
        _MASKS[key] = ms
    return _MASKS[key]


def _median3(a, b, c):
    return ttnn.maximum(ttnn.minimum(a, b), ttnn.minimum(ttnn.maximum(a, b), c))


def _folded_mask(device, k, pattern, terms):
    """[1, terms * EXACT_LANES * k] 0/1 mask: block (term, lane r) keeps K entries of lane r."""
    key = (id(device), k, pattern, "folded", terms)
    if key not in _MASKS:
        lane = [_lane_of(i, pattern) for i in range(k)]
        vals = [1.0 if lane[i] == r else 0.0 for _ in range(terms) for r in range(EXACT_LANES) for i in range(k)]
        n = len(vals)
        _MASKS[key] = ttnn.typecast(ttnn.Tensor(vals, [1, n], ttnn.float32, ttnn.TILE_LAYOUT, device), ttnn.bfloat16)
    return _MASKS[key]


def fold_b(b):
    """b's K-folded operand of the FOLD_LANES exact_matmul_bt ([bh] * n + [bl] * n + [bh] * n along K):
    built once and passed as b_rep when several products share b (e.g. the joint attention's K)."""
    bh, bl = split_bf16(b)
    n = EXACT_LANES
    return ttnn.concat([bh] * n + [bl] * n + [bh] * n, dim=-1)


def exact_matmul_bt(a, b, b_rep=None):
    """float32 a @ float32 b^T (reduction over the last dim of both) with exact accumulation:
    3-term bf16 split (ah.bh + ah.bl + al.bh), 8 exact lanes over K, median of 3 lane partitions.

    The 3 terms x 8 lanes are folded into the reduction dim: a's masked lane copies and b's matching
    pieces are laid side by side along K, so each pattern is ONE matmul over 24*K instead of 24 matmuls
    and 23 float32 adds of the full [Sq, Sk] logits. Every 32-group of the folded K still lies inside one
    (term, lane) block, so each tile dot keeps <= 4 nonzeros and stays exact; the blocks are summed in
    the float32 accumulator, as the adds did."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    k = a.shape[-1]
    if not FOLD_LANES:
        bh, bl = split_bf16(b)
        ests = []
        for pattern in PATTERNS:
            y = None
            for m in _masks(a.device(), k, pattern):
                for pa, pb in ((ah, bh), (ah, bl), (al, bh)):
                    t = ttnn.matmul(
                        ttnn.multiply(pa, m), pb, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32
                    )
                    y = t if y is None else ttnn.add(y, t)
            ests.append(y)
        return _median3(*ests)
    n = EXACT_LANES
    a_rep = ttnn.concat([ah] * (2 * n) + [al] * n, dim=-1)
    own_b = b_rep is None
    if own_b:
        b_rep = fold_b(b)
    ests = []
    for pattern in PATTERNS:
        pa = ttnn.multiply(a_rep, _folded_mask(a.device(), k, pattern, 3))
        ests.append(ttnn.matmul(pa, b_rep, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32))
        ttnn.deallocate(pa)
    ttnn.deallocate(a_rep)
    if own_b:
        ttnn.deallocate(b_rep)
    return _median3(*ests)


def matmul(a, b, b_parts=None):
    """float32 a @ float32 b -> float32 as the 3-term bf16 split (dense accumulation). b_parts: b's
    (hi, lo) when several products share b."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b) if b_parts is None else b_parts
    mm = lambda p, q: ttnn.matmul(p, q, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))


def matmul_bt(a, b):
    """float32 a @ float32 b^T as the dense 3-term bf16 split."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b)
    mm = lambda p, q: ttnn.matmul(p, q, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))
