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
EXACT_QK = True  # within precise mode: exact-lane QK^T (else the dense 3-term split)
EXACT_LANES = 8
PATTERNS = ("strided", "rot1", "rot3")
_MASKS = {}


def precise_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def split_bf16(x):
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    hi = ttnn.typecast(x32, ttnn.bfloat16)
    lo = ttnn.typecast(ttnn.subtract(x32, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
    return hi, lo


# Voted linears: a dense tile matmul occasionally returns one output off by an exact power of two
# (measured on a 3072 -> 12288 FF projection, hi+lo inputs: 13 of 25.2M outputs off by 1, 2 or 4 while the
# median error is 4.5e-4; those 13 carry ~9x the squared error of all the others). The glitch depends on
# the tile's contents, so each linear is also computed on two dithered inputs x + d (d: one nonzero per
# 32-wide K group, so every tile's contents change) with the dither's exact contribution d @ w removed,
# and the output is the elementwise median of the three estimates. No extra weight copy is needed.
# Off: on Qwen-Image-Edit (50 steps, 4 seeds) it removed the glitches but moved the final latents by
# < 1e-5 PCC (0.999911 -> 0.999915 worst sample) at 3x the linear cost.
VOTE = False
# (position within each 32-group of K, value): bf16-exact and small next to the O(1) activations that
# enter these linears (normalised / modulated streams), so the hi + lo split keeps its precision
_DITHERS = ((0, 0.0625), (16, -0.09375))
_DITHER_CACHE = {}


def _dense(x, w):
    cfg = precise_config()
    hi, lo = split_bf16(x)
    return ttnn.add(
        ttnn.linear(hi, w, compute_kernel_config=cfg, dtype=ttnn.float32),
        ttnn.linear(lo, w, compute_kernel_config=cfg, dtype=ttnn.float32),
    )


def _dithers(w, k):
    """[(d [1, k] fp32, d @ w [1, n] fp32)] for a weight; d @ w has one term per 32-group: exact."""
    key = (id(w), k)
    if key not in _DITHER_CACHE:
        dev = w.device()
        out = []
        for pos, val in _DITHERS:
            vals = [val if i % 32 == pos else 0.0 for i in range(k)]
            d = ttnn.Tensor(vals, [1, k], ttnn.float32, ttnn.TILE_LAYOUT, dev)
            c = ttnn.linear(
                ttnn.typecast(d, ttnn.bfloat16), w, compute_kernel_config=precise_config(), dtype=ttnn.float32
            )
            out.append((d, c))
        _DITHER_CACHE[key] = (w, out)  # keep w referenced so its id stays unique
    return _DITHER_CACHE[key][1]


# exact-lane products for the linears: True = the text-encoder ports' split_linear (8 lanes x 2 limbs x
# 3 rotated partitions, 48 matmuls per linear), "guarded" = one partition checked against the dense
# product (18 matmuls), False = the dense 2-term product. EXACT_PV: the same for P @ V.
EXACT_LINEAR = False
EXACT_PV = False


def _guarded_exact(x, w):
    """One exact-lane partition (8 lanes x hi / lo: 16 matmuls, float32-exact accumulation) instead of the
    median of three (48): an exact-lane output that disagrees with the dense product by more than the
    dense floor allows is a (rare, power-of-two) glitch and takes the dense value instead."""
    from models.demos.qwen_image_edit_text_encoder._stubs.attention import _lanes

    cfg = precise_config()
    hi, lo = split_bf16(x)
    dense = None
    ex = None
    for part in (hi, lo):
        d = ttnn.linear(part, w, compute_kernel_config=cfg, dtype=ttnn.float32)
        dense = d if dense is None else ttnn.add(dense, d)
        for lane in _lanes(part, "strided"):
            t = ttnn.linear(lane, w, compute_kernel_config=cfg, dtype=ttnn.float32)
            ex = t if ex is None else ttnn.add(ex, t)
    tol = ttnn.add(ttnn.multiply(ttnn.abs(dense), 4e-3), 4e-3)
    return ttnn.where(ttnn.le(ttnn.abs(ttnn.subtract(ex, dense)), tol), ex, dense)


# Row budget of one exact-lane linear. The exact path keeps 8 lane copies of x, 3 float32 partition
# estimates and a running sum alive at once: at B=32 x 512 tokens a 1536 -> 3072 projection is ~1.4 GB
# of transients per chip on top of the resident weights (measured: DRAM OOM in the FF out projection).
# Rows are independent and the exact-lane sum is exact whatever the matmul blocking, so running row
# chunks and concatenating computes the same per-row products as one call.
EXACT_ROW_CHUNK = 4096


def _row_chunked(fn, x):
    shape = list(x.shape)
    rows = 1
    for s in shape[:-1]:
        rows *= s
    if rows <= EXACT_ROW_CHUNK:
        return fn(x)
    # split the batch when there is one (whole samples), else the token dim in tile-aligned pieces
    ax = 0 if len(shape) >= 3 and shape[0] > 1 else len(shape) - 2
    per = rows // shape[ax]  # rows per index of `ax`
    step = max(1, EXACT_ROW_CHUNK // per)
    if ax == len(shape) - 2:
        step = max(32, step // 32 * 32)
    outs = []
    for s0 in range(0, shape[ax], step):
        lo = [0] * len(shape)
        hi = list(shape)
        lo[ax], hi[ax] = s0, min(shape[ax], s0 + step)
        outs.append(fn(ttnn.slice(x, lo, hi)))
    y = ttnn.concat(outs, dim=ax)
    for o in outs:
        ttnn.deallocate(o)
    return y


def linear(x, w, bias=None, compute_kernel_config=None):
    """float32 x @ bf16 w -> float32, x as bf16 hi + lo; bias (float32) added separately."""
    x = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    if EXACT_LINEAR == "guarded":
        y = _row_chunked(lambda c: _guarded_exact(c, w), x)
    elif EXACT_LINEAR:
        from models.demos.qwen_image_edit_text_encoder._stubs.attention import split_linear

        y = _row_chunked(lambda c: split_linear(c, w, exact=True, limbs=2), x)
    else:
        y = None
    if y is not None:
        if bias is not None:
            y = ttnn.add(y, bias if bias.dtype == ttnn.float32 else ttnn.typecast(bias, ttnn.float32))
        return y
    y = _dense(x, w)
    if VOTE:
        k = x.shape[-1]
        a, b = (ttnn.subtract(_dense(ttnn.add(x, d), w), c) for d, c in _dithers(w, k))
        y = _median3(y, a, b)
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


# Limbs of q / k in the exact QK^T. Two bf16 limbs carry 16 mantissa bits (2^-17 relative), and this
# checkpoint's attention logits reach ~4e4 into a near-argmax softmax: ~0.3 of absolute logit error.
# Three limbs carry float32's 24 bits (6 limb products instead of 3); the e2e pipeline sets 3.
QK_LIMBS = 2


def split_limbs(x, n):
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    parts, r = [], x32
    for i in range(n):
        p = ttnn.typecast(r, ttnn.bfloat16)
        parts.append(p)
        if i + 1 < n:
            r = ttnn.subtract(r, ttnn.typecast(p, ttnn.float32))
    return parts


def exact_matmul_bt(a, b):
    """float32 a @ float32 b^T (reduction over the last dim of both) with exact accumulation:
    QK_LIMBS-limb bf16 split (products of order < QK_LIMBS), 8 exact lanes over K, median of 3 lane
    partitions."""
    cfg = precise_config()
    pa_, pb_ = split_limbs(a, QK_LIMBS), split_limbs(b, QK_LIMBS)
    terms = [(pa_[i], pb_[j]) for i in range(QK_LIMBS) for j in range(QK_LIMBS) if i + j < QK_LIMBS]
    k = a.shape[-1]
    ests = []
    for pattern in PATTERNS:
        y = None
        for m in _masks(a.device(), k, pattern):
            for pa, pb in terms:
                t = ttnn.matmul(
                    ttnn.multiply(pa, m), pb, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32
                )
                y = t if y is None else ttnn.add(y, t)
        ests.append(y)
    return _median3(*ests)


def _matmul3(a, b):
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b)
    mm = lambda p, q: ttnn.matmul(p, q, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))


def matmul(a, b):
    """float32 a @ float32 b -> float32 as the 3-term bf16 split (dense accumulation). Not voted: its
    operands (attention probabilities ~1e-3) are too small for a fixed dither (see VOTE above)."""
    if EXACT_PV:
        from models.demos.qwen_image_edit_text_encoder._stubs.attention import split_matmul

        return split_matmul(a, b, exact=True, limbs=2)
    return _matmul3(a, b)


def matmul_bt(a, b):
    """float32 a @ float32 b^T as the dense 3-term bf16 split."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b)
    mm = lambda p, q: ttnn.matmul(p, q, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))
