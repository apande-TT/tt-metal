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

import math

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
# optional fused epilogue of linear: a callable (y_hi, y_lo, bias_rows) -> (y_hi + y_lo) + bias in one pass, with
# bias_rows the float32 bias row replicated over a tile's 32 rows; None: two ttnn.add
LINEAR_SUM_BIAS_FN = None
_BIAS_ROWS = {}
EXACT_QK = True  # within precise mode: exact-lane QK^T (else the dense 3-term split)
FOLD_LANES = False  # exact QK^T as one K-folded matmul per lane pattern (the pipeline switches this on)
EXACT_LANES = 8
# optional fused exact QK^T: a callable ((ah, al), (bh, bl), [lane ids of K per pattern]) -> the float32
# median over PATTERNS of the exact-lane a @ b^T, or None (the stock exact-lane ops then run)
EXACT_QK_FN = None
PATTERNS = ("strided", "rot1", "rot3")
_MASKS = {}


def precise_config(fidelity=None):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4 if fidelity is None else fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


# optional math fidelity of linear's lo-limb product (lo ~ 2^-8 of x, so its own rounding is scaled down by
# the same factor); None: HiFi4 like the hi product
LO_FIDELITY = None


# optional fn(hi, lo, w) -> hi @ w + lo @ w (float32) as one call, e.g. a fused kernel, or None when it does not
# fit (the two stock matmuls then run); the bias is added after it
LIMB_LINEAR_FN = None
# LIMB_LINEAR_FN that also adds a float32 bias in the same pass: (hi, lo, w, bias_rows [32, N]) -> (hi@w + lo@w) + bias,
# or None when it cannot (the caller then adds the bias separately); None: the bias is always a separate add
LIMB_LINEAR_BIAS_FN = None
# the exact QK^T's q / k limbs split straight into L1 for EXACT_QK_FN (its reader re-reads k's rows per q row block);
# False: in the input's memory (DRAM)
EXACT_QK_LIMBS_L1 = False


# optional replacement for the hi/lo split of a float32 tensor: a callable x32 -> (hi, lo), e.g. a fused
# kernel; None: typecast / typecast / subtract / typecast
SPLIT_FN = None


# limbs made by x's producer in the same pass (id(x) -> (x, (hi, lo))): split_bf16(x) hands them out once instead of
# reading x (see register_split)
_PRESPLIT = {}


class PackedLimbs:
    """x's limbs in one tensor t, e.g. gathered in one collective: stacked bf16 [2, ...] (hi, then lo), or paired
    (t typed float32 like x, each page holding a bf16 tile of hi then of lo). Fused limb linears read both limbs from t
    (LIMB_LINEAR_FN / LIMB_LINEAR_BIAS_FN with lo None); unpack() slices stacked ones apart for anything else."""

    def __init__(self, t):
        self.t = t

    def unpack(self):
        if self.t.dtype == ttnn.float32:  # limbs paired per page: only the fused limb linear reads that layout
            raise NotImplementedError("paired limbs need the fused limb linear (LIMB_LINEAR_FN / LIMB_LINEAR_BIAS_FN)")
        shape = [int(d) for d in self.t.shape]
        start = [0] * len(shape)
        hi = ttnn.slice(self.t, start, [1] + shape[1:])
        lo = ttnn.slice(self.t, [1] + start[1:], shape)
        return ttnn.reshape(hi, shape[1:]), ttnn.reshape(lo, shape[1:])


def register_split(x, parts):
    """x's (hi, lo) limbs, made by its producer as SPLIT_FN would: the next split_bf16(x) returns them."""
    _PRESPLIT[id(x)] = (x, parts)


def split_bf16(x, mem=None):
    pre = _PRESPLIT.pop(id(x), None)
    if pre is not None and pre[0] is x and mem is None:
        return pre[1]
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    if SPLIT_FN is not None and x32.layout == ttnn.TILE_LAYOUT and x32.is_sharded() is False:
        return SPLIT_FN(x32) if mem is None else SPLIT_FN(x32, mem)
    hi = ttnn.typecast(x32, ttnn.bfloat16)
    lo = ttnn.typecast(ttnn.subtract(x32, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
    return hi, lo


def linear(x, w, bias=None, compute_kernel_config=None, parts=None):
    """float32 x @ bf16 w -> float32, x as bf16 hi + lo; bias (float32) added separately. parts: x's
    (hi, lo) from split_bf16 when several projections share x (split once, not once per projection)."""
    cfg = precise_config()
    limbs = split_bf16(x) if parts is None else parts
    if isinstance(limbs, PackedLimbs):  # both limbs in one tensor: the fused limb linear reads them there
        y = _packed_linear(limbs.t, w, bias)
        if y is not None:
            return y
        limbs = limbs.unpack()
    hi, lo = limbs
    if (
        LIMB_LINEAR_BIAS_FN is not None
        and LO_FIDELITY is None
        and bias is not None
        and bias.dtype == ttnn.float32
        and math.prod(int(d) for d in list(bias.shape)[:-1]) == 1  # one bias row broadcast over the rows
    ):
        y = LIMB_LINEAR_BIAS_FN(hi, lo, w, _bias_rows(bias))
        if y is not None:
            return y
    y = LIMB_LINEAR_FN(hi, lo, w) if LIMB_LINEAR_FN is not None and LO_FIDELITY is None else None
    if y is not None:
        return (
            y
            if bias is None
            else ttnn.add(y, bias if bias.dtype == ttnn.float32 else ttnn.typecast(bias, ttnn.float32))
        )
    pc = LINEAR_PC_FN(hi, w) if LINEAR_PC_FN is not None else None
    kw = {} if pc is None else {"core_grid": pc} if isinstance(pc, ttnn.CoreGrid) else {"program_config": pc}
    y_hi = ttnn.linear(hi, w, compute_kernel_config=cfg, dtype=ttnn.float32, **kw)
    cfg_lo = cfg if LO_FIDELITY is None else precise_config(LO_FIDELITY)
    y_lo = ttnn.linear(lo, w, compute_kernel_config=cfg_lo, dtype=ttnn.float32, **kw)
    if LINEAR_SUM_BIAS_FN is not None and bias is not None and bias.dtype == ttnn.float32:
        n = int(y_hi.padded_shape[-1])
        one_row = math.prod(int(d) for d in list(bias.shape)[:-1]) == 1  # a bias row broadcast over the rows
        if one_row and n % 32 == 0 and int(bias.padded_shape[-1]) == n and not y_hi.is_sharded():
            y = LINEAR_SUM_BIAS_FN(y_hi, y_lo, _bias_rows(bias))
            ttnn.deallocate(y_hi)
            ttnn.deallocate(y_lo)
            return y
    y = ttnn.add(y_hi, y_lo)
    if bias is not None:
        y = ttnn.add(y, bias if bias.dtype == ttnn.float32 else ttnn.typecast(bias, ttnn.float32))
    return y


def _packed_linear(t, w, bias):
    """linear() of stacked limbs t through the fused limb linear (lo None), or None when it does not take them."""
    if LO_FIDELITY is not None:
        return None
    if (
        LIMB_LINEAR_BIAS_FN is not None
        and bias is not None
        and bias.dtype == ttnn.float32
        and math.prod(int(d) for d in list(bias.shape)[:-1]) == 1
    ):
        return LIMB_LINEAR_BIAS_FN(t, None, w, _bias_rows(bias))
    if LIMB_LINEAR_FN is None:
        return None
    y = LIMB_LINEAR_FN(t, None, w)
    if y is None or bias is None:
        return y
    return ttnn.add(y, bias if bias.dtype == ttnn.float32 else ttnn.typecast(bias, ttnn.float32))


def _bias_rows(bias):
    """float32 [32, N]: the bias row replicated over a tile's rows (built once per bias tensor)."""
    if id(bias) not in _BIAS_ROWS:
        n = int(bias.shape[-1])
        row = ttnn.reshape(bias, (1, n)) if len(bias.shape) != 2 else bias
        zeros = ttnn.zeros([32, n], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=bias.device())
        _BIAS_ROWS[id(bias)] = (bias, ttnn.add(zeros, row))  # 0 + b: the row broadcast over 32 rows, exact
    return _BIAS_ROWS[id(bias)][1]


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
    fused = EXACT_QK_FN is not None and b_rep is None
    lmem = ttnn.L1_MEMORY_CONFIG if fused and EXACT_QK_LIMBS_L1 else None
    ah, al = split_bf16(a, lmem)
    k = a.shape[-1]
    if fused:
        y = EXACT_QK_FN((ah, al), split_bf16(b, lmem), [[_lane_of(i, p) for i in range(k)] for p in PATTERNS])
        if y is not None:
            return y
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
    pre = _PRESPLIT.pop(id(a), None)
    l1_limbs = MATMUL_A_L1 and SPLIT_FN is not None and a.dtype == ttnn.float32 and not a.is_sharded()
    if pre is not None and pre[0] is a:  # a's producer made its limbs (only this product uses them)
        (ah, al), l1_limbs = pre[1], True
    else:
        ah, al = SPLIT_FN(a, ttnn.L1_MEMORY_CONFIG) if l1_limbs else split_bf16(a)
    bh, bl = split_bf16(b) if b_parts is None else b_parts
    if MATMUL_FN is not None:
        y = MATMUL_FN(ah, al, bh, bl)
        if y is not None:
            if l1_limbs:
                ttnn.deallocate(ah)
                ttnn.deallocate(al)
            return y
    if not MATMUL_PRODUCTS_L1:
        mm = lambda p, q: ttnn.matmul(p, q, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
        return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))
    l1 = ttnn.L1_MEMORY_CONFIG
    mm = lambda p, q: ttnn.matmul(p, q, compute_kernel_config=cfg, dtype=ttnn.float32, memory_config=l1)  # noqa: E731
    hh, hl = mm(ah, bh), mm(ah, bl)
    s = ttnn.add(hh, hl, memory_config=l1)
    ttnn.deallocate(hh)
    ttnn.deallocate(hl)
    lh = mm(al, bh)
    if l1_limbs:
        ttnn.deallocate(ah)
        ttnn.deallocate(al)
    y = ttnn.add(s, lh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(s)
    ttnn.deallocate(lh)
    return y


# matmul's float32 limb products and their partial sum in L1 (consumed right away by the adds)
MATMUL_PRODUCTS_L1 = False
# with MATMUL_PRODUCTS_L1 and SPLIT_FN(x, mem): a's bf16 limbs written to L1 by the split, read there by the products
MATMUL_A_L1 = False
# fn(ah, al, bh, bl) -> matmul's (ah @ bh + ah @ bl) + al @ bh as one program, or None to run the ttnn ops
MATMUL_FN = None


def matmul_bt(a, b):
    """float32 a @ float32 b^T as the dense 3-term bf16 split."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b)
    mm = lambda p, q: ttnn.matmul(p, q, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))
