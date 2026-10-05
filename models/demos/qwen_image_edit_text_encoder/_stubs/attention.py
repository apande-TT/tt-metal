# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of `Qwen2_5_VLVisionAttention` (`model.visual.blocks[i].attn`).

Reference math (HF):
    qkv = x @ Wqkv^T + bqkv            x: [S, C], C = num_heads * head_dim
    q, k, v = split(qkv)               each [S, H, D]
    q, k = rope(q, k, cos, sin)        rotate_half RoPE, cos/sin: [S, D]
    o = softmax(q k^T / sqrt(D) + block_mask(cu_seqlens)) v
    out = o @ Wproj^T + bproj

TP scheme (heads split across the TP axis of the mesh, Megatron-style):
    * qkv is column-parallel: each chip owns H/TP heads' q, k and v columns (weight + bias).
    * attention runs per chip on its local heads -- no communication.
    * proj is row-parallel: each chip owns the matching H/TP * D input rows, produces a partial
      sum, and an all_reduce over the TP axis completes it. bproj is replicated and added after.
Any extra (DP) mesh axis replicates everything.

The sequence is padded to a tile multiple; the additive mask hides padded keys, and the
cu_seqlens block structure (per-image / per-window attention) is expressed in the same mask.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

import ttnn

TILE = 32
MASK_NEG = -1e9


def pad_to_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


def mesh_shape(device):
    try:
        shape = tuple(device.shape)
        if len(shape) == 2:
            return shape
    except Exception:
        pass
    n = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    return (1, n)


def shard_mapper(device, dim):
    """Shard `dim` across the TP (column) axis of the mesh, replicate across any DP (row) axis."""
    rows, cols = mesh_shape(device)
    if rows == 1:
        return ttnn.ShardTensorToMesh(device, dim=dim)
    return ttnn.ShardTensor2dMesh(device, mesh_shape=(rows, cols), dims=(None, dim))


def replicate(device):
    return ttnn.ReplicateTensorToMesh(device)


def hifi4_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def upload(device, array, dtype=ttnn.bfloat16, mapper=None):
    """numpy / torch host tensor -> tile-layout device tensor (replicated by default)."""
    t = torch.from_numpy(np.ascontiguousarray(array)) if isinstance(array, np.ndarray) else array.contiguous()
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper or replicate(device)
    )


def block_mask(cu_seqlens, s, s_pad):
    """[1, 1, s_pad, s_pad] additive mask: 0 within a cu_seqlens segment, MASK_NEG elsewhere.
    Padded key columns are masked; padded query rows are left unmasked (they are sliced away)."""
    seg = np.full((s_pad,), -1, dtype=np.int64)
    bounds = [int(v) for v in cu_seqlens]
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg[lo:hi] = i
    mask = np.where((seg[:, None] == seg[None, :]) & (seg[None, :] >= 0), 0.0, MASK_NEG).astype(np.float32)
    mask[s:, :] = 0.0
    return mask.reshape(1, 1, s_pad, s_pad)


def upload_rows(device, arrays, dtype=ttnn.bfloat16, col_dim=None, layout=ttnn.TILE_LAYOUT):
    """Row-staged upload: device (r, c) receives arrays[r], sharded over the TP columns along `col_dim`
    (None = replicated over the columns). One array per mesh row -- used to place DIFFERENT layers on
    the two rows of a 2xN mesh (pipeline stages) while every layer keeps its TP split over the columns."""
    rows, cols = mesh_shape(device)
    assert len(arrays) == rows, f"need one array per mesh row ({rows}), got {len(arrays)}"
    ts = [torch.from_numpy(np.ascontiguousarray(a)) if isinstance(a, np.ndarray) else a.contiguous() for a in arrays]
    t = torch.stack(ts, 0)
    cd = None if col_dim is None else (col_dim + 1 if col_dim >= 0 else col_dim)
    mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(rows, cols), dims=(0, cd))
    tt = ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=mapper)
    return ttnn.reshape(tt, list(tt.shape)[1:])


def exact_all_reduce(y, device, cluster_axis=1):
    """Sum over the TP axis without rounding: gather the float32 partials (bit-exact data movement) and
    add them in float32. ttnn.all_reduce rounds float32 partials at bf16 level (measured ~7e-3 abs on
    O(3) sums on this T3K), which a 32-block fp32 residual stream accumulates."""
    n = mesh_shape(device)[cluster_axis]
    if n == 1:
        return y
    shape = list(y.shape)
    g = ttnn.all_gather(
        ttnn.reshape(y, [1] + shape), dim=0, cluster_axis=cluster_axis, num_links=1, topology=ttnn.Topology.Linear
    )
    out = None
    for i in range(n):
        part = ttnn.slice(g, [i] + [0] * len(shape), [i + 1] + shape)
        out = part if out is None else ttnn.add(out, part)
    ttnn.deallocate(g)
    return ttnn.reshape(out, shape)


def split_bf16(x, limbs=2):
    """float32 -> `limbs` bf16 parts summing to x (2 limbs ~16 mantissa bits, 3 limbs ~24 = float32).
    A matmul consumes its inputs as bf16; the limbs keep every product exact against bf16 weights."""
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    parts = []
    rest = x32
    for i in range(limbs):
        p = ttnn.typecast(rest, ttnn.bfloat16)
        parts.append(p)
        if i + 1 < limbs:
            rest = ttnn.subtract(rest, ttnn.typecast(p, ttnn.float32))
    return tuple(parts)


# Exact-accumulation matmul. A Wormhole tile matmul accumulates each 32-term dot product at ~2^-12 of
# its largest term: measured 3.1e-4 relative error, independent of K, even for exact bf16 products.
# With at most 4 nonzero terms per 32-group the tile result is exact (measured 2.9e-7), and the
# accumulation across K tiles is exact float32. So: carry the float32 input as bf16 limbs and split K
# into 8 lanes of 4 terms per 32-group (zeros elsewhere). Summed in float32, that is a float32-exact
# matmul, at 8x the MACs per limb and no extra weight memory.
#
# Only lanes whose nonzeros sit 8 apart inside every 32-group are exact (measured: k%8 lanes 1.2e-6
# abs; 4 contiguous, 2+2 or 2-per-16 lanes 5e-4..8e-4). Rarely an exact-lane result comes back off by
# a power of two (measured on this T3K: errors of exactly 1, 4, 8 on O(1..5) outputs, ~1 per 6e5,
# deterministic in the data). The position moves when the lane assignment is rotated per 32-group, so
# the product is formed with three 8-spaced partitions (lane = k%8, (k%8 + g)%8, (k%8 + 3g)%8 for
# group g) and the elementwise median of the three is taken: a glitch in one is outvoted by two.
EXACT_LANES = 8
LANE_PATTERNS = ("strided", "rot1", "rot3")
_LANE_MASKS = {}


def precise_config():
    """HiFi4 + fp32 dest, packer_l1_acc OFF. With L1 accumulation on, the auto-picked matmul configs of
    several of this model's shapes lose 3-5x (measured: vision qkv 1.4e-3, LM gate 1.7e-3 relative vs
    3.1e-4 with it off, bf16-exact operands)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _lane_of(k, pattern):
    j, g = k % 32, k // 32
    if pattern == "strided":
        return j % 8
    if pattern == "rot1":
        return (j % 8 + g) % 8
    return (j % 8 + 3 * g) % 8  # "rot3"


def _lane_masks(device, k, dtype, pattern):
    """Constant 0/1 lane masks [1, K], built once per (K, dtype, pattern) with plain Python lists and
    ttnn.Tensor (no torch op), so a first call inside an observed forward stays host-op free."""
    key = (id(device), k, dtype, pattern)
    if key not in _LANE_MASKS:
        lane = [_lane_of(i, pattern) for i in range(k)]
        masks = []
        for r in range(EXACT_LANES):
            vals = [1.0 if lane[i] == r else 0.0 for i in range(k)]
            m = ttnn.Tensor(vals, [1, k], ttnn.float32, ttnn.TILE_LAYOUT, device)
            masks.append(m if dtype == ttnn.float32 else ttnn.typecast(m, dtype))
        _LANE_MASKS[key] = masks
    return _LANE_MASKS[key]


def _lanes(part, pattern="strided"):
    """bf16 part [..., K] -> EXACT_LANES copies; each keeps <= 4 of every 32 K entries (see _lane_of)."""
    return [ttnn.multiply(part, m) for m in _lane_masks(part.device(), part.shape[-1], part.dtype, pattern)]


def _median3(a, b, c):
    return ttnn.maximum(ttnn.minimum(a, b), ttnn.minimum(ttnn.maximum(a, b), c))


# EXACT_MODE "median3": the median over LANE_PATTERNS of the full lane sum, every limb exact-lane
# (8 lanes x limbs x 3 partitions: 48 matmuls per 2-limb linear, 72 at 3 limbs).
# EXACT_MODE "guarded": only the LEADING limb takes exact lanes (one partition, 8 matmuls). The trailing
# limbs are ~2^-8 of it, so their dense accumulation error (~2^-12 of their terms) is ~2^-20 of the
# output's terms, float32's own level. The glitch guard is a bound, not a second partition: a dense
# product is within 2^-12 * sum|terms| <= 2^-12 * |x_row| |w_col| (Cauchy-Schwarz) of the exact one, so
# an exact-lane output further than GUARD_SLACK x that from the dense product holds a glitch -- or the
# dense one does. There the median of (exact, dense, -dense(-x)) is kept: negating the input moves the
# glitch, so two of the three agree. 8 + 2 + (limbs - 1) matmuls.
EXACT_MODE = os.environ.get("QIE_EXACT_MODE", "median3")
GUARD_SLACK = 4.0
_COL_NORMS = {}


def _norm_last(t, keepdim=True):
    """L2 norm over the last dim, float32."""
    t32 = t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)
    return ttnn.sqrt(ttnn.sum(ttnn.multiply(t32, t32), dim=-1, keepdim=keepdim))


def _col_norms(w, transpose_b=False, cache=True):
    """|w[:, n]| (the K reduction's columns) as [..., 1, N] float32; a weight's norms are cached."""
    key = (id(w), transpose_b)
    if cache and key in _COL_NORMS:
        return _COL_NORMS[key][1]
    if transpose_b:  # w: [..., N, K]
        n = ttnn.transpose(_norm_last(w), -2, -1)
    else:  # w: [..., K, N]
        n = _norm_last(ttnn.transpose(w, -2, -1))
        n = ttnn.transpose(n, -2, -1)
    if cache:
        _COL_NORMS[key] = (w, n)  # keep w referenced so its id stays unique
    return n


def _guarded(ex, dn, nn, a_norm, b_norm):
    """ex where it is within the dense floor of dn, else median(ex, dn, -nn); nn = mm(-x), the negated-input
    product before its sign is restored. The negation and the |ex - dn| are fused into the binary ops that
    consume them and the slack scales the [.., M, 1] norm, so no output-sized pass is spent on any of them."""
    tol = ttnn.multiply(ttnn.multiply(a_norm, GUARD_SLACK * 2.0**-12), b_norm)
    near = ttnn.le(ttnn.subtract(ex, dn, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]), tol)
    lo, hi = ttnn.minimum(ex, dn), ttnn.maximum(ex, dn)
    neg = [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]
    med = ttnn.maximum(lo, ttnn.minimum(hi, nn, input_tensor_b_activations=neg))  # median(ex, dn, -nn)
    return ttnn.where(near, ex, med)


def _guarded_sum(mm, parts, b_norm):
    """EXACT_MODE "guarded" for a linear: mm(p) = p @ w; parts = the input's bf16 limbs."""
    lead = parts[0]
    ex = None
    for lane in _lanes(lead, "strided"):
        t = mm(lane)
        ex = t if ex is None else ttnn.add(ex, t)
    dn = mm(lead)
    y = _guarded(ex, dn, mm(ttnn.neg(lead)), _norm_last(lead), b_norm)
    for part in parts[1:]:
        y = ttnn.add(y, mm(part))
    return y


def _exact_sum(mm, parts):
    """median over LANE_PATTERNS of sum_{part, lane} mm(lane(part))."""
    ests = []
    for pattern in LANE_PATTERNS:
        y = None
        for part in parts:
            for lane in _lanes(part, pattern):
                t = mm(lane)
                y = t if y is None else ttnn.add(y, t)
        ests.append(y)
    return _median3(*ests)


_LINEAR_CONFIGS = {}


def _largest_divisor(n, cap, step=1):
    """Largest d <= cap dividing n with d % step == 0 (at least step)."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0 and d % step == 0:
            return d
    return step


def linear_program_config(x, w):
    """2D-multicast config for a batched float32-output linear x [..., M, K] @ w [K, N], or None (auto).

    The auto pick for these shapes is a 1D in0-multicast config with fuse_batch off: it loops over the
    batch and streams the whole weight from DRAM once per batch entry (LM gate/up at B=32: ~1.1 GB per
    matmul per chip), on the N/3 cores the N width fills. Folding the batch into M and multicasting both
    operands over the full grid reads each weight column block once per output block row; the per-core
    float32 output streams through L1 in out_block_h x out_block_w blocks. The K reduction runs in
    in0_block_w steps with float32 partials either way."""
    # tile layout pads every batch entry's rows to a tile, so the batch folds over the PADDED rows (a
    # 110-token text stream is 4 tile rows per sample either way)
    xs, ws = list(x.padded_shape), list(w.shape)
    if x.layout != ttnn.TILE_LAYOUT or len(ws) != 2 or xs[-1] % 32 or ws[-1] % 32:
        return None
    rows = 1
    for d in xs[:-1]:
        rows *= d
    mt, kt, nt = rows // 32, xs[-1] // 32, ws[-1] // 32
    if mt < 8 or kt < 4 or nt < 8:
        return None
    grid = w.device().compute_with_storage_grid_size()
    key = (mt, kt, nt, grid.x, grid.y)
    if key not in _LINEAR_CONFIGS:
        gx, gy = grid.x, grid.y
        pcm = -(-mt // gy)
        pcn = -(-nt // gx)
        pcn += pcn % 2 if pcn > 1 else 0  # an even width admits 2-wide subblocks
        obw = _largest_divisor(pcn, 24)
        sbw = 2 if obw % 2 == 0 else 1
        # output block <= 120 float32 tiles (480 KB), height a multiple of the subblock height
        sbh_pref = 2 if sbw == 2 else 4
        obh = _largest_divisor(pcm, max(1, 120 // obw))
        sbh = _largest_divisor(obh, sbh_pref)
        if sbh * sbw > 4:
            sbh = 1
        bw = _largest_divisor(kt, 4)
        while bw > 1 and obw * bw * 2 * 2048 > 320 * 1024:  # double-buffered bf16 in1 block <= 320 KB
            bw = _largest_divisor(kt, bw - 1)
        # a deeper K step halves the float32 partial spill / reload per output block (packer_l1_acc is off)
        # when a >= 3-row output block still fits ~1000 KB of L1 beside the doubled operand blocks
        # when an output block of >= 4 x 4 tiles still fits ~1000 KB of L1 beside the doubled operand blocks
        if kt % 8 == 0:
            obw8 = _largest_divisor(pcn, 12, sbw)
            obh8 = _largest_divisor(pcm, max(1, (1000 * 1024 // 4096 - obw8 * 8) // (obw8 + 8)), sbh)
            if obh8 >= 4 and obw8 >= 4 and obh8 % sbh == 0 and obw8 % sbw == 0:
                obh, obw, bw = obh8, obw8, 8
        _LINEAR_CONFIGS[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw,
            out_subblock_h=sbh,
            out_subblock_w=sbw,
            out_block_h=obh,
            out_block_w=obw,
            per_core_M=pcm,
            per_core_N=pcn,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
    return _LINEAR_CONFIGS[key]


def split_linear(x, w, bias=None, compute_kernel_config=None, exact=True, limbs=2):
    """float32 x @ bf16 w (+ bias) -> float32. x is carried as `limbs` bf16 parts (exact products). With
    exact=True the K reduction is the exact lane sum (see above); otherwise one matmul per part."""
    cfg = precise_config()
    parts = split_bf16(x, limbs)
    pc = linear_program_config(x, w)
    if pc is not None:  # the hand config's float32 output CB takes the K-step partials in L1 directly
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
    mm = lambda p: ttnn.linear(p, w, compute_kernel_config=cfg, dtype=ttnn.float32, program_config=pc)  # noqa: E731
    if exact and EXACT_MODE == "guarded":
        y = _guarded_sum(mm, parts, _col_norms(w))
    elif exact:
        y = _exact_sum(mm, parts)
    else:
        y = None
        for part in parts:
            t = mm(part)
            y = t if y is None else ttnn.add(y, t)
    # bias added separately in float32 (the fused bias add of ttnn.linear rounds: 3.1e-4 -> 5.1e-4)
    return ttnn.add(y, ttnn.typecast(bias, ttnn.float32)) if bias is not None else y


def rotate_half(t):
    """HF rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2]) by slicing (exact; the x @ R form rounds)."""
    shape = list(t.shape)
    half = shape[-1] // 2
    lo_end = shape[:-1] + [half]
    x1 = ttnn.slice(t, [0] * len(shape), lo_end)
    x2 = ttnn.slice(t, [0] * (len(shape) - 1) + [half], shape)
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def split_matmul(a, b, transpose_b=False, compute_kernel_config=None, exact=True, limbs=2):
    """float32 a @ float32 b over bf16 limbs of both operands, keeping the limb products whose order
    (i + j) is below `limbs` (2 limbs: ah.bh + ah.bl + al.bh; 3 limbs: six terms, ~float32). With
    exact=True each term's K reduction is the exact lane sum (the lanes mask a's last dim)."""
    cfg = precise_config()
    pa_all = split_bf16(a, limbs)
    pb_all = split_bf16(b, limbs)
    terms = [(pa_all[i], pb_all[j]) for i in range(limbs) for j in range(limbs) if i + j < limbs]
    mm = lambda p, q: ttnn.matmul(  # noqa: E731
        p, q, transpose_b=transpose_b, compute_kernel_config=cfg, dtype=ttnn.float32
    )
    if not exact:
        y = None
        for pa, pb in terms:
            t = mm(pa, pb)
            y = t if y is None else ttnn.add(y, t)
        return y
    if EXACT_MODE == "guarded":
        return _guarded_terms(mm, terms, a, b, transpose_b)
    ests = []
    for pattern in LANE_PATTERNS:
        y = None
        for pa, pb in terms:
            for lane in _lanes(pa, pattern):
                t = mm(lane, pb)
                y = t if y is None else ttnn.add(y, t)
        ests.append(y)
    return _median3(*ests)


def _guarded_terms(mm, terms, a, b, transpose_b):
    """EXACT_MODE "guarded" for a @ b over limb terms: the leading term (hi x hi) exact-lane and guarded,
    the rest dense. The bound uses the float32 operands' norms (not cached: b is an activation)."""
    (pa0, pb0), rest = terms[0], terms[1:]
    ex = None
    for lane in _lanes(pa0, "strided"):
        t = mm(lane, pb0)
        ex = t if ex is None else ttnn.add(ex, t)
    dn = mm(pa0, pb0)
    y = _guarded(ex, dn, mm(ttnn.neg(pa0), pb0), _norm_last(a), _col_norms(b, transpose_b, cache=False))
    for pa, pb in rest:
        y = ttnn.add(y, mm(pa, pb))
    return y


def pad_rows(array, s_pad):
    out = np.zeros((s_pad,) + array.shape[1:], dtype=np.float32)
    out[: array.shape[0]] = array
    return out


class TtVisionAttention:
    def __init__(self, device, torch_module):
        self.device = device
        _, self.tp = mesh_shape(device)
        self.num_heads = int(torch_module.num_heads)
        dim = torch_module.qkv.in_features
        self.dim = dim
        self.head_dim = dim // self.num_heads
        assert self.num_heads % self.tp == 0, f"{self.num_heads} heads not divisible by TP={self.tp}"
        self.local_heads = self.num_heads // self.tp
        self.scale = 1.0 / math.sqrt(self.head_dim)
        H, D, hp = self.num_heads, self.head_dim, self.local_heads

        w = torch_module.qkv.weight.detach().float()  # [3*C, C] rows = [q | k | v], each head-major
        b = torch_module.qkv.bias.detach().float()
        wq, wk, wv = (t.reshape(H, D, dim) for t in w.split(dim, dim=0))
        bq, bk, bv = (t.reshape(H, D) for t in b.split(dim, dim=0))
        # Regroup so chip d's contiguous column block is [q_d | k_d | v_d] for its heads.
        w_blocks, b_blocks = [], []
        for d in range(self.tp):
            sl = slice(d * hp, (d + 1) * hp)
            w_blocks += [wq[sl].reshape(-1, dim), wk[sl].reshape(-1, dim), wv[sl].reshape(-1, dim)]
            b_blocks += [bq[sl].reshape(-1), bk[sl].reshape(-1), bv[sl].reshape(-1)]
        wqkv_t = torch.cat(w_blocks, dim=0).t().contiguous()  # [C, 3*C] column-parallel
        bqkv = torch.cat(b_blocks, dim=0).reshape(1, 1, 1, -1)
        self.wqkv = upload(device, wqkv_t, mapper=shard_mapper(device, -1))
        self.bqkv = upload(device, bqkv, mapper=shard_mapper(device, -1))
        # proj: row-parallel -- input rows are head-major, so chip d's rows match its local heads.
        self.wproj = upload(
            device, torch_module.proj.weight.detach().float().t().contiguous(), mapper=shard_mapper(device, 0)
        )
        self.bproj = upload(device, torch_module.proj.bias.detach().float().reshape(1, 1, 1, -1))
        # rotate_half(x) == x @ R
        half = D // 2
        r = np.zeros((D, D), dtype=np.float32)
        r[np.arange(half) + half, np.arange(half)] = -1.0
        r[np.arange(half), np.arange(half) + half] = 1.0
        self.rot = upload(device, r.reshape(1, 1, D, D))
        self.compute_cfg = hifi4_config()
        # precise: q/k/v, RoPE, scores and softmax in float32, output projection in float32 with an exact
        # TP reduce (the bf16 path loses PCC over 32 blocks x 324 tokens; see forward_padded).
        self.precise = False

    def rope_tables(self, cos, sin, s_pad):
        """Host cos/sin [S, D] -> padded [1, 1, s_pad, D] device tables."""
        D = self.head_dim
        cos = pad_rows(np.asarray(cos, dtype=np.float32), s_pad).reshape(1, 1, s_pad, D)
        sin = pad_rows(np.asarray(sin, dtype=np.float32), s_pad).reshape(1, 1, s_pad, D)
        return upload(self.device, cos), upload(self.device, sin)

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask):
        """x: [N, 1, s_pad, C] replicated (N independent images) -> [N, 1, s_pad, C] (bias included).
        tt_cos / tt_sin [1, 1, s_pad, D] and tt_mask [1, 1, s_pad, s_pad] broadcast over N."""
        D, hp = self.head_dim, self.local_heads
        n, s_pad = x.shape[0], x.shape[-2]
        if self.precise:
            return self._forward_precise(x, tt_cos, tt_sin, tt_mask)
        # column-parallel qkv on local heads: [1, 1, S, 3*hp*D]
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=self.compute_cfg)
        loc = hp * D

        def _heads(i):
            t = ttnn.slice(qkv, [0, 0, 0, i * loc], [n, 1, s_pad, (i + 1) * loc])
            t = ttnn.reshape(t, (n, s_pad, hp, D))
            return ttnn.permute(t, (0, 2, 1, 3))  # [N, hp, S, D]

        q, k, v = _heads(0), _heads(1), _heads(2)

        def _rope(t):
            rot = ttnn.matmul(t, self.rot, compute_kernel_config=self.compute_cfg)
            return ttnn.add(ttnn.multiply(t, tt_cos), ttnn.multiply(rot, tt_sin))

        q, k = _rope(q), _rope(k)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_cfg)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), tt_mask)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        o = ttnn.matmul(probs, v, compute_kernel_config=self.compute_cfg)  # [N, hp, S, D]
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (n, 1, s_pad, loc))

        # row-parallel proj: partial sums over local heads, then reduce across the TP axis.
        out = ttnn.linear(o, self.wproj, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        return ttnn.add(out, self.bproj)

    def _forward_precise(self, x, tt_cos, tt_sin, tt_mask):
        D, hp = self.head_dim, self.local_heads
        n, s_pad = x.shape[0], x.shape[-2]
        loc = hp * D
        cfg = self.compute_cfg
        L = getattr(self, "limbs", 2)
        qkv = split_linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=cfg, limbs=L)

        def _heads(i):
            t = ttnn.slice(qkv, [0, 0, 0, i * loc], [n, 1, s_pad, (i + 1) * loc])
            t = ttnn.reshape(t, (n, s_pad, hp, D))
            return ttnn.permute(t, (0, 2, 1, 3))

        q, k, v = _heads(0), _heads(1), _heads(2)
        cos = tt_cos if tt_cos.dtype == ttnn.float32 else ttnn.typecast(tt_cos, ttnn.float32)
        sin = tt_sin if tt_sin.dtype == ttnn.float32 else ttnn.typecast(tt_sin, ttnn.float32)

        def _rope(t):
            return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(rotate_half(t), sin))

        q, k = _rope(q), _rope(k)
        mask = tt_mask if tt_mask.dtype == ttnn.float32 else ttnn.typecast(tt_mask, ttnn.float32)
        scores = split_matmul(q, k, transpose_b=True, compute_kernel_config=cfg, limbs=L)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), mask)
        # explicit float32 softmax (the fused op mis-scales some masked window rows: one token of block
        # 12 measured 6.7% off with exact inputs)
        mx = ttnn.max(scores, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(scores, mx))
        probs = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=cfg))
        o = split_matmul(probs, v, compute_kernel_config=cfg, limbs=L)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (n, 1, s_pad, loc))
        out = split_linear(o, self.wproj, compute_kernel_config=cfg, limbs=L)
        if self.tp > 1:
            out = exact_all_reduce(out, self.device)
        # float32 bias: adding a bf16 tensor to a float32 one rounds the SUM to bf16 (~4e-4 relative)
        return ttnn.add(out, ttnn.typecast(self.bproj, ttnn.float32))

    def __call__(self, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
        s = hidden_states.shape[0]
        s_pad = pad_to_tile(s)
        cos, sin = position_embeddings
        tt_cos, tt_sin = self.rope_tables(cos.float().numpy(), sin.float().numpy(), s_pad)
        bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, s]
        tt_mask = upload(self.device, block_mask(bounds, s, s_pad))

        x = ttnn.reshape(hidden_states, (1, 1, s, hidden_states.shape[-1]))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        out = self.forward_padded(x, tt_cos, tt_sin, tt_mask)
        if s_pad != s:
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, s, out.shape[-1]])
        return ttnn.reshape(out, (s, out.shape[-1]))


def build(device, torch_module=None):
    return TtVisionAttention(device, torch_module)


def attention(device, torch_module=None):
    return TtVisionAttention(device, torch_module)
