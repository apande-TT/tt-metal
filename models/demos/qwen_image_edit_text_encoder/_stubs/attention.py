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
    """Row-staged upload: one array per pipeline stage; the mesh rows are split into len(arrays) equal
    contiguous groups and every device of group s receives arrays[s], sharded over the TP columns
    along `col_dim` (None = replicated over the columns). Used to place DIFFERENT layers on the row
    groups of an RxN mesh (pipeline stages: rows [0, R/2) and [R/2, R) for two stages) while every
    layer keeps its TP split over the columns. On a 2xN mesh this is one array per row."""
    rows, cols = mesh_shape(device)
    n = len(arrays)
    assert n >= 1 and rows % n == 0, f"{n} stages do not split the {rows} mesh rows evenly"
    per = rows // n
    ts = [torch.from_numpy(np.ascontiguousarray(a)) if isinstance(a, np.ndarray) else a.contiguous() for a in arrays]
    ts = [t for t in ts for _ in range(per)]  # stage s on rows [s*per, (s+1)*per)
    t = torch.stack(ts, 0)
    cd = None if col_dim is None else (col_dim + 1 if col_dim >= 0 else col_dim)
    mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(rows, cols), dims=(0, cd))
    tt = ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=mapper)
    return ttnn.reshape(tt, list(tt.shape)[1:])


# L1 bytes for one gathered chunk of exact_all_reduce (0: gather the whole tensor into DRAM)
EXACT_REDUCE_L1_BYTES = 0


def _gather_sum(y, n, cluster_axis, mem=None):
    """all_gather the float32 partials over `cluster_axis` and add the n slices in float32; the gathered
    tensor and the running sum live in `mem` (None: the input's memory config)."""
    shape = list(y.shape)
    kw = {} if mem is None else {"memory_config": mem}
    g = ttnn.all_gather(
        ttnn.reshape(y, [1] + shape), dim=0, cluster_axis=cluster_axis, num_links=1, topology=ttnn.Topology.Linear, **kw
    )
    out = None
    for i in range(n):
        part = ttnn.slice(g, [i] + [0] * len(shape), [i + 1] + shape, **kw)
        out = part if out is None else ttnn.add(out, part, **kw)
    ttnn.deallocate(g)
    return ttnn.reshape(out, shape)


# exact reduce as an all_to_all over the batch (each device receives every partial of its 1/n of the batch),
# local float32 adds in source order, and an all_gather of the sums: ~2(n-1)/n of the tensor on the fabric
# instead of (n-1)x, and 1/n of the slices/adds (same adds, same order)
A2A_REDUCE = False


def _a2a_sum(y, n, cluster_axis):
    """y [B, 1, S, C] with B % n == 0. The all_to_all splits out_dim 0 over the devices and stacks the
    sources along in_dim 1: [B / n, n, S, C], block i = source device i's partial of this device's batch."""
    shape = list(y.shape)
    g = ttnn.experimental.all_to_all_async_generic(
        y, in_dim=1, out_dim=0, cluster_axis=cluster_axis, topology=ttnn.Topology.Linear
    )
    part = [shape[0] // n, 1] + shape[2:]
    out = None
    for i in range(n):
        p = ttnn.slice(g, [0, i, 0, 0], [part[0], i + 1] + shape[2:])
        out = p if out is None else ttnn.add(out, p)
    ttnn.deallocate(g)
    return ttnn.all_gather(out, dim=0, cluster_axis=cluster_axis, num_links=1, topology=ttnn.Topology.Linear)


def exact_all_reduce(y, device, cluster_axis=1):
    """Sum over the TP axis without rounding: gather the float32 partials (bit-exact data movement) and
    add them in float32. ttnn.all_reduce rounds float32 partials at bf16 level (measured ~7e-3 abs on
    O(3) sums on this T3K), which a 32-block fp32 residual stream accumulates.

    With EXACT_REDUCE_L1_BYTES set, a tensor whose gather would not fit is reduced in leading-dim chunks
    whose gather + slices + adds stay in L1 (the same adds in the same order; only the final sum of each
    chunk is written back to DRAM)."""
    n = mesh_shape(device)[cluster_axis]
    if n == 1:
        return y
    shape = list(y.shape)
    if A2A_REDUCE and len(shape) == 4 and shape[1] == 1 and shape[0] % n == 0:
        return _a2a_sum(y, n, cluster_axis)
    per_item = n * math.prod(shape[1:]) * 4
    if not EXACT_REDUCE_L1_BYTES or len(shape) < 2 or per_item > EXACT_REDUCE_L1_BYTES:
        return _gather_sum(y, n, cluster_axis)
    c = max(d for d in range(1, shape[0] + 1) if shape[0] % d == 0 and d * per_item <= EXACT_REDUCE_L1_BYTES)
    outs = []
    for b0 in range(0, shape[0], c):
        yc = ttnn.slice(y, [b0] + [0] * (len(shape) - 1), [b0 + c] + shape[1:])
        s = _gather_sum(yc, n, cluster_axis, mem=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(yc)
        outs.append(ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG))
        ttnn.deallocate(s)
    out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)
    return out


# optional replacement for the 2-limb split of a float32 tensor: a callable x32 -> (hi, lo), e.g. a
# fused kernel; None: typecast / typecast / subtract / typecast
SPLIT_FN = None


def split_bf16(x, limbs=2):
    """float32 -> `limbs` bf16 parts summing to x (2 limbs ~16 mantissa bits, 3 limbs ~24 = float32).
    A matmul consumes its inputs as bf16; the limbs keep every product exact against bf16 weights."""
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    if limbs == 2 and SPLIT_FN is not None and x32.layout == ttnn.TILE_LAYOUT and x32.is_sharded() is False:
        return tuple(SPLIT_FN(x32))
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


def _patterns(exact):
    """exact=True -> all LANE_PATTERNS (median of three); a tuple of pattern names -> just those."""
    return tuple(exact) if isinstance(exact, (tuple, list)) else LANE_PATTERNS


def _vote(ests):
    return ests[0] if len(ests) == 1 else _median3(*ests)


# small exact-lane products keep their lane copies, products and running sums in L1 (bytes per float32
# product; 0: everything in DRAM)
EXACT_SUM_L1_BYTES = 0


def _exact_sum(mm, parts, patterns=LANE_PATTERNS, out_bytes=0):
    """median over `patterns` of sum_{part, lane} mm(lane(part)). mm(lane, memory_config) -> product.
    With EXACT_SUM_L1_BYTES >= out_bytes the lane copies, products and sums live in L1 (one lane copy
    at a time) and only the voted result goes to DRAM."""
    l1 = bool(EXACT_SUM_L1_BYTES) and 0 < out_bytes <= EXACT_SUM_L1_BYTES
    mem = ttnn.L1_MEMORY_CONFIG if l1 else None
    kw = {"memory_config": mem} if l1 else {}
    ests = []
    for pattern in patterns:
        y = None
        for part in parts:
            for m in _lane_masks(part.device(), part.shape[-1], part.dtype, pattern):
                lane = ttnn.multiply(part, m, **kw)
                t = mm(lane, mem)
                ttnn.deallocate(lane)
                y = t if y is None else ttnn.add(y, t, **kw)
        ests.append(y)
    out = _vote(ests)
    return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) if l1 else out


# projections of one activation share its bf16 limb split (MLP gate / up)
SHARE_SPLIT = False
# core grid for the non-exact split_linear products (None: ttnn.linear picks its own)
LINEAR_CORE_GRID = None
# core grid for split_matmul's exact-lane batched products, each on a full-K batch-parallel config
# (None: ttnn picks its own; a core_grid alone gets in0_block_w=1, whose fp32 partial reloads break
# the exact-lane sum: e2e PCC 0.9395)
EXACT_CORE_GRID = None
# core grid for split_linear's exact-lane products (None: ttnn.linear picks its own)
EXACT_LINEAR_CORE_GRID = None
# hand-shaped 2-D multicast config for those products on LINEAR_CORE_GRID (False: core_grid only;
# ttnn then picks per_core_N=19 for the LM MLP, i.e. 1x1 subblocks and 2-row output blocks that
# re-stream the weight per block)
LINEAR_BLOCK = False
# True: M over the grid columns (x), N over the rows (y)
LINEAR_BLOCK_TRANSPOSE = False
# L1 tiles per core for the float32 output block
_OUT_BLOCK_TILES = 160


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _block_config(x, w):
    """2-D multicast with the output block / subblock picked per shape: every core re-receives its in0
    rows once per output-block column pass and its in1 columns once per output-block row pass, so the
    pick minimises that per-core traffic divided by sqrt(subblock tiles) (subblocks up to 4 tiles with
    the fp32 dest), with the output block within _OUT_BLOCK_TILES. Per-core M/N are rounded up to even
    tile counts so a 2-wide subblock always exists (ttnn's own pick for the LM MLP, per_core_N=19, is
    stuck at 1x1). None when the shapes are not tile aligned or M is a single tile row."""
    grid = LINEAR_CORE_GRID
    xs, ws = [int(d) for d in x.shape], [int(d) for d in w.shape]
    m = math.prod(xs[:-1])
    if m % TILE or ws[-2] % TILE or ws[-1] % TILE or m // TILE < 2:
        return None
    m_t, k_t, n_t = m // TILE, ws[-2] // TILE, ws[-1] // TILE
    rows, cols = (grid.x, grid.y) if LINEAR_BLOCK_TRANSPOSE else (grid.y, grid.x)
    even = lambda v: v + (v % 2)  # noqa: E731
    per_m, per_n = even(-(-m_t // rows)), even(-(-n_t // cols))
    best = None
    for out_h in _divisors(per_m):
        for out_w in _divisors(per_n):
            if out_h * out_w > _OUT_BLOCK_TILES:
                continue
            sub = max(
                ((sh, sw) for sh in _divisors(out_h) for sw in _divisors(out_w) if sh * sw <= 4),
                key=lambda t: (t[0] * t[1], t[1]),
            )
            traffic = per_m * (per_n // out_w) + per_n * (per_m // out_h)
            score = traffic / math.sqrt(sub[0] * sub[1])
            if best is None or score < best[0]:
                best = (score, out_h, out_w, sub)
    _, out_h, out_w, (sub_h, sub_w) = best
    # in0 / in1 K blocks double-buffered next to the output block
    in0_w = max(d for d in _divisors(k_t) if d <= 8 and 2 * (out_h + out_w) * d * 2048 <= 512 * 1024)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=out_h,
        out_block_w=out_w,
        per_core_M=per_m,
        per_core_N=per_n,
        transpose_mcast=LINEAR_BLOCK_TRANSPOSE,
        fused_activation=None,
        fuse_batch=True,
    )


def split_linear(x, w, bias=None, compute_kernel_config=None, exact=True, limbs=2, parts=None):
    """float32 x @ bf16 w (+ bias) -> float32. x is carried as `limbs` bf16 parts (exact products). With
    exact=True the K reduction is the exact lane sum (see above); otherwise one matmul per part.
    parts: split_bf16(x, limbs) when several projections share x (split once, not once per projection)."""
    cfg = precise_config()
    parts = split_bf16(x, limbs) if parts is None else parts

    def mm(p, mem=None):
        kw = {} if mem is None else {"memory_config": mem}
        return ttnn.linear(p, w, compute_kernel_config=cfg, dtype=ttnn.float32, core_grid=EXACT_LINEAR_CORE_GRID, **kw)

    if exact:
        out_bytes = 4 * math.prod([int(d) for d in x.padded_shape][:-1]) * int(w.padded_shape[-1])
        y = _exact_sum(mm, parts, _patterns(exact), out_bytes=out_bytes)
    else:
        pc = _block_config(x, w) if LINEAR_BLOCK and LINEAR_CORE_GRID is not None else None
        kw = dict(program_config=pc) if pc is not None else dict(core_grid=LINEAR_CORE_GRID)
        y = None
        for part in parts:
            t = ttnn.linear(part, w, compute_kernel_config=cfg, dtype=ttnn.float32, **kw)
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


def _bmm_full_k_config(a, b, transpose_b):
    """Batch-parallel (one output block per core) config for an exact-lane batched product, with the
    WHOLE K in one in0 block: each output tile is formed in the fp32 dest in one pass, never spilled
    and reloaded between K blocks. None when the shapes are not tile aligned or the blocks overflow L1."""
    a_s, b_s = [int(d) for d in a.padded_shape], [int(d) for d in b.padded_shape]
    k, n = (b_s[-1], b_s[-2]) if transpose_b else (b_s[-2], b_s[-1])
    m = a_s[-2]
    if m % TILE or k % TILE or n % TILE:
        return None
    m_t, k_t, n_t = m // TILE, k // TILE, n // TILE
    # double-buffered bf16 in0/in1 blocks + the float32 output block must fit one core's L1
    if (2 * m_t * k_t + 2 * k_t * n_t) * 2048 + m_t * n_t * 4096 > 1024 * 1024:
        return None
    sub_w = max(d for d in range(1, min(n_t, 4) + 1) if n_t % d == 0)
    sub_h = max(d for d in range(1, min(m_t, 4 // sub_w) + 1) if m_t % d == 0)
    grid = EXACT_CORE_GRID
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k_t,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_t,
        per_core_N=n_t,
    )


def split_matmul(a, b, transpose_b=False, compute_kernel_config=None, exact=True, limbs=2):
    """float32 a @ float32 b over bf16 limbs of both operands, keeping the limb products whose order
    (i + j) is below `limbs` (2 limbs: ah.bh + ah.bl + al.bh; 3 limbs: six terms, ~float32). With
    exact=True each term's K reduction is the exact lane sum (the lanes mask a's last dim)."""
    cfg = precise_config()
    pa_all = split_bf16(a, limbs)
    pb_all = split_bf16(b, limbs)
    terms = [(pa_all[i], pb_all[j]) for i in range(limbs) for j in range(limbs) if i + j < limbs]
    pc = _bmm_full_k_config(pa_all[0], pb_all[0], transpose_b) if exact and EXACT_CORE_GRID is not None else None
    mm = lambda p, q: ttnn.matmul(  # noqa: E731
        p, q, transpose_b=transpose_b, compute_kernel_config=cfg, dtype=ttnn.float32, program_config=pc
    )
    if not exact:
        y = None
        for pa, pb in terms:
            t = mm(pa, pb)
            y = t if y is None else ttnn.add(y, t)
        return y
    ests = []
    for pattern in _patterns(exact):
        y = None
        for pa, pb in terms:
            for lane in _lanes(pa, pattern):
                t = mm(lane, pb)
                y = t if y is None else ttnn.add(y, t)
        ests.append(y)
    return _vote(ests)


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
        ex = getattr(self, "exact", True)
        qkv = split_linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=cfg, exact=ex, limbs=L)

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
        scores = split_matmul(q, k, transpose_b=True, compute_kernel_config=cfg, exact=ex, limbs=L)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), mask)
        # explicit float32 softmax (the fused op mis-scales some masked window rows: one token of block
        # 12 measured 6.7% off with exact inputs)
        mx = ttnn.max(scores, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(scores, mx))
        probs = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=cfg))
        o = split_matmul(probs, v, compute_kernel_config=cfg, exact=ex, limbs=L)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (n, 1, s_pad, loc))
        out = split_linear(o, self.wproj, compute_kernel_config=cfg, exact=ex, limbs=L)
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
