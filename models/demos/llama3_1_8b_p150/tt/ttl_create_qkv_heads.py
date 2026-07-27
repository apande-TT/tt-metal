"""tt-lang QKV head split on the short-prefill shape -- AUTHORED and MEASURED.

The tt-lang rung for `NlpCreateHeadsDeviceOperation` (prefill `nlp_create_qkv_heads`).

WHY a kernel is the right thing to try here. The stock op is pure data movement, and its
interleaved program factory sizes cores from

    num_blocks = batch * seq_len / TILE_HEIGHT

-- one work unit per INPUT ROW-TILE. At batch 1 / seq_len 128 that is four blocks, so the op
runs on 4 of the P150's 110 cores and no program_config can widen it (the only other factory
is the sharded one, whose output shard spec is hard-coded {TILE_HEIGHT, head_dim} and so only
holds at seq_len == 32). The knob rungs are therefore closed by construction, which is exactly
the case GUIDELINES/11 names for a hand kernel: the parallelisation is baked into the op, and
the only way past it is to write one whose work decomposition differs.

This kernel parallelises over (seq_tile x head) instead of over seq_tile alone. The head split
is a pure tile gather -- output tile (head h, seq tile s, col c) is input tile [s, h*4 + c] --
so every output tile is independent and the decomposition is free to be as wide as the tile
count. On the 128-token shape that is 4 seq tiles x (32 Q + 8 K + 8 V) heads, mapped to a 4x8
grid: the row axis takes the seq tile, the column axis takes a group of 4 Q heads plus one K
and one V head. Division-free, so all indexing stays affine in the node coords.

Run directly to reproduce the numbers in TTL_HEADS_*.
"""
import time

import torch

import ttnn
import ttl

TILE = 32
# seq_len 128 -> 4 seq tiles, one per grid row; 32 Q heads / 8 grid columns -> 4 Q heads per
# column, plus one K and one V head each (8 KV heads == 8 columns).
GRID_Y, GRID_X = 4, 8
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 32, 8, 128
SEQ_LEN = 128

HD_T = HEAD_DIM // TILE  # tiles spanned by one head
Q_PER_COL = N_Q_HEADS // GRID_X


@ttl.operation(grid=(GRID_X, GRID_Y))
def ttl_create_heads(x: ttnn.Tensor, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> None:
    """x [S, (nq + 2*nkv)*hd] -> q [nq*S, hd], k/v [nkv*S, hd], all 2-D tile views.

    A [1, H, S, hd] tiled tensor has exactly the tile ordering of a 2-D [H*S, hd] one, so the
    caller reshapes rather than copying.
    """
    seq_tiles = x.shape[0] // TILE
    # TTNN interop wants exactly one compute + two data-movement kernels (a core has two NOCs),
    # so the gather runs reader -> compute passthrough -> writer rather than DM straight to DM.
    in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), block_count=2)

    # Tiles each core moves: its Q heads, then one K head, then one V head -- all HD_T wide.
    per_core_tiles = (Q_PER_COL + 2) * HD_T

    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        for hh in range(Q_PER_COL):
            for c in range(HD_T):
                with in_dfb.reserve() as blk:
                    ttl.copy(x[cy, (cx * Q_PER_COL + hh) * HD_T + c], blk).wait()
        for c in range(HD_T):
            with in_dfb.reserve() as blk:
                ttl.copy(x[cy, (N_Q_HEADS + cx) * HD_T + c], blk).wait()
        for c in range(HD_T):
            with in_dfb.reserve() as blk:
                ttl.copy(x[cy, (N_Q_HEADS + N_KV_HEADS + cx) * HD_T + c], blk).wait()

    @ttl.compute()
    def passthrough():
        for _ in range(per_core_tiles):
            with in_dfb.wait() as ib:
                with out_dfb.reserve() as ob:
                    ob.store(ib)

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        for hh in range(Q_PER_COL):
            for c in range(HD_T):
                with out_dfb.wait() as blk:
                    ttl.copy(blk, q[(cx * Q_PER_COL + hh) * seq_tiles + cy, c]).wait()
        for c in range(HD_T):
            with out_dfb.wait() as blk:
                ttl.copy(blk, k[cx * seq_tiles + cy, c]).wait()
        for c in range(HD_T):
            with out_dfb.wait() as blk:
                ttl.copy(blk, v[cx * seq_tiles + cy, c]).wait()


@ttl.operation(grid=(GRID_X, GRID_Y))
def ttl_create_heads_rope(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans: ttnn.Tensor,
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
) -> None:
    """Head split WITH the rotary embedding folded in -- the structural rung for the prefill rope.

    The prefill rope is DISPATCH bound, not compute bound: 736 launches of 9.55 us each against a
    1.21 us roofline, because a [32, 1, 128, 128] rotation is only ~8 tiles per core. No knob
    removes a fixed launch cost, so the lever is to remove the LAUNCHES: the head split already
    streams every Q/K/V tile through this kernel's compute stage, and the rotation is entirely
    TILE-LOCAL (the stock kernel computes `x*cos + (x @ trans_mat)*sin` with a single 32x32
    trans_mat tile applied to each tile independently, and cos/sin are head-broadcast [1, 1, S, D]
    so a tile only needs the cos/sin tile at its own (seq_tile, dim_tile)). So the rotation can be
    done in the pass-through slot the split already pays for, and the two rope ops disappear:
    3 dispatches per layer (split, rope Q, rope K) collapse to 1.

    V is written unrotated. Core (cx, cy) owns seq tile cy, so it needs exactly cos/sin[cy, 0:HD_T].
    """
    seq_tiles = x.shape[0] // TILE
    in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    cos_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), block_count=2)
    sin_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), block_count=2)
    trans_dfb = ttl.make_dataflow_buffer_like(trans, shape=(1, 1), block_count=2)
    # The rotate is staged through its own L1 buffer rather than written as one expression. The
    # stock kernel does the same (rotated_in_interm_cb) because the matmul and the eltwise chain
    # want different dst/unpacker states; folding them into a single `x*cos + (x @ trans)*sin`
    # store measured PCC 0.850 against 0.986 for the op it replaces.
    rot_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(q, shape=(1, 1), block_count=2)

    # Per core: this column's Q heads plus its one K head are rotated; its V head is pass-through.
    pass_tiles = HD_T

    # DIM-TILE OUTER, HEAD INNER. A core's cos/sin tile depends only on (seq_tile, dim_tile), and
    # its seq_tile is fixed by cy -- so all Q_PER_COL+1 rotated heads at a given dim tile share ONE
    # cos tile, one sin tile and the one trans tile. Reading them per (head, dim_tile) instead cost
    # 60 tile reads per core against 12; a cos block stays live for the whole `with` body, so the
    # head loop nests INSIDE it and each constant is fetched once per dim tile.
    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        for c in range(HD_T):
            with cos_dfb.reserve() as cb, sin_dfb.reserve() as sb, trans_dfb.reserve() as tb:
                t1 = ttl.copy(cos[cy, c], cb)
                t2 = ttl.copy(sin[cy, c], sb)
                t3 = ttl.copy(trans[0, 0], tb)
                t1.wait()
                t2.wait()
                t3.wait()
            for hh in range(Q_PER_COL):
                with in_dfb.reserve() as blk:
                    ttl.copy(x[cy, (cx * Q_PER_COL + hh) * HD_T + c], blk).wait()
            with in_dfb.reserve() as blk:
                ttl.copy(x[cy, (N_Q_HEADS + cx) * HD_T + c], blk).wait()
        for c in range(HD_T):
            with in_dfb.reserve() as blk:
                ttl.copy(x[cy, (N_Q_HEADS + N_KV_HEADS + cx) * HD_T + c], blk).wait()

    @ttl.compute()
    def rotate():
        for _ in range(HD_T):
            with cos_dfb.wait() as cb, sin_dfb.wait() as sb, trans_dfb.wait() as tb:
                for _ in range(Q_PER_COL + 1):
                    with in_dfb.wait() as ib:
                        with rot_dfb.reserve() as rb:
                            rb.store(ib @ tb)  # rotated = x @ trans_mat (tile-local pairwise swap)
                        with rot_dfb.wait() as rr:
                            with out_dfb.reserve() as ob:
                                ob.store(ib * cb + rr * sb)
        for _ in range(pass_tiles):
            with in_dfb.wait() as ib:
                with out_dfb.reserve() as ob:
                    ob.store(ib)

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        for c in range(HD_T):
            for hh in range(Q_PER_COL):
                with out_dfb.wait() as blk:
                    ttl.copy(blk, q[(cx * Q_PER_COL + hh) * seq_tiles + cy, c]).wait()
            with out_dfb.wait() as blk:
                ttl.copy(blk, k[cx * seq_tiles + cy, c]).wait()
        for c in range(HD_T):
            with out_dfb.wait() as blk:
                ttl.copy(blk, v[cx * seq_tiles + cy, c]).wait()


def supports(xqkv_fused, num_heads, num_kv_heads, head_dim) -> bool:
    """Is this call the exact shape the kernel is specialised for?

    The grid mapping is division-free only because the extents line up: 4 seq tiles on the y
    axis, 8 grid columns each owning 4 Q heads plus one K and one V head. Anything else falls
    back to the stock op.
    """
    return (
        num_heads == N_Q_HEADS
        and num_kv_heads == N_KV_HEADS
        and head_dim == HEAD_DIM
        and len(xqkv_fused.shape) == 4
        and xqkv_fused.shape[0] == 1
        and xqkv_fused.shape[1] == 1
        and xqkv_fused.shape[-2] == SEQ_LEN
        and xqkv_fused.shape[-1] == (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
    )


def create_qkv_heads_ttl(xqkv_fused, memory_config):
    """Drop-in for ``ttnn.experimental.nlp_create_qkv_heads`` at the specialised shape.

    A [1, H, S, hd] tiled tensor and a 2-D [H*S, hd] one have identical tile ordering, so the
    reshapes on both sides are metadata-only -- the kernel writes head-major rows directly.
    """
    seq_len = int(xqkv_fused.shape[-2])
    fused_width = int(xqkv_fused.shape[-1])
    device = xqkv_fused.device()
    dtype = xqkv_fused.dtype

    x2 = ttnn.reshape(xqkv_fused, [seq_len, fused_width])

    def _out(heads):
        return ttnn.empty([heads * seq_len, HEAD_DIM], dtype, ttnn.TILE_LAYOUT, device, memory_config)

    q, k, v = _out(N_Q_HEADS), _out(N_KV_HEADS), _out(N_KV_HEADS)
    ttl_create_heads(x2, q, k, v)

    return (
        ttnn.reshape(q, [1, N_Q_HEADS, seq_len, HEAD_DIM]),
        ttnn.reshape(k, [1, N_KV_HEADS, seq_len, HEAD_DIM]),
        ttnn.reshape(v, [1, N_KV_HEADS, seq_len, HEAD_DIM]),
    )


def supports_rope(xqkv_fused, num_heads, num_kv_heads, head_dim, cos, sin, trans) -> bool:
    """Can the rope be folded into the split for THIS call?

    Everything `supports` requires, plus the three things that make the rotation tile-local:
    cos/sin must be HEAD-BROADCAST ([1, 1, S, head_dim] with dim 1 == 1 -- a per-head frequency
    table would need a different cos tile per head), they must cover the seq tiles the kernel
    walks, and trans_mat must be the single 32x32 tile the stock kernel also assumes. All four
    tensors must be bf16, since the fused kernel does the rotate in the split's dtype.
    """
    if not supports(xqkv_fused, num_heads, num_kv_heads, head_dim):
        return False
    if cos is None or sin is None or trans is None:
        return False
    for t in (xqkv_fused, cos, sin, trans):
        if t.dtype != ttnn.bfloat16:
            return False
    for t in (cos, sin):
        if len(t.shape) != 4 or t.shape[0] != 1 or t.shape[1] != 1:
            return False
        if int(t.shape[-1]) != HEAD_DIM or int(t.shape[-2]) < SEQ_LEN:
            return False
    return len(trans.shape) == 4 and tuple(int(d) for d in trans.shape) == (1, 1, TILE, TILE)


def create_qkv_heads_rope_ttl(xqkv_fused, cos, sin, trans, memory_config):
    """Drop-in for `nlp_create_qkv_heads` + `rotary_embedding_llama(q)` + `...(k)` in one dispatch.

    Returns q/k already rotated and v untouched, in the same [1, H, S, hd] shape/dtype/memory
    config the three-op chain produced, so nothing downstream changes.
    """
    seq_len = int(xqkv_fused.shape[-2])
    fused_width = int(xqkv_fused.shape[-1])
    device = xqkv_fused.device()
    dtype = xqkv_fused.dtype

    x2 = ttnn.reshape(xqkv_fused, [seq_len, fused_width])
    cos2 = ttnn.reshape(cos, [int(cos.shape[-2]), HEAD_DIM])
    sin2 = ttnn.reshape(sin, [int(sin.shape[-2]), HEAD_DIM])
    trans2 = ttnn.reshape(trans, [TILE, TILE])

    def _out(heads):
        return ttnn.empty([heads * seq_len, HEAD_DIM], dtype, ttnn.TILE_LAYOUT, device, memory_config)

    q, k, v = _out(N_Q_HEADS), _out(N_KV_HEADS), _out(N_KV_HEADS)
    ttl_create_heads_rope(x2, cos2, sin2, trans2, q, k, v)

    return (
        ttnn.reshape(q, [1, N_Q_HEADS, seq_len, HEAD_DIM]),
        ttnn.reshape(k, [1, N_KV_HEADS, seq_len, HEAD_DIM]),
        ttnn.reshape(v, [1, N_KV_HEADS, seq_len, HEAD_DIM]),
    )


def measure(device):
    fused_width = (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
    tx = torch.randn(SEQ_LEN, fused_width, dtype=torch.bfloat16)

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    x = ttnn.from_torch(tx, **kw)
    q = ttnn.from_torch(torch.zeros(N_Q_HEADS * SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16), **kw)
    k = ttnn.from_torch(torch.zeros(N_KV_HEADS * SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16), **kw)
    v = ttnn.from_torch(torch.zeros(N_KV_HEADS * SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16), **kw)

    ttl_create_heads(x, q, k, v)

    # Golden: the same split torch-side, laid out head-major like the kernel writes it.
    gq = tx[:, : N_Q_HEADS * HEAD_DIM].reshape(SEQ_LEN, N_Q_HEADS, HEAD_DIM).permute(1, 0, 2).reshape(-1, HEAD_DIM)
    got = ttnn.to_torch(q).float()
    pcc = torch.corrcoef(torch.stack([gq.float().flatten(), got.flatten()]))[0, 1].item()
    print("TTL_HEADS_PCC=%.6f" % pcc)

    x4 = ttnn.reshape(x, [1, 1, SEQ_LEN, fused_width])

    def run_ttl():
        ttl_create_heads(x, q, k, v)

    def run_ttnn():
        a, b, c = ttnn.experimental.nlp_create_qkv_heads(
            x4,
            num_heads=N_Q_HEADS,
            num_kv_heads=N_KV_HEADS,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(c)

    for label, fn in (("TTL_HEADS_MS", run_ttl), ("TTNN_HEADS_MS", run_ttnn)):
        fn()
        ttnn.synchronize_device(device)
        t0 = time.monotonic()
        for _ in range(20):
            fn()
        ttnn.synchronize_device(device)
        print("%s=%.4f" % (label, (time.monotonic() - t0) * 1000.0 / 20))


def measure_rope(device):
    """Correctness + speed of the fused split+rope against the 3-op chain it replaces.

    Golden is ttnn's own `nlp_create_qkv_heads` + `rotary_embedding_llama` on the same tensors --
    that chain IS the contract, so anything else would be measuring the wrong thing.
    """
    fused_width = (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
    tx = torch.randn(SEQ_LEN, fused_width, dtype=torch.bfloat16) * 0.5
    tcos = torch.cos(torch.randn(SEQ_LEN, HEAD_DIM, dtype=torch.float32)).to(torch.bfloat16)
    tsin = torch.sin(torch.randn(SEQ_LEN, HEAD_DIM, dtype=torch.float32)).to(torch.bfloat16)
    ttrans = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    for i in range(TILE // 2):  # the llama rope transformation tile: pairwise swap with a sign flip
        ttrans[2 * i + 1, 2 * i] = 1.0
        ttrans[2 * i, 2 * i + 1] = -1.0

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    x = ttnn.from_torch(tx, **kw)
    cos = ttnn.from_torch(tcos, **kw)
    sin = ttnn.from_torch(tsin, **kw)
    trans = ttnn.from_torch(ttrans, **kw)
    x4 = ttnn.reshape(x, [1, 1, SEQ_LEN, fused_width])
    cos4 = ttnn.reshape(cos, [1, 1, SEQ_LEN, HEAD_DIM])
    sin4 = ttnn.reshape(sin, [1, 1, SEQ_LEN, HEAD_DIM])
    trans4 = ttnn.reshape(trans, [1, 1, TILE, TILE])

    def stock():
        a, b, c = ttnn.experimental.nlp_create_qkv_heads(
            x4,
            num_heads=N_Q_HEADS,
            num_kv_heads=N_KV_HEADS,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ar = ttnn.experimental.rotary_embedding_llama(
            ttnn.reshape(a, [N_Q_HEADS, 1, SEQ_LEN, HEAD_DIM]), cos4, sin4, trans4, is_decode_mode=False
        )
        br = ttnn.experimental.rotary_embedding_llama(
            ttnn.reshape(b, [N_KV_HEADS, 1, SEQ_LEN, HEAD_DIM]), cos4, sin4, trans4, is_decode_mode=False
        )
        return ar, br, c

    gq, gk, _ = stock()
    fq, fk, fv = create_qkv_heads_rope_ttl(x4, cos4, sin4, trans4, ttnn.L1_MEMORY_CONFIG)
    for label, ref, got in (("Q", gq, fq), ("K", gk, fk)):
        a = ttnn.to_torch(ref).float().flatten()
        b = ttnn.to_torch(got).float().flatten()
        print("TTL_HEADSROPE_PCC_%s=%.6f" % (label, torch.corrcoef(torch.stack([a, b]))[0, 1].item()))

    def run_fused():
        a, b, c = create_qkv_heads_rope_ttl(x4, cos4, sin4, trans4, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(c)

    def run_stock():
        a, b, c = stock()
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(c)

    for label, fn in (("TTL_HEADSROPE_MS", run_fused), ("TTNN_HEADSROPE_MS", run_stock)):
        fn()
        ttnn.synchronize_device(device)
        t0 = time.monotonic()
        for _ in range(20):
            fn()
        ttnn.synchronize_device(device)
        print("%s=%.4f" % (label, (time.monotonic() - t0) * 1000.0 / 20))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        import sys

        if "--rope" in sys.argv:
            measure_rope(device)
        else:
            measure(device)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
