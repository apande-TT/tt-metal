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


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        measure(device)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
