"""tt-lang attention-output concat-heads on the short-prefill shape -- AUTHORED and MEASURED.

The tt-lang rung for `NLPConcatHeadsDeviceOperation` (prefill `nlp_concat_heads`), and the exact
mirror of tt/ttl_create_qkv_heads.py: that kernel scatters a fused QKV row into head-major views,
this one gathers head-major attention output back into a single row.

WHY a kernel. Same reason as the head split, and the knob rungs proved it: the stock op's
interleaved program factory sizes cores from

    num_blocks = batch * seq_len / TILE_HEIGHT

-- heads (dim 1) never enter the split -- so at batch 1 / seq_len 128 it runs on 4 of the P150's
110 cores with 32 heads of independent work idle. Its sharded factory would adopt the input's own
shard grid, but feeding it a head-sharded input is measured-UNSAFE here: a ragged core set scrambles
the output (25.6% top-1, because the factory mixes corerange_to_cores with grid_to_cores) and a
rectangular one hangs, at both one and two heads per core. So the parallelisation is baked into the
op and the only way past it is a kernel that decomposes the work differently.

The concat is a pure tile gather -- output tile (seq s, col h*HD_T + c) is input tile
(head h, seq s, col c) -- so every output tile is independent. Mapped to an 8x4 grid: the y axis
takes the seq tile, the x axis takes a group of 4 heads. Division-free, so all indexing stays affine
in the node coords.

Run directly to reproduce the numbers in TTL_CONCAT_*.
"""
import time

import torch

import ttnn
import ttl

TILE = 32
# GRID RUNG. The work has THREE dimensions -- head (32) x seq tile (4) x dim tile (4) -- but
# `ttl.node` is 2-D, so a division-free mapping can carry only two of them and every division-free
# choice lands on 32 of the P150's 110 cores (head groups must divide 32 and fit in <=11 columns,
# i.e. 8; the other axis is then seq(4) or dim(4)). Getting past 32 therefore needs ONE coordinate
# to carry two work dimensions, which needs a constant divide on the node coord. So: y carries
# (seq tile, dim half) as cy = dim_half * SEQ_T + s, giving 8 rows and 64 cores at 8 tiles each.
# If the compiler will not lower `//`/`%` on a node coord this falls back to the 4-row form.
SEQ_LEN = 128
N_HEADS, HEAD_DIM = 32, 128
HD_T = HEAD_DIM // TILE
SEQ_T = SEQ_LEN // TILE
DIM_HALVES = 2
HALF_T = HD_T // DIM_HALVES

GRID_X, GRID_Y = 8, SEQ_T * DIM_HALVES
HEADS_PER_COL = N_HEADS // GRID_X


@ttl.operation(grid=(GRID_X, GRID_Y))
def ttl_concat_heads(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
    """x [H*S, hd] -> y [S, H*hd], both 2-D tile views.

    A [1, H, S, hd] tiled tensor has the tile ordering of a 2-D [H*S, hd] one, and a
    [1, 1, S, H*hd] tensor that of an [S, H*hd] one, so the caller reshapes rather than copies.
    """
    seq_tiles = y.shape[0] // TILE
    in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    per_core_tiles = HEADS_PER_COL * HALF_T

    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        st = cy % SEQ_T
        c0 = (cy // SEQ_T) * HALF_T
        for hh in range(HEADS_PER_COL):
            for c in range(HALF_T):
                with in_dfb.reserve() as blk:
                    ttl.copy(x[(cx * HEADS_PER_COL + hh) * seq_tiles + st, c0 + c], blk).wait()

    # TTNN interop requires exactly one compute + two data-movement kernels (a core has two NOCs),
    # so the gather runs reader -> compute passthrough -> writer rather than DM straight to DM.
    @ttl.compute()
    def passthrough():
        for _ in range(per_core_tiles):
            with in_dfb.wait() as ib:
                with out_dfb.reserve() as ob:
                    ob.store(ib)

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        st = cy % SEQ_T
        c0 = (cy // SEQ_T) * HALF_T
        for hh in range(HEADS_PER_COL):
            for c in range(HALF_T):
                with out_dfb.wait() as blk:
                    ttl.copy(blk, y[st, (cx * HEADS_PER_COL + hh) * HD_T + c0 + c]).wait()


# GRID RUNG for the SINGLE-SEQ-TILE prefill (seq_len == 32) -- the same lever, and the same latent
# bug, as the head-split kernel's 1T twin. The guard below was pinned to SEQ_LEN == 128, so a 32-token
# padded prompt fell straight through to the stock nlp_concat_heads, which assigns one work unit per
# input row-tile and therefore ran on a SINGLE core: the grid=tiny tag on a DISPATCH-bound op.
#
# At SEQ_T == 1 the work is head(32) x dim tile(4) with no seq extent, which FREES the y axis: instead
# of splitting a head's dim tiles into halves it carries ONE dim tile per row, so DIM_HALVES rises to
# HD_T (4) and HALF_T falls to 1. That is 8 x 4 = 32 cores at HEADS_PER_COL * 1 = 4 tiles each, and
# 32 x 4 = 128 tiles covers all 32 heads x 1 seq tile x 4 dim tiles exactly. With SEQ_T == 1 the seq
# index is constant 0 and c0 collapses to cy, so this variant needs no divide on the node coord at all.
SEQ_LEN_1T = TILE
SEQ_T_1T = SEQ_LEN_1T // TILE  # == 1
DIM_HALVES_1T = HD_T  # one dim tile per y row, the whole point of this variant
HALF_T_1T = HD_T // DIM_HALVES_1T  # == 1
GRID_Y_1T = SEQ_T_1T * DIM_HALVES_1T  # == 4


@ttl.operation(grid=(GRID_X, GRID_Y_1T))
def ttl_concat_heads_1t(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
    """Single-seq-tile twin of ``ttl_concat_heads``: x [H*32, hd] -> y [32, H*hd]."""
    seq_tiles = y.shape[0] // TILE
    in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    per_core_tiles = HEADS_PER_COL * HALF_T_1T

    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        for hh in range(HEADS_PER_COL):
            with in_dfb.reserve() as blk:
                ttl.copy(x[(cx * HEADS_PER_COL + hh) * seq_tiles, cy], blk).wait()

    @ttl.compute()
    def passthrough():
        for _ in range(per_core_tiles):
            with in_dfb.wait() as ib:
                with out_dfb.reserve() as ob:
                    ob.store(ib)

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        for hh in range(HEADS_PER_COL):
            with out_dfb.wait() as blk:
                ttl.copy(blk, y[0, (cx * HEADS_PER_COL + hh) * HD_T + cy]).wait()


# seq_len -> the specialised concat op for it.
_CONCAT_OPS = {SEQ_LEN: ttl_concat_heads, SEQ_LEN_1T: ttl_concat_heads_1t}


def supports(attn_output, num_heads, head_dim) -> bool:
    """Is this call one of the shapes a kernel is specialised for?"""
    return (
        num_heads == N_HEADS
        and head_dim == HEAD_DIM
        and len(attn_output.shape) == 4
        and attn_output.shape[0] == 1
        and attn_output.shape[1] == N_HEADS
        and int(attn_output.shape[2]) in _CONCAT_OPS
        and attn_output.shape[3] == HEAD_DIM
    )


def concat_heads_ttl(attn_output, memory_config):
    """Drop-in for ``ttnn.experimental.nlp_concat_heads`` at the specialised shape."""
    seq_len = int(attn_output.shape[2])
    device = attn_output.device()
    dtype = attn_output.dtype

    x2 = ttnn.reshape(attn_output, [N_HEADS * seq_len, HEAD_DIM])
    y2 = ttnn.empty([seq_len, N_HEADS * HEAD_DIM], dtype, ttnn.TILE_LAYOUT, device, memory_config)
    _CONCAT_OPS[seq_len](x2, y2)
    return ttnn.reshape(y2, [1, 1, seq_len, N_HEADS * HEAD_DIM])


def measure(device):
    tx = torch.randn(1, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    x4 = ttnn.from_torch(tx, **kw)
    x2 = ttnn.reshape(x4, [N_HEADS * SEQ_LEN, HEAD_DIM])
    y2 = ttnn.from_torch(torch.zeros(SEQ_LEN, N_HEADS * HEAD_DIM, dtype=torch.bfloat16), **kw)

    ttl_concat_heads(x2, y2)

    golden = tx[0].permute(1, 0, 2).reshape(SEQ_LEN, N_HEADS * HEAD_DIM)
    got = ttnn.to_torch(y2).float()
    pcc = torch.corrcoef(torch.stack([golden.float().flatten(), got.flatten()]))[0, 1].item()
    print("TTL_CONCAT_PCC=%.6f" % pcc)

    def run_ttl():
        ttl_concat_heads(x2, y2)

    def run_ttnn():
        ttnn.deallocate(ttnn.experimental.nlp_concat_heads(x4, memory_config=ttnn.L1_MEMORY_CONFIG))

    for label, fn in (("TTL_CONCAT_MS", run_ttl), ("TTNN_CONCAT_MS", run_ttnn)):
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
