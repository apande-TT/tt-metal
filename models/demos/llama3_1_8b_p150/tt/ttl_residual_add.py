"""tt-lang residual add on the short-prefill shape -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the tt-lang rung for `BinaryNgDeviceOperation`.

WHICH instance, and why this one. The op code covers five shapes. Only ONE is a same-dtype bf16
add that a ttl kernel can legally replace -- the post-MLP residual add. The others are closed:
  * the post-attention residual add is MIXED dtype (bf16 residual + bf8_b wo output), which the
    kernel's single-dtype dataflow buffers cannot express;
  * both SILU gate multiplies (the largest instances, 8.23 ms and 3.50 ms) are bf8_b, a BLOCK float
    format whose shared-exponent pack the tt-lang packer does not produce correctly on this build --
    measured on this model, having a ttl kernel emit bf8_b drove the e2e gate to a degenerate 28.917
    (a "PCC" above 1), so a kernel is not available for them at any grid.

WORK DECOMPOSITION. [128, 4096] is 4 height tiles x 128 width tiles = 512 independent output tiles
and an add has no reduction, so every tile may go anywhere. `ttl.node` is 2-D, and naively that caps
the grid: a width split must divide 128 and fit 11 columns (so 8), leaving height (4) on y = 32
cores. As with the head-split kernels the way past it is to let ONE coordinate carry two work
dimensions through a constant divide, which the compiler does lower:

    cy = w_half * H_T + h_tile        ->  8 rows, 64 cores, 8 tiles each

MEASURED in the model, against the stock ttnn.add it replaces (352 calls each, same shape/dtype):

    correctness   e2e PCC 0.985099, unchanged bit-for-bit from the stock op
    ttl kernel    4.87 us/call   on 64 cores
    stock ttnn    3.70 us/call   on 110 cores
    whole model   648.17 -> 648.35 ms  (BinaryNg -1.44 ms, GenericOp +1.74 ms)

So it is 1.3x SLOWER and deliberately not on the hot path. The reason is structural and is the same
one that closed this op's grid rung: an interleaved binary_ng gets ALL 110 cores from
split_work_to_cores, while any tt-lang decomposition of a 4 x 128 tile grid tops out at 64 (the
width split needs a divisor of 128 that fits 11 columns, and y can only reach 8 rows before
H_T * W_HALVES exceeds the board's 10). A lone eltwise op moves a fixed number of bytes, so losing
42% of the cores cannot be bought back -- exactly the case GUIDELINES/11 names when it warns that a
single op is usually already at its floor and only a FUSION the op library cannot express wins.

Run directly (`python -m ...tt.ttl_residual_add`) to reproduce the numbers in TTL_RESADD_*.
"""
import time

import torch

import ttnn
import ttl

TILE = 32
SEQ_LEN, DIM = 128, 4096
H_T = SEQ_LEN // TILE  # 4 height tiles
W_T = DIM // TILE  # 128 width tiles

GRID_X = 8  # width groups: must divide W_T and fit the board's 11 columns
W_HALVES = 2
GRID_Y = H_T * W_HALVES  # 8 rows: y carries (width half, height tile)
W_PER_CORE = W_T // (GRID_X * W_HALVES)  # 8 width tiles per core


@ttl.operation(grid=(GRID_X, GRID_Y))
def ttl_residual_add(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
    """y = a + b on [SEQ_LEN, DIM] 2-D tile views.

    A [1, 1, S, D] tiled tensor has the tile ordering of a 2-D [S, D] one, so the caller reshapes
    rather than copying.
    """
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        ht = cy % H_T
        wt0 = (cx * W_HALVES + (cy // H_T)) * W_PER_CORE
        for j in range(W_PER_CORE):
            with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                ta = ttl.copy(a[ht, wt0 + j], ab)
                tb = ttl.copy(b[ht, wt0 + j], bb)
                ta.wait()
                tb.wait()

    @ttl.compute()
    def add():
        for _ in range(W_PER_CORE):
            with a_dfb.wait() as ab, b_dfb.wait() as bb:
                with y_dfb.reserve() as yb:
                    yb.store(ab + bb)

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        ht = cy % H_T
        wt0 = (cx * W_HALVES + (cy // H_T)) * W_PER_CORE
        for j in range(W_PER_CORE):
            with y_dfb.wait() as yb:
                ttl.copy(yb, y[ht, wt0 + j]).wait()


def supports(a, b) -> bool:
    """Is this call the exact shape/dtype the kernel is specialised for?"""
    for t in (a, b):
        if t is None or t.dtype != ttnn.bfloat16 or len(t.shape) != 4:
            return False
        if int(t.shape[0]) != 1 or int(t.shape[1]) != 1:
            return False
        if int(t.shape[-2]) != SEQ_LEN or int(t.shape[-1]) != DIM:
            return False
    return a.memory_config() == b.memory_config()


def residual_add_ttl(a, b, memory_config):
    """Drop-in for `ttnn.add(a, b, memory_config=...)` at the specialised shape."""
    device = a.device()
    a2 = ttnn.reshape(a, [SEQ_LEN, DIM])
    b2 = ttnn.reshape(b, [SEQ_LEN, DIM])
    y2 = ttnn.empty([SEQ_LEN, DIM], a.dtype, ttnn.TILE_LAYOUT, device, memory_config)
    ttl_residual_add(a2, b2, y2)
    return ttnn.reshape(y2, [1, 1, SEQ_LEN, DIM])


def measure(device):
    ta = torch.randn(1, 1, SEQ_LEN, DIM, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, SEQ_LEN, DIM, dtype=torch.bfloat16)

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    a = ttnn.from_torch(ta, **kw)
    b = ttnn.from_torch(tb, **kw)

    got = ttnn.to_torch(residual_add_ttl(a, b, ttnn.L1_MEMORY_CONFIG)).float().flatten()
    golden = (ta.float() + tb.float()).flatten()
    print("TTL_RESADD_PCC=%.6f" % torch.corrcoef(torch.stack([golden, got]))[0, 1].item())

    def run_ttl():
        ttnn.deallocate(residual_add_ttl(a, b, ttnn.L1_MEMORY_CONFIG))

    def run_ttnn():
        ttnn.deallocate(ttnn.add(a, b, memory_config=ttnn.L1_MEMORY_CONFIG))

    for label, fn in (("TTL_RESADD_MS", run_ttl), ("TTNN_RESADD_MS", run_ttnn)):
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
