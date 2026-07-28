"""tt-lang kernel for the prefill gated FFN -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the tt-lang rung for `MatmulDeviceOperation 128 x 4096 x 14336`.

`ttl_gated_ffn` computes y = silu(x @ w1) * (x @ w3) in ONE kernel, so both [seq, hidden]
intermediates stay in L1 and never round-trip to DRAM -- the fusion GUIDELINES/11 names as the
highest-value tt-lang target, and one ttnn cannot express (ttnn.linear(activation=...) fuses an
activation into ONE matmul, not across two). Each core owns a strip of the N tiles; K is reduced
in-core with an accumulator DFB ping-pong (ttl 1.0.1 has no block.fill, so the accumulator is
seeded from the first partial product rather than zeroed).

MEASURED on the real shape (M=128, K=4096, N=14336) on an 8x8 grid, against the stock
ttnn.linear/linear/mul chain it replaces:

    correctness   PCC 0.999833   (the kernel is right)
    fused ttl     4.065 ms/call
    stock ttnn    0.714 ms/call  -> the kernel is 5.7x SLOWER

So it is deliberately not on the hot path. Two reasons it loses, both structural:
  * It streams single tiles (DFB shape (1,1)) and re-reads each x tile once per N column, while
    ttnn's 2D-mcast matmul broadcasts each in0 tile across a whole core row. The intermediate
    traffic the fusion saves (~6 MB/layer) is dwarfed by the ~88 MB/layer of weight reads it
    cannot avoid -- which is also why the pure-TTNN version of the same idea (an L1 island for
    the intermediates) measured only -0.5% device time and failed the production gate.
  * It requires bf16 operands. Production w1/w3 are bf4_b in a DRAM-sharded memory config, and
    tt-lang cannot index those; converting them to bf16 would 4x the weight bytes and break the
    op's dtype contract, which GUIDELINES/11 forbids.

Run directly (`python -m ...tt.ttl_gated_ffn`) to reproduce the numbers above.
"""
import time

import torch

import ttnn
import ttl

TILE = 32
M, K, N = 128, 4096, 14336
GRID_X, GRID_Y = 8, 8


@ttl.operation(grid=(GRID_Y, GRID_X))
def ttl_gated_ffn(x: ttnn.Tensor, w1: ttnn.Tensor, w3: ttnn.Tensor, y: ttnn.Tensor) -> None:
    """y = silu(x @ w1) * (x @ w3), with both intermediates staying in L1."""
    m_tiles = x.shape[0] // TILE
    n_tiles = w1.shape[1] // TILE
    k_tiles = x.shape[1] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    w1_dfb = ttl.make_dataflow_buffer_like(w1, shape=(1, 1), block_count=2)
    w3_dfb = ttl.make_dataflow_buffer_like(w3, shape=(1, 1), block_count=2)
    acc1_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    acc3_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    per_core_n = n_tiles // (GRID_X * GRID_Y)

    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        n_base = (cy * GRID_X + cx) * per_core_n
        for mt in range(m_tiles):
            for j in range(per_core_n):
                nt = n_base + j
                for kt in range(k_tiles):
                    with x_dfb.reserve() as xb, w1_dfb.reserve() as w1b, w3_dfb.reserve() as w3b:
                        t0 = ttl.copy(x[mt, kt], xb)
                        t1 = ttl.copy(w1[kt, nt], w1b)
                        t2 = ttl.copy(w3[kt, nt], w3b)
                        t0.wait()
                        t1.wait()
                        t2.wait()

    @ttl.compute()
    def compute():
        for _ in range(m_tiles):
            for _ in range(per_core_n):
                with x_dfb.wait() as xb, w1_dfb.wait() as w1b, w3_dfb.wait() as w3b:
                    with acc1_dfb.reserve() as a1, acc3_dfb.reserve() as a3:
                        a1.store(xb @ w1b)
                        a3.store(xb @ w3b)
                for _ in range(k_tiles - 1):
                    with x_dfb.wait() as xb, w1_dfb.wait() as w1b, w3_dfb.wait() as w3b:
                        with acc1_dfb.wait() as p1, acc3_dfb.wait() as p3:
                            with acc1_dfb.reserve() as a1, acc3_dfb.reserve() as a3:
                                a1.store(p1 + xb @ w1b)
                                a3.store(p3 + xb @ w3b)
                # the fusion: gate and multiply while both partials are still in L1
                with acc1_dfb.wait() as a1, acc3_dfb.wait() as a3:
                    with y_dfb.reserve() as yb:
                        yb.store(ttl.silu(a1) * a3)

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        n_base = (cy * GRID_X + cx) * per_core_n
        for mt in range(m_tiles):
            for j in range(per_core_n):
                with y_dfb.wait() as yb:
                    ttl.copy(yb, y[mt, n_base + j]).wait()


def bench(fn, iters=5):
    fn()
    ttnn.synchronize_device(fn.device)
    t0 = time.monotonic()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(fn.device)
    return (time.monotonic() - t0) * 1000.0 / iters


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        tx = torch.randn(M, K, dtype=torch.bfloat16) * 0.05
        t1 = torch.randn(K, N, dtype=torch.bfloat16) * 0.05
        t3 = torch.randn(K, N, dtype=torch.bfloat16) * 0.05
        golden = torch.nn.functional.silu(tx.float() @ t1.float()) * (tx.float() @ t3.float())

        kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.from_torch(tx, **kw)
        w1 = ttnn.from_torch(t1, **kw)
        w3 = ttnn.from_torch(t3, **kw)
        y = ttnn.from_torch(torch.zeros(M, N, dtype=torch.bfloat16), **kw)

        ttl_gated_ffn(x, w1, w3, y)
        got = ttnn.to_torch(y).float()
        pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
        print("TTL_FFN_PCC=%.6f" % pcc)

        def run_ttl():
            ttl_gated_ffn(x, w1, w3, y)

        def run_ttnn():
            a = ttnn.linear(x, w1, activation="silu")
            b = ttnn.linear(x, w3)
            c = ttnn.mul(a, b)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            ttnn.deallocate(c)

        run_ttl.device = device
        run_ttnn.device = device
        print("TTL_FFN_MS=%.3f" % bench(run_ttl))
        print("TTNN_FFN_MS=%.3f" % bench(run_ttnn))
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
