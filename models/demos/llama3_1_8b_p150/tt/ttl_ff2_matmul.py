"""tt-lang matmul on the ff2 shape -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the tt-lang rung for `MatmulDeviceOperation 128 x 14336 x 4096`, measured on
ff2's OWN shape rather than reusing the w1/w3 result. Each core owns a strip of the N tiles
(128 N-tiles / 64 cores = 2 each, exact); K is reduced in-core with an accumulator DFB ping-pong
seeded from the first partial product (ttl 1.0.1 has no block.fill).

MEASURED on M=128, K=14336, N=4096, 8x8 grid:

    correctness   PCC 0.999682   (the kernel is right)
    ttl matmul    2.961 ms/call
    ttnn.linear   0.364 ms/call  -> 8.1x SLOWER

The C++ Metalium rung was measured on the same shape via tt/cpp_mm_generic.py: PCC 0.993625,
3.113 ms vs 0.362 ms -> 8.6x slower.

Unlike w1/w3 there is no fusion available here either: ff2's output feeds a residual add and (on a
multi-device mesh) a CCL, and its input round-trip is already removed by the L1 island in mlp.py. So
a hand kernel has nothing to add beyond dataflow it cannot win on -- each core re-reads every A tile
per output tile, while ttnn's matmul multicasts in0 across a core row and blocks K.

Run directly to reproduce.
"""
import time

import torch

import ttnn
import ttl

TILE = 32
M, K, N = 128, 14336, 4096
GRID_X, GRID_Y = 8, 8


@ttl.operation(grid=(GRID_Y, GRID_X))
def ttl_mm(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
    m_tiles = a.shape[0] // TILE
    n_tiles = b.shape[1] // TILE
    k_tiles = a.shape[1] // TILE
    per_core_n = n_tiles // (GRID_X * GRID_Y)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        cx, cy = ttl.node(dims=2)
        n_base = (cy * GRID_X + cx) * per_core_n
        for mt in range(m_tiles):
            for j in range(per_core_n):
                nt = n_base + j
                for kt in range(k_tiles):
                    with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                        t0 = ttl.copy(a[mt, kt], ab)
                        t1 = ttl.copy(b[kt, nt], bb)
                        t0.wait()
                        t1.wait()

    @ttl.compute()
    def compute():
        for _ in range(m_tiles):
            for _ in range(per_core_n):
                with a_dfb.wait() as ab, b_dfb.wait() as bb:
                    with acc_dfb.reserve() as acc:
                        acc.store(ab @ bb)
                for _ in range(k_tiles - 1):
                    with a_dfb.wait() as ab, b_dfb.wait() as bb, acc_dfb.wait() as pre:
                        with acc_dfb.reserve() as acc:
                            acc.store(pre + ab @ bb)
                with acc_dfb.wait() as acc:
                    with y_dfb.reserve() as yb:
                        yb.store(acc)

    @ttl.datamovement()
    def write():
        cx, cy = ttl.node(dims=2)
        n_base = (cy * GRID_X + cx) * per_core_n
        for mt in range(m_tiles):
            for j in range(per_core_n):
                with y_dfb.wait() as yb:
                    ttl.copy(yb, y[mt, n_base + j]).wait()


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        ta = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
        tb = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        golden = ta.float() @ tb.float()

        kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        a = ttnn.from_torch(ta, **kw)
        b = ttnn.from_torch(tb, **kw)
        y = ttnn.from_torch(torch.zeros(M, N, dtype=torch.bfloat16), **kw)

        ttl_mm(a, b, y)
        got = ttnn.to_torch(y).float()
        print("TTL_FF2_PCC=%.6f" % torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item())

        def run_ttl():
            ttl_mm(a, b, y)

        def run_ttnn():
            ttnn.deallocate(ttnn.linear(a, b))

        for label, fn in (("TTL_FF2_MS", run_ttl), ("TTNN_FF2_MS", run_ttnn)):
            fn()
            ttnn.synchronize_device(device)
            t0 = time.monotonic()
            for _ in range(5):
                fn()
            ttnn.synchronize_device(device)
            print("%s=%.3f" % (label, (time.monotonic() - t0) * 1000.0 / 5))
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
