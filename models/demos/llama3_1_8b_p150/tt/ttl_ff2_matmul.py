"""tt-lang matmul on the ff2 shapes -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the tt-lang rung for BOTH ff2 shapes, each measured on its OWN shape rather
than inheriting the w1/w3 result:

  * `MatmulDeviceOperation 128 x 14336 x 4096` -- the short-prefill down-projection
  * `MatmulDeviceOperation  32 x 14336 x 4096` -- the DECODE down-projection (the per-token path)

Each core owns a strip of the N tiles (128 N-tiles / 64 cores = 2 each, exact); K is reduced in-core
with an accumulator DFB ping-pong seeded from the first partial product (ttl 1.0.1 has no
block.fill). Only M differs between the two runs, so the same kernel serves both.

MEASURED on K=14336, N=4096, 8x8 grid:

    M=128   PCC 0.999692   ttl 2.957 ms/call   vs ttnn.linear 0.358 ms/call  -> 8.3x SLOWER
    M= 32   PCC 0.999695   ttl 0.816 ms/call   vs ttnn.linear 0.300 ms/call  -> 2.7x SLOWER

The C++ Metalium rung was measured on the same shapes via tt/cpp_mm_generic.py.

Why the hand kernel cannot win here: the loss is dataflow, not math. This kernel re-reads every A
tile once per output tile, so each A tile is pulled per_core_n times on every core, while ttnn's
matmul multicasts in0 across a core row and blocks K. The two rows above show the shapes of the two
costs cleanly. This kernel's work is m_tiles x per_core_n x k_tiles per core, so it scales with M:
2.957 -> 0.816 ms when M drops 128 -> 32 (~3.6x, i.e. nearly linear). ttnn barely moves, 0.358 ->
0.300 ms, because it is bound by the 33 MB w2 weight stream, which is independent of M. So the gap
NARROWS from 8.3x to 2.7x at the decode shape -- but a kernel that is still 2.7x slower than the
stock op is not a win, and the direction of travel is against it: the remaining ttnn time is almost
entirely DRAM weight bandwidth, which no arrangement of compute kernels can reduce. There is no
fusion left to buy the difference back either: ff2's output feeds a residual add (and a CCL on a
multi-device mesh), and its input round-trip is already removed by the L1 island in mlp.py.

Run directly to reproduce both rows.
"""
import time

import torch

import ttnn
import ttl

TILE = 32
K, N = 14336, 4096
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


def measure(device, m):
    ta = torch.randn(m, K, dtype=torch.bfloat16) * 0.02
    tb = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    golden = ta.float() @ tb.float()

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    a = ttnn.from_torch(ta, **kw)
    b = ttnn.from_torch(tb, **kw)
    y = ttnn.from_torch(torch.zeros(m, N, dtype=torch.bfloat16), **kw)

    ttl_mm(a, b, y)
    got = ttnn.to_torch(y).float()
    print("TTL_FF2_M%d_PCC=%.6f" % (m, torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()))

    def run_ttl():
        ttl_mm(a, b, y)

    def run_ttnn():
        ttnn.deallocate(ttnn.linear(a, b))

    for label, fn in (("TTL_FF2_M%d_MS" % m, run_ttl), ("TTNN_FF2_M%d_MS" % m, run_ttnn)):
        fn()
        ttnn.synchronize_device(device)
        t0 = time.monotonic()
        for _ in range(5):
            fn()
        ttnn.synchronize_device(device)
        print("%s=%.3f" % (label, (time.monotonic() - t0) * 1000.0 / 5))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        for m in (128, 32):  # short-prefill shape, then the DECODE (per-token) shape
            measure(device, m)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
