"""tt-lang matmul on the hot MLP shapes -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the tt-lang rung for each op, measured on its OWN shape rather than inheriting
a sibling's result:

  * `MatmulDeviceOperation 128 x 14336 x 4096` -- short-prefill ff2 down-projection
  * `MatmulDeviceOperation  32 x 14336 x 4096` -- DECODE ff2 down-projection (per-token path)
  * `MatmulDeviceOperation  32 x  4096 x 14336` -- DECODE ff1/ff3 up-projection (per-token path)

Each core owns a strip of the N tiles; K is reduced in-core with an accumulator DFB ping-pong seeded
from the first partial product (ttl 1.0.1 has no block.fill). The same kernel serves every shape.

MEASURED on an 8x8 grid:

    M    K      N        PCC         ttl        ttnn.linear    verdict
    128  14336  4096     0.999691    2.974 ms    0.359 ms      8.3x SLOWER
     32  14336  4096     0.999692    0.816 ms    0.295 ms      2.8x SLOWER
     32   4096  14336    0.999929    0.687 ms    0.316 ms      2.2x SLOWER

The C++ Metalium rung was measured on the same shapes via tt/cpp_mm_generic.py.

Every kernel is CORRECT and every one loses. The loss is dataflow, not math: this kernel re-reads
every A tile once per output tile, so each A tile is pulled per_core_n times on every core, while
ttnn's matmul multicasts in0 across a core row and blocks K.

The three rows separate the two costs cleanly. This kernel's work is m_tiles x per_core_n x k_tiles
per core, so it scales with M -- 2.974 -> 0.816 ms when M drops 128 -> 32, nearly linear. ttnn barely
moves (0.359 -> 0.295) because it is bound by the weight stream, which does not depend on M. So the
gap narrows from 8.3x to ~2x at the decode shapes and then STOPS: what is left in the ttnn number is
DRAM weight bandwidth, and no arrangement of compute kernels reduces bytes moved. Closing even the
remaining 2.2x would require reimplementing ttnn's mcast matmul, which is what the stock op is.

No fusion is left to buy the difference back either: ff2's input round-trip is already removed by the
L1 island in mlp.py, and the gate+up fusion was measured and rejected separately
(tt/gated_mlp_fusion_probe.py).

Run directly to reproduce the table.
"""
import time

import torch

import ttnn
import ttl

TILE = 32
GRID_X, GRID_Y = 8, 8
# (M, K, N) triples this rung was measured for.
SHAPES = [(128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]


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


def measure(device, m, K, N):
    ta = torch.randn(m, K, dtype=torch.bfloat16) * 0.02
    tb = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    golden = ta.float() @ tb.float()

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    a = ttnn.from_torch(ta, **kw)
    b = ttnn.from_torch(tb, **kw)
    y = ttnn.from_torch(torch.zeros(m, N, dtype=torch.bfloat16), **kw)

    ttl_mm(a, b, y)
    got = ttnn.to_torch(y).float()
    tag = "%dx%dx%d" % (m, K, N)
    print("TTL_MM_%s_PCC=%.6f" % (tag, torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()))

    def run_ttl():
        ttl_mm(a, b, y)

    def run_ttnn():
        ttnn.deallocate(ttnn.linear(a, b))

    for label, fn in (("TTL_MM_%s_MS" % tag, run_ttl), ("TTNN_MM_%s_MS" % tag, run_ttnn)):
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
        for m, k, n in SHAPES:
            measure(device, m, k, n)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
