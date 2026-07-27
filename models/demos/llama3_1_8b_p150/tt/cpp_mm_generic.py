"""C++ Metalium matmul via ttnn.generic_op -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the cpp rung for every hot dense matmul in the MLP:

  * `MatmulDeviceOperation 128 x  4096 x 14336` -- the short-prefill ff1/ff3 up-projection
  * `MatmulDeviceOperation 128 x 14336 x  4096` -- the short-prefill ff2 down-projection
  * `MatmulDeviceOperation  32 x 14336 x  4096` -- the DECODE ff2 down-projection (per-token path)
  * `MatmulDeviceOperation  32 x  4096 x 14336` -- the DECODE ff1/ff3 up-projection (per-token path)
  * `MatmulDeviceOperation  32 x  4096 x  6144` -- the DECODE fused QKV projection (per-token path)

Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
the output tiles partitioned across the entire compute grid.

MEASURED on the real shapes, on the full 11x10 P150 grid:

    M    K      N        PCC        generic_op    ttnn.linear     verdict
    128  4096   14336    0.999015    3.562 ms      0.333 ms       10.7x SLOWER
    128  14336  4096     0.993626    3.113 ms      0.358 ms        8.7x SLOWER
     32  14336  4096     0.993630    0.916 ms      0.294 ms        3.1x SLOWER
     32  4096   14336    0.999003    1.041 ms      0.309 ms        3.4x SLOWER
     32  4096    6144    0.999014    0.344 ms      0.139 ms        2.5x SLOWER

Every kernel is CORRECT and every one loses. The cause is dataflow, not tuning: this reader fetches
every A tile again for each output tile, so A is re-read Nt times from DRAM, while ttnn's production
matmul multicasts each in0 tile across a whole core row and blocks K, moving a small fraction of the
bytes. Beating it would mean reimplementing that mcast matmul -- which is what the stock op already is.

The decode row is the one worth reading twice. Dropping M from 128 to 32 shrinks THIS kernel ~3.4x
(3.118 -> 0.916) because its work is Mt x Nt x Kt, but barely moves ttnn (0.357 -> 0.292) because ttnn
is bound by the 33 MB w2 weight stream, which does not depend on M at all. So the gap narrows from
8.7x to 3.1x and then stops: what remains in the ttnn number is DRAM weight bandwidth, and no
arrangement of compute kernels reduces bytes moved.

Together with the tt-lang rung (tt/ttl_ff2_matmul.py, same shapes, 2.7x-8.3x slower) this bounds the
custom-kernel lever for these ops: the only thing a hand kernel could add is fusion, and the fusion is
worth <=0.5% (measured independently via an L1 island for the same intermediates).

Run directly to reproduce the table above.
"""
import time

import torch

import ttnn
from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32

TILE = 32
# (M, K, N) of every op this rung was measured for: the ff1/ff3 up-projection, then ff2's
# down-projection at the short-prefill and the DECODE row counts.
SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336), (32, 4096, 6144)]
ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"


def build_program(a, b, c, grid, m, k, n):
    Mt, Kt, Nt = m // TILE, k // TILE, n // TILE
    num_cores = grid.x * grid.y
    total_tiles = Mt * Nt

    core_ranges = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    tile_bytes = 2 * TILE * TILE  # bfloat16

    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * tile_bytes,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)
            ],
        )
        for idx in (0, 1, 16)
    ]

    a_ct = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    b_ct = ttnn.TensorAccessorArgs(b).get_compile_time_args()
    c_ct = ttnn.TensorAccessorArgs(c).get_compile_time_args()

    # Partition the output tiles across cores, remainder spread over the first cores.
    per_core = total_tiles // num_cores
    extra = total_tiles % num_cores
    reader_rt, compute_rt, writer_rt = [], [], []
    start = 0
    for i in range(num_cores):
        cx, cy = i % grid.x, i // grid.x
        n_tiles = per_core + (1 if i < extra else 0)
        coord = ttnn.CoreCoord(cx, cy)
        reader_rt.append((coord, _VU32([a.buffer_address(), b.buffer_address(), Mt, Kt, Nt, start, n_tiles])))
        compute_rt.append((coord, _VU32([n_tiles, Kt])))
        writer_rt.append((coord, _VU32([c.buffer_address(), n_tiles, start])))
        start += n_tiles

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{ROOT}/dataflow/reader_mm_partitioned.cpp",
            core_ranges=core_ranges,
            compile_time_args=_VU32(list(a_ct) + list(b_ct)),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{ROOT}/compute/mm_gated_ffn.cpp",
            core_ranges=core_ranges,
            compile_time_args=_VU32([]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{ROOT}/dataflow/writer_mm_partitioned.cpp",
            core_ranges=core_ranges,
            compile_time_args=_VU32(list(c_ct)),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def measure(device, grid, m, k, n):
    ta = torch.randn(m, k, dtype=torch.bfloat16) * 0.05
    tb = torch.randn(k, n, dtype=torch.bfloat16) * 0.05
    golden = ta.float() @ tb.float()

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    a = ttnn.from_torch(ta, **kw)
    b = ttnn.from_torch(tb, **kw)
    c = ttnn.from_torch(torch.zeros(m, n, dtype=torch.bfloat16), **kw)

    tag = "%dx%dx%d" % (m, k, n)
    prog = build_program(a, b, c, grid, m, k, n)
    ttnn.generic_op([a, b, c], prog)
    got = ttnn.to_torch(c).float()
    pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
    print("CPP_MM_%s_PCC=%.6f" % (tag, pcc))

    def run_cpp():
        ttnn.generic_op([a, b, c], prog)

    def run_ttnn():
        ttnn.deallocate(ttnn.linear(a, b))

    for label, fn in (("CPP_MM_%s_MS" % tag, run_cpp), ("TTNN_MM_%s_MS" % tag, run_ttnn)):
        fn()
        ttnn.synchronize_device(device)
        t0 = time.monotonic()
        for _ in range(5):
            fn()
        ttnn.synchronize_device(device)
        print("%s=%.3f" % (label, (time.monotonic() - t0) * 1000.0 / 5))

    for t in (a, b, c):
        ttnn.deallocate(t)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        grid = device.compute_with_storage_grid_size()
        print("GRID=%dx%d" % (grid.x, grid.y))
        for m, k, n in SHAPES:
            measure(device, grid, m, k, n)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
