"""C++ Metalium matmul via ttnn.generic_op -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the cpp rung for `MatmulDeviceOperation 128 x 4096 x 14336`.

Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
the output tiles partitioned across the entire compute grid.

MEASURED on the real shape (M=128, K=4096, N=14336) on the full 11x10 P150 grid:

    correctness   PCC 0.999040   (the kernel is right)
    generic_op    3.573 ms/call
    ttnn.linear   0.331 ms/call  -> the kernel is 10.8x SLOWER

Same root cause as the tt-lang attempt in ttl_gated_ffn.py, and it is a dataflow property, not a
tuning miss: this reader fetches every A tile again for each output tile, so A is re-read Nt times
from DRAM. ttnn's production matmul multicasts each in0 tile across a whole core row and blocks K,
so it moves a small fraction of the bytes. Beating it would mean reimplementing that mcast matmul --
which is what the stock op already is.

Together the two rungs bound the custom-kernel lever for this op: the only thing a hand kernel can
add here is fusion, the fusion is worth <=0.5% (measured independently via an L1 island for the
same intermediates), and both hand kernels lose by 5.7x-10.8x on the dataflow.

Run directly to reproduce the numbers above.
"""
import time

import torch

import ttnn
from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32

TILE = 32
M, K, N = 128, 4096, 14336
ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"


def build_program(a, b, c, grid):
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
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


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        grid = device.compute_with_storage_grid_size()
        print("GRID=%dx%d" % (grid.x, grid.y))

        ta = torch.randn(M, K, dtype=torch.bfloat16) * 0.05
        tb = torch.randn(K, N, dtype=torch.bfloat16) * 0.05
        golden = ta.float() @ tb.float()

        kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        a = ttnn.from_torch(ta, **kw)
        b = ttnn.from_torch(tb, **kw)
        c = ttnn.from_torch(torch.zeros(M, N, dtype=torch.bfloat16), **kw)

        prog = build_program(a, b, c, grid)
        ttnn.generic_op([a, b, c], prog)
        got = ttnn.to_torch(c).float()
        pcc = torch.corrcoef(torch.stack([golden.flatten(), got.flatten()]))[0, 1].item()
        print("CPP_MM_PCC=%.6f" % pcc)

        def run_cpp():
            ttnn.generic_op([a, b, c], prog)

        def run_ttnn():
            out = ttnn.linear(a, b)
            ttnn.deallocate(out)

        for label, fn in (("CPP_MM_MS", run_cpp), ("TTNN_MM_MS", run_ttnn)):
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
