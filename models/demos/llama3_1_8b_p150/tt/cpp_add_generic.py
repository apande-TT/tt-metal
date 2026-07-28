"""C++ Metalium eltwise add via ttnn.generic_op -- AUTHORED, MEASURED, and NOT WIRED IN.

Kept as the record of the cpp rung for `BinaryNgDeviceOperation`.

WHICH instance. Same reasoning as the tt-lang rung (tt/ttl_residual_add.py): of this op's five
shapes only the post-MLP residual add, [128, 4096] bf16, is available to a hand kernel. The two
SILU gate multiplies -- the LARGEST at 8.23 ms and 3.50 ms -- are bf8_b, and the post-attention add
is mixed bf16/bf8_b; a generic_op CB carries one data_format per buffer index, so neither is
expressible without changing the op's dtype contract, which GUIDELINES/12 forbids.

Drives a real reader/compute/writer triple through ttnn.generic_op, adapted from the repo's own
tt_metal/programming_examples/eltwise_binary (kernels copied into tt/kernels/ and generalised from
"one core walks page 0..n" to "each core owns a [start, start+n) slice"), with the output tiles
partitioned across the entire 11x10 compute grid -- i.e. the SAME 110 cores the stock op gets, which
is the one thing the tt-lang rung could not have (its 2-D node grid topped out at 64).

MEASURED in the model against the stock ttnn.add it replaces (352 calls each, same shape/dtype and
the same 110 cores):

    correctness   e2e PCC 0.985099, unchanged from the stock op
    cpp kernel    3.87 us/call
    stock ttnn    3.67 us/call
    whole model   648.17 -> 647.82 ms (BinaryNg -1.47 ms, GenericOp +1.37 ms)

So it lands at PARITY -- 5% slower per call, and a whole-model delta of 0.05% that is inside
run-to-run noise. Not wired in, because the per-call number at equal core count is the honest signal
and it does not beat the stock op.

Worth recording WHY this is the interesting result of the two kernel rungs. The tt-lang attempt on
the same add was 4.87 us/call, and it was slower for a specific reason: `ttl.node` is 2-D, so its
decomposition of a 4x128 tile grid could reach only 64 of the board's 110 cores. generic_op has no
such restriction -- the host hands each core an explicit tile slice -- so this kernel runs the same
110 cores as the stock op, and the gap duly closes from 1.32x to 1.05x. That isolates the cause: the
tt-lang loss was OCCUPANCY, not code quality, and once occupancy is equalised a hand-written
reader/add/writer triple is simply what binary_ng already is. The residual 5% is the last of the
per-tile bookkeeping the production op has had tuned out of it. No arrangement of compute kernels
reduces the bytes an eltwise op must move, which is what the roofline gap on this op actually is.

Run directly to reproduce the numbers in CPP_ADD_*.
"""
import time

import torch

import ttnn
from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32

TILE = 32
SEQ_LEN, DIM = 128, 4096
ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"


def build_program(a, b, c, grid, m, n):
    Mt, Nt = m // TILE, n // TILE
    total_tiles = Mt * Nt
    num_cores = grid.x * grid.y

    core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    tile_bytes = 2 * TILE * TILE  # bfloat16

    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * tile_bytes,  # double-buffered so the reader runs ahead of the FPU
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

    # Partition the tiles across every core, remainder onto the first cores -- the same split
    # binary_ng's own split_work_to_cores does, so the comparison is grid-for-grid.
    per_core = total_tiles // num_cores
    extra = total_tiles % num_cores
    reader_rt, compute_rt, writer_rt = [], [], []
    start = 0
    for i in range(num_cores):
        coord = ttnn.CoreCoord(i % grid.x, i // grid.x)
        n_tiles = per_core + (1 if i < extra else 0)
        reader_rt.append((coord, _VU32([a.buffer_address(), b.buffer_address(), start, n_tiles])))
        compute_rt.append((coord, _VU32([n_tiles])))
        writer_rt.append((coord, _VU32([c.buffer_address(), n_tiles, start])))
        start += n_tiles

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{ROOT}/dataflow/reader_add_partitioned.cpp",
            core_ranges=core_ranges,
            compile_time_args=_VU32(list(a_ct) + list(b_ct)),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{ROOT}/compute/add_tiles_stream.cpp",
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


def supports(a, b) -> bool:
    """Is this call the exact shape/dtype/layout the kernel is specialised for?"""
    for t in (a, b):
        if t is None or t.dtype != ttnn.bfloat16 or len(t.shape) != 4:
            return False
        if int(t.shape[0]) != 1 or int(t.shape[1]) != 1:
            return False
        if int(t.shape[-2]) != SEQ_LEN or int(t.shape[-1]) != DIM:
            return False
    return a.memory_config() == b.memory_config()


def add_cpp(a, b, memory_config):
    """Drop-in for `ttnn.add(a, b, memory_config=...)` at the specialised shape."""
    device = a.device()
    grid = device.compute_with_storage_grid_size()
    c = ttnn.empty([1, 1, SEQ_LEN, DIM], a.dtype, ttnn.TILE_LAYOUT, device, memory_config)
    ttnn.generic_op([a, b, c], build_program(a, b, c, grid, SEQ_LEN, DIM))
    return c


def measure(device, grid):
    ta = torch.randn(1, 1, SEQ_LEN, DIM, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, SEQ_LEN, DIM, dtype=torch.bfloat16)
    golden = (ta.float() + tb.float()).flatten()

    kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    a = ttnn.from_torch(ta, **kw)
    b = ttnn.from_torch(tb, **kw)

    got = ttnn.to_torch(add_cpp(a, b, ttnn.L1_MEMORY_CONFIG)).float().flatten()
    print("CPP_ADD_PCC=%.6f" % torch.corrcoef(torch.stack([golden, got]))[0, 1].item())

    def run_cpp():
        ttnn.deallocate(add_cpp(a, b, ttnn.L1_MEMORY_CONFIG))

    def run_ttnn():
        ttnn.deallocate(ttnn.add(a, b, memory_config=ttnn.L1_MEMORY_CONFIG))

    for label, fn in (("CPP_ADD_MS", run_cpp), ("TTNN_ADD_MS", run_ttnn)):
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
        grid = device.compute_with_storage_grid_size()
        print("GRID=%dx%d" % (grid.x, grid.y))
        measure(device, grid)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
