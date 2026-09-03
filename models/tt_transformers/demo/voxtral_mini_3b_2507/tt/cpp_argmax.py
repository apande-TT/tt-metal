# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A hand-written Metalium argmax for the greedy-sampling step, driven through ttnn.generic_op.

WHY NOT THE STOCK OP.  ttnn's multicore argmax is already reached (the input is handed over
ROW_MAJOR so the multicore program factory is selected instead of the single-core one), and it
already spreads over the whole grid.  What is left is per-element cost, not fan-out: fitting
T = a + S/N across a 110-core and a 32-core run of this same call put S at ~17.4 ms-core against a
~30 us fixed term, i.e. ~22 cycles per element on the data-movement RISC-V.  Narrowing the core set
to cut multicast synchronisation made it WORSE, which is the signature of a scan-bound op.  Three
things in the stock kernel account for those cycles, and `_kernels/argmax_scan.cpp` removes all
three: the sign-dispatching bfloat16_greater comparison, the second equality branch that keeps the
lowest index among ties, and the per-batch-row read/scan/handshake serialisation.

TRACE SAFETY.  ttnn.generic_op hashes a program descriptor's runtime-arg COUNT, not its values, and
nothing patches buffer addresses back in on a cache hit -- so a descriptor is only correct while the
tensors it was built against keep their addresses.  Hence: the descriptors are built exactly once,
against PERSISTENT tensors allocated once, and each call copies the fresh logits into that same
resident input buffer.  Rebuilding a descriptor per call would also make the op uncapturable, since
trace records commands rather than re-running the build.
"""
from __future__ import annotations

import os

import ttnn

_KERNEL_DIR = "models/tt_transformers/demo/voxtral_mini_3b_2507/_kernels"
_SCAN_KERNEL = f"{_KERNEL_DIR}/argmax_scan.cpp"
_REDUCE_KERNEL = f"{_KERNEL_DIR}/argmax_reduce.cpp"

# Elements per core per batch row are rounded up to this. Each core's slice is halved between its
# two data-movement processors, so the HALF has to stay 16-byte aligned (64 bf16 = 128 bytes) --
# hence 128 here, not 64.
_ALIGN_ELEMS = 128

# One (scratch, staging) circular-buffer pair per data-movement processor: the two scan instances
# run concurrently on the same core and cannot share either buffer.
_CB_PAIRS = ((0, 1), (2, 3))


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_ARGMAX", "1") == "1"


def _core_ranges(grid_x: int, n: int):
    """The first `n` cores of the grid in row-major (x fastest) order."""
    full, rem = divmod(n, grid_x)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


class CppArgmax:
    """Resident two-pass argmax over the last dim of a [.., rows, vocab] bf16 ROW_MAJOR tensor.

    Pass 1 reduces each core's vocab slice, for every row, to one (key, index) pair; pass 2 folds
    the per-core pairs on a single core.  Output matches the stock op exactly: uint32 ROW_MAJOR
    [1, rows, 1] holding the index of the FIRST maximal element of each row.
    """

    def __init__(self, device, rows: int, vocab: int, out=None):
        self.device = device
        self.rows = int(rows)
        self.vocab = int(vocab)
        # WRITE THE SAMPLED IDS WHERE THE CALLER ALREADY KEEPS THEM.  Pass 2 writes 4 bytes to page
        # `row` of its destination, so ANY resident uint32 ROW_MAJOR tensor with one page per row
        # serves -- including the pipeline's own next-token buffer.  Handing that in removes the
        # `ttnn.copy(ids -> next_ids)` the caller would otherwise run every token, which is a whole
        # launch for 32 bytes.  It must be RESIDENT (allocated once) for the same reason our own
        # buffers are: the descriptor is built against its address and nothing patches it later.
        self._out_external = out is not None

        grid = device.compute_with_storage_grid_size()
        per = -(-self.vocab // (grid.x * grid.y))
        per = -(-per // _ALIGN_ELEMS) * _ALIGN_ELEMS
        self.per = per
        self.half = per // 2
        self.ncores = -(-self.vocab // per)
        # Two chunks per core, one per data-movement processor. Chunk index ASCENDS with vocab
        # offset (core c owns chunks 2c and 2c+1, in that order), which is what lets pass 2 keep
        # the lowest index among equal maxima by a single forward strict-greater scan.
        self.nparts = 2 * self.ncores
        self.cores = _core_ranges(grid.x, self.ncores)
        self.grid_x = grid.x

        l1 = ttnn.L1_MEMORY_CONFIG
        self.src = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.rows, self.vocab]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, l1
        )
        # ONE page, not one per chunk: pass 2 then pulls every partial in a single transfer
        # instead of `nparts` serial page requests.
        self.part = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.nparts * 2 * self.rows]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, l1
        )
        self.out = out
        if self.out is None:
            self.out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, self.rows, 1]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, l1
            )

        self._scan_desc = self._build_scan()
        self._reduce_desc = self._build_reduce()

    # ------------------------------------------------------------------ build
    @staticmethod
    def _accessor_args(tensor):
        acc = ttnn.TensorAccessorArgs(tensor)
        if list(acc.get_common_runtime_args()):
            # A sharded/complex accessor splits its description across compile-time AND common
            # runtime args; these kernels only wire the compile-time half, so refuse rather than
            # build a descriptor that reads garbage.
            raise RuntimeError("cpp_argmax: tensor needs common runtime accessor args")
        return list(acc.get_compile_time_args())

    def _build_scan(self):
        row_stride = self.half * 2
        part_bytes = self.rows * 8
        src_addr = self.src.buffer_address()
        dst_addr = self.part.buffer_address()
        src_acc = self._accessor_args(self.src)
        part_acc = self._accessor_args(self.part)

        kernels, cbs = [], []
        for slot, (src_cb, dst_cb) in enumerate(_CB_PAIRS):
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=self.rows * row_stride,
                    core_ranges=self.cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=src_cb, data_format=ttnn.bfloat16, page_size=row_stride)
                    ],
                )
            )
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=part_bytes,
                    core_ranges=self.cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=dst_cb, data_format=ttnn.uint32, page_size=part_bytes)
                    ],
                )
            )

            rt = ttnn.RuntimeArgs()
            for core in range(self.ncores):
                y, x = divmod(core, self.grid_x)
                chunk = 2 * core + slot
                start = core * self.per + slot * self.half
                count = max(0, min(self.half, self.vocab - start))
                rt[x][y] = [src_addr, dst_addr, start, count, chunk]

            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_SCAN_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.cores,
                    compile_time_args=[src_cb, dst_cb, self.rows, row_stride] + src_acc + part_acc,
                    runtime_args=rt,
                    # Reader binds one data-movement RISC-V, writer the other; the pair is how a
                    # single program reaches both processors on the same core.
                    config=ttnn.ReaderConfigDescriptor() if slot == 0 else ttnn.WriterConfigDescriptor(),
                )
            )
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def _build_reduce(self):
        src_cb, dst_cb = _CB_PAIRS[0]
        cta = [src_cb, dst_cb, self.rows, self.nparts]
        cta += self._accessor_args(self.part)
        cta += self._accessor_args(self.out)

        all_bytes = self.nparts * self.rows * 8
        # One core per batch row, so each core's fold is `nparts` entries rather than
        # `nparts * rows`. The rows are independent, so no core has to wait on any other.
        fold_cores = _core_ranges(self.grid_x, self.rows)
        cbs = [
            ttnn.CBDescriptor(
                total_size=all_bytes,
                core_ranges=fold_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=src_cb, data_format=ttnn.uint32, page_size=all_bytes)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=16,
                core_ranges=fold_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=dst_cb, data_format=ttnn.uint32, page_size=16)
                ],
            ),
        ]

        rt = ttnn.RuntimeArgs()
        part_addr = self.part.buffer_address()
        out_addr = self.out.buffer_address()
        for row in range(self.rows):
            y, x = divmod(row, self.grid_x)
            rt[x][y] = [part_addr, out_addr, row]

        kernel = ttnn.KernelDescriptor(
            kernel_source=_REDUCE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=fold_cores,
            compile_time_args=cta,
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)

    # ------------------------------------------------------------------- call
    def __call__(self, logits_rm):
        # The two call sites hand back [1, rows, V] and [rows, 1, V]; both page identically under
        # ROW_MAJOR (one page per row, vocab last), so one resident buffer serves both.
        ttnn.copy(ttnn.reshape(logits_rm, (1, self.rows, self.vocab)), self.src)
        ttnn.generic_op([self.src, self.part], self._scan_desc)
        return ttnn.generic_op([self.part, self.out], self._reduce_desc)
