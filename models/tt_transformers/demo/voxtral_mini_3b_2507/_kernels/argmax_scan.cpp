// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PASS 1 of the greedy-sampling argmax: every core reduces its own slice of the vocab, for all
// rows of the batch, down to one (key, index) pair per row.  Pass 2 (argmax_reduce.cpp) folds the
// per-core pairs together.
//
// This source is instantiated TWICE per core, once on each of the two data-movement RISC-Vs, each
// over half the core's slice.  A Tensix core has two of them and a scan-bound kernel that binds
// only the reader leaves the other completely idle, so the second instance doubles the scan
// engines for free -- the reads are far too small to contend for the core's L1 or NoC.
//
// Why this exists at all: ttnn's stock multicore argmax is SCAN-bound, not bandwidth-bound.  A
// fit of T = a + S/N over a 110-core vs 32-core run put S at ~17.4 ms-core for this vocab, i.e.
// ~22 cycles per element on the data-movement RISC-V, against a ~30 us fixed term.  The reads are
// nearly free; the per-element work is what costs.  Two things in the stock inner loop account for
// it, and both are removed here:
//
//   1. bfloat16_greater() dispatches on the sign bits with up to three unpredictable branches per
//      element.  bf16_key() below replaces the whole thing with a branchless monotone remap, so a
//      single unsigned compare orders any two values.
//   2. the stock loop carries a second `else if (val == max_val) max_idx = min(...)` branch to keep
//      the lowest index among equal maxima.  Scanning forward with a STRICT `>` already keeps the
//      first maximum, so that branch is pure overhead.
//
// The other structural change is the read: the stock kernel interleaves one read and one scan per
// batch row, so each row pays full NoC latency with the RISC-V idle, and re-runs the whole
// multicast/semaphore handshake with the reduce core once per row.  Here every row's slice is
// fetched up front behind a single barrier, and the cross-core fold happens once, in pass 2.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#include <stdint.h>

// Map a bf16 bit pattern to a uint16 whose UNSIGNED order is the numeric order of the bf16.
//   sign 0 (>= 0): flip the top bit,  [0x0000, 0x7FFF] -> [0x8000, 0xFFFF]
//   sign 1 (<  0): flip every bit,    [0x8000, 0xFFFF] -> [0x7FFF, 0x0000]
// so every negative key lands below every non-negative one and, within the negatives, a larger
// magnitude gives a smaller key.  This is exact and total on the values a logit can take, and it
// agrees with bfloat16_greater() on the -0/+0 pair (+0 compares greater), so the sampled token is
// bit-identical to the stock op's.
static inline uint32_t bf16_key(uint32_t v) {
    const uint32_t mask = 0x8000u | (uint32_t)(0u - (v >> 15));
    return (v ^ mask) & 0xFFFFu;
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t chunk = get_arg_val<uint32_t>(4);

    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);
    // Rows of the batch. The input is ROW_MAJOR with the vocab last, so one row is one page.
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    // Bytes reserved per row in the scratch buffer; a multiple of 128 so every slice the NoC
    // lands is 16-byte aligned at both ends.
    constexpr uint32_t row_stride = get_compile_time_arg_val(3);

    constexpr auto s_src_args = TensorAccessorArgs<4>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();

    const auto s_src = TensorAccessor(s_src_args, src_addr);
    const auto s_dst = TensorAccessor(s_dst_args, dst_addr);

    Noc noc;
    CircularBuffer src_cb(src_cb_idx);
    CircularBuffer dst_cb(dst_cb_idx);

    const uint32_t base = src_cb.get_write_ptr();
    const uint32_t nbytes = count << 1;
    const uint32_t byte_off = start << 1;

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_cb.get_write_ptr());

    if (count == 0) {
        // The vocab does not always divide evenly over (cores x 2 processors), so the last chunk
        // can be empty.  Key 0 sits below every key a real bf16 maps to, so pass 2's strict `>`
        // can never pick this slot, and chunk 0 -- which pass 2 seeds from -- is never empty.
        for (uint32_t b = 0; b < rows; ++b) {
            out[2 * b] = 0;
            out[2 * b + 1] = 0;
        }
    } else {
        // All rows in flight, then ONE barrier.
        for (uint32_t b = 0; b < rows; ++b) {
            noc.async_read(
                s_src, src_cb, nbytes, {.page_id = b, .offset_bytes = byte_off}, {.offset_bytes = b * row_stride});
        }
        noc.async_read_barrier();
        // The scan below reads through a NON-volatile pointer so the compiler is free to unroll
        // and schedule it -- with `volatile` every 2-byte load is emitted separately and in order,
        // which measured ~20 cycles/element for a ~7-instruction body.  The barrier above is not
        // by itself a guarantee to the compiler that L1 changed underneath it, so state that
        // explicitly here rather than relying on the read barrier to imply it.
        asm volatile("" ::: "memory");

        // TWO ELEMENTS PER LOAD.  count is a multiple of 64 and row_stride of 128, so every row is
        // 4-byte aligned and even-length; on this little-endian core the low half of each word is
        // the earlier vocab index, which is the order the first-maximum tie rule needs.
        const uint32_t nwords = count >> 1;
        for (uint32_t b = 0; b < rows; ++b) {
            const tt_l1_ptr uint32_t* q = reinterpret_cast<const tt_l1_ptr uint32_t*>(base + b * row_stride);
            // Seeded from element 0 rather than a sentinel, so the "first maximum wins" rule holds
            // even for a slice whose every value is the same.
            uint32_t w = q[0];
            uint32_t best = bf16_key(w & 0xFFFFu);
            uint32_t best_i = 0;
            uint32_t k = bf16_key(w >> 16);
            if (k > best) {
                best = k;
                best_i = 1;
            }
            for (uint32_t j = 1; j < nwords; ++j) {
                w = q[j];
                k = bf16_key(w & 0xFFFFu);
                if (k > best) {
                    best = k;
                    best_i = 2 * j;
                }
                k = bf16_key(w >> 16);
                if (k > best) {
                    best = k;
                    best_i = 2 * j + 1;
                }
            }
            out[2 * b] = best;
            out[2 * b + 1] = start + best_i;
        }
    }

    // The partials are ONE page shared by every chunk (so pass 2 reads them in a single
    // transfer); this chunk owns the 8-bytes-per-row slot at `chunk * rows * 8`.
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(dst_cb), s_dst, rows * 8, {.offset_bytes = 0},
        {.page_id = 0, .offset_bytes = chunk * rows * 8});
    noc.async_write_barrier();
}
