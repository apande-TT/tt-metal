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
//      element.  bf16_key_pair() below replaces the whole thing with a branchless monotone remap,
//      so a single unsigned compare orders any two values -- and it remaps both elements of a
//      32-bit load together, so the remap itself costs less than the compares it feeds.
//   2. the stock loop carries a second `else if (val == max_val) max_idx = min(...)` branch to keep
//      the lowest index among equal maxima.  Scanning forward with a STRICT `>` already keeps the
//      first maximum, so that branch is pure overhead.
//
// The other structural change is the read: the stock kernel interleaves one read and one scan per
// batch row, so each row pays full NoC latency with the RISC-V idle, and re-runs the whole
// multicast/semaphore handshake with the reduce core once per row.  Here there is no read at all --
// the resident input is WIDTH-SHARDED onto exactly these cores, so each processor's slice is its
// own core's L1 and `src_addr` (the same address on every core, as sharded buffers are) already
// points at it.  The cross-core fold happens once, in pass 2.

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
//
// ...AND BOTH LANES OF A WORD AT ONCE, IN FIVE OPERATIONS RATHER THAN TEN.  The scan loads two
// elements per word anyway, and deriving the mask twice is most of what the inner loop costs.  Per
// lane the mask above is 0xFFFF when the sign bit is set and 0x8000 when it is not, and it can be
// BUILT for both lanes together:
//
//   s  = w & 0x80008000        isolate the two sign bits (only bits 15 and 31 can be set)
//   s - (s >> 15)              0x7FFF in each lane whose sign was set, 0 in the others.  No borrow
//                              can cross the lane boundary: the only nonzero minuend bit in a lane
//                              is its top one, and the subtrahend is 1, so 0x8000 - 1 = 0x7FFF
//                              stays inside the lane and a clear lane computes 0 - 0.
//   | 0x80008000               makes that 0xFFFF where the sign was set and 0x8000 where it was not
//   w ^ mask                   both lanes remapped
//
// Bit-identical to the per-lane form above -- same keys, so the same token -- it just stops paying
// for the sign dispatch twice.
static inline uint32_t bf16_key_pair(uint32_t w) {
    const uint32_t s = w & 0x80008000u;
    return w ^ ((s - (s >> 15)) | 0x80008000u);
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t chunk = get_arg_val<uint32_t>(4);

    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(0);
    // Rows of the batch.
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    // Bytes between consecutive rows inside THIS CORE'S SHARD: `per` bf16 values, and `per` is a
    // multiple of 16 elements = 32 bytes, so the stick carries no L1 alignment padding.
    constexpr uint32_t row_stride = get_compile_time_arg_val(2);
    // Byte offset of this processor's half within the shard's row; constant per slot.
    constexpr uint32_t slot_off = get_compile_time_arg_val(3);

    constexpr auto s_dst_args = TensorAccessorArgs<4>();
    const auto s_dst = TensorAccessor(s_dst_args, dst_addr);

    Noc noc;
    CircularBuffer dst_cb(dst_cb_idx);

    // THE SOURCE IS THIS CORE'S OWN SHARD, SO THERE IS NOTHING TO FETCH.  The resident input is
    // width-sharded onto exactly the cores this kernel runs on, and a sharded L1 buffer sits at the
    // SAME address on every core -- so `src_addr` already points at the slice this processor owns.
    // The whole read path below (a TensorAccessor, a row-ahead prefetch, and a barrier per row)
    // existed only because an interleaved source put every core's slice on someone else's bank.
    const uint32_t base = src_addr + slot_off;

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
        // The scan reads through a NON-volatile pointer so the compiler is free to unroll and
        // schedule it -- with `volatile` every 2-byte load is emitted separately and in order,
        // which measured ~20 cycles/element for a ~7-instruction body.  The fill op that wrote this
        // shard is a separate program, so its writes are ordered before this kernel starts; the
        // barrier this comment used to pair with is gone with the fetch.
        asm volatile("" ::: "memory");

        // TWO ELEMENTS PER LOAD.  The host rounds `per` to a multiple of 16 elements, so count is
        // even and row_stride is a multiple of 32 bytes -- every row is 4-byte aligned and
        // even-length; on this little-endian core the low half of each word is the earlier vocab
        // index, which is the order the first-maximum tie rule needs.
        const uint32_t nwords = count >> 1;
        for (uint32_t b = 0; b < rows; ++b) {
            const tt_l1_ptr uint32_t* q = reinterpret_cast<const tt_l1_ptr uint32_t*>(base + b * row_stride);
            // Seeded from element 0 rather than a sentinel, so the "first maximum wins" rule holds
            // even for a slice whose every value is the same.
            uint32_t w = bf16_key_pair(q[0]);
            uint32_t best = w & 0xFFFFu;
            uint32_t best_i = 0;
            uint32_t k = w >> 16;
            if (k > best) {
                best = k;
                best_i = 1;
            }
            for (uint32_t j = 1; j < nwords; ++j) {
                w = bf16_key_pair(q[j]);
                k = w & 0xFFFFu;
                if (k > best) {
                    best = k;
                    best_i = 2 * j;
                }
                k = w >> 16;
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
