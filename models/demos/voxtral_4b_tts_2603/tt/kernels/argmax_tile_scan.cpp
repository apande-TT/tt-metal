// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PASS 1 of the greedy sampler: reduce a slice of the vocab, for every row of the batch, to one
// (key, index) pair per row. `argmax_tile_reduce.cpp` folds the per-slice pairs.
//
// THIS KERNEL READS THE TILE LAYOUT DIRECTLY, which is the point of it. `ttnn.argmax` needs a
// ROW_MAJOR input to reach its multicore factory at all, so the stock sampling path is
// untilize-then-scan: the 131072-wide logits are rewritten once purely to change their layout,
// and then read again by the reduction. A scan does not care what order it visits elements in --
// it is a commutative reduce whose only ordering requirement is the tie-break, and that is
// carried by the INDEX, not by the visit order. So the relayout buys the reduction nothing and
// this kernel skips it, taking tiles straight from the matmul's output.
//
// The other half of the win is per-element cost. The stock scan is SCAN-bound rather than
// bandwidth-bound -- 8.4 MB in 0.45 ms is 18.6 GB/s, twenty times off what the fabric would give
// -- so what matters is instructions per logit, and two of the stock kernel's are removed here:
//   1. `bfloat16_greater` dispatches on the sign bits, up to three unpredictable branches per
//      element. `bf16_key` below is a branchless monotone remap, so one unsigned compare orders
//      any two values.
//   2. the stock loop carries a second `else if (val == max) idx = min(idx, i)` branch to keep the
//      lowest index among equal maxima. Scanning FORWARD with a strict `>` already keeps the
//      first maximum, so that branch is pure overhead.
// Two elements also come in per 32-bit load, and the scan pointer is deliberately NOT `volatile`
// (a compiler barrier after the read barrier gives the ordering without forcing every 2-byte load
// to be emitted separately and unscheduled).
//
// Instantiated TWICE per core, once on each data-movement RISC-V, over a half-open tile range
// each -- a scan-bound kernel that binds only the reader leaves the other processor idle, and the
// two ranges cover the work exactly once with no semaphore and no shared cursor.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Map a bf16 bit pattern to a uint16 whose UNSIGNED order is the numeric order of the bf16.
//   sign 0 (>= 0): flip the top bit,  [0x0000, 0x7FFF] -> [0x8000, 0xFFFF]
//   sign 1 (<  0): flip every bit,    [0x8000, 0xFFFF] -> [0x7FFF, 0x0000]
// Every negative key lands below every non-negative one and, within the negatives, a larger
// magnitude gives a smaller key. Exact and total on the values a logit can take, and it agrees
// with `bfloat16_greater` on the -0/+0 pair, so the sampled token is bit-identical.
FORCE_INLINE uint32_t bf16_key(uint32_t v) { return (v ^ (0x8000u | (0u - (v >> 15)))) & 0xFFFFu; }

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t part_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t tile_count = get_arg_val<uint32_t>(3);
    const uint32_t chunk = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t batch_tiles = get_compile_time_arg_val(2);
    // Rows of the batch actually in use. The logits are folded to `[1, batch, vocab]`, so the
    // whole batch is ONE tile row and a tile carries 32 of these rows side by side.
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    // Bytes per partial record. Sixteen, not eight: each record is written straight into the
    // destination page at `chunk * record_bytes`, and a NoC write wants a 16-byte-aligned offset.
    constexpr uint32_t record_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t part_page_bytes = get_compile_time_arg_val(5);

    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr auto part_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    const auto part = TensorAccessor(part_args, part_addr, part_page_bytes);

    const uint32_t scratch = get_write_ptr(cb_id);
    // One tile slot past the staging ring, reused as the outgoing record block (rows * 16 bytes,
    // which is well under a tile for any batch that fits a tile row).
    const uint32_t out_l1 = scratch + batch_tiles * tile_bytes;

    uint32_t best[32];
    uint32_t best_index[32];
    for (uint32_t r = 0; r < rows; ++r) {
        // Key 0 is the floor of the remap, and the seed index is the FIRST column this chunk
        // owns -- so a slice whose every value ties still reports its first element, which is the
        // same rule the scan itself follows.
        best[r] = 0;
        best_index[r] = tile_count == 0 ? 0 : start_tile * 32;
    }

    uint32_t done = 0;
    while (done < tile_count) {
        uint32_t run = tile_count - done;
        if (run > batch_tiles) {
            run = batch_tiles;
        }
        for (uint32_t i = 0; i < run; ++i) {
            noc_async_read_page(start_tile + done + i, src, scratch + i * tile_bytes);
        }
        noc_async_read_barrier();
        // Ordering without `volatile`: the barrier already made the data visible, this only stops
        // the compiler hoisting the loads above it.
        asm volatile("" ::: "memory");

        for (uint32_t i = 0; i < run; ++i) {
            const uint32_t tile_base = scratch + i * tile_bytes;
            const uint32_t tile_col0 = (start_tile + done + i) * 32;
            for (uint32_t r = 0; r < rows; ++r) {
                // A 32x32 tile is four 16x16 faces in the order (0,0) (0,1) (1,0) (1,1), so row r
                // is two contiguous 16-element runs: the left half in face `(r/16)*2` and the
                // right half in the face after it, both at element offset `(r%16)*16`.
                const uint32_t face = (r >> 4) * 2;
                const uint32_t row_off = (r & 15) * 16;
                uint32_t top = best[r];
                uint32_t top_index = best_index[r];
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t* p =
                        reinterpret_cast<const uint32_t*>(tile_base + (((face + half) << 8) + row_off) * 2);
                    const uint32_t col0 = tile_col0 + (half << 4);
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t w = p[j];
                        uint32_t k = bf16_key(w & 0xFFFFu);
                        if (k > top) {
                            top = k;
                            top_index = col0 + 2 * j;
                        }
                        k = bf16_key(w >> 16);
                        if (k > top) {
                            top = k;
                            top_index = col0 + 2 * j + 1;
                        }
                    }
                }
                best[r] = top;
                best_index[r] = top_index;
            }
        }
        done += run;
    }

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t r = 0; r < rows; ++r) {
        out[r * 4] = best[r];
        out[r * 4 + 1] = best_index[r];
    }
    // ONE PAGE PER ROW, so pass 2's fold for a row is a single transfer of that row's records
    // rather than a walk over every chunk's page. The rows are independent, so the writes from
    // different chunks never collide.
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_write_page(r, part, out_l1 + r * record_bytes, record_bytes, chunk * record_bytes);
    }
    noc_async_write_barrier();
}
