// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PASS 2 of the greedy-sampling argmax: fold the per-chunk (key, index) pairs pass 1 produced into
// one index per batch row.
//
// ONE CORE PER BATCH ROW, and the whole partial buffer is a SINGLE page so each core pulls it in
// one transfer.  Both of those matter: an earlier revision ran the entire fold on one core and read
// the partials page-by-page, and at a full grid (two chunks per core, so ~256 chunks) that came to
// ~256 serial NoC requests plus a ~2000-entry scan -- tens of microseconds, which on this op is the
// same order as pass 1's actual scan.  Splitting by row makes each core's fold `nparts` entries
// instead of `nparts * rows`, and the single-page read collapses the request count to one.
//
// Still no semaphores anywhere: the rows are independent and each core owns one output page, so
// there is nothing to synchronise and nothing to deadlock.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#include <stdint.h>

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);

    constexpr uint32_t src_cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    constexpr uint32_t nparts = get_compile_time_arg_val(3);

    constexpr auto s_src_args = TensorAccessorArgs<4>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();

    const auto s_src = TensorAccessor(s_src_args, src_addr);
    const auto s_dst = TensorAccessor(s_dst_args, dst_addr);

    Noc noc;
    CircularBuffer src_cb(src_cb_idx);
    CircularBuffer dst_cb(dst_cb_idx);

    // The partials are one page of nparts * rows * 2 uint32s: [chunk][row][key, index].
    constexpr uint32_t all_bytes = nparts * rows * 8;

    const uint32_t base = src_cb.get_write_ptr();
    noc.async_read(s_src, src_cb, all_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const uint32_t stride = 2 * rows;  // uint32s per chunk
    uint32_t o = 2 * row;

    uint32_t best = p[o];
    uint32_t best_i = p[o + 1];
    for (uint32_t c = 1; c < nparts; ++c) {
        o += stride;
        const uint32_t k = p[o];
        // Pass 1 numbers chunks in ASCENDING vocab order, so a forward scan with a STRICT `>`
        // keeps the lowest index among equal maxima -- the same tie rule the within-chunk scan
        // uses, and the one torch.argmax and the stock op follow.
        if (k > best) {
            best = k;
            best_i = p[o + 1];
        }
    }

    const uint32_t out_addr = dst_cb.get_write_ptr();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr) = best_i;
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(dst_cb), s_dst, 4, {.offset_bytes = 0}, {.page_id = row});
    noc.async_write_barrier();
}
