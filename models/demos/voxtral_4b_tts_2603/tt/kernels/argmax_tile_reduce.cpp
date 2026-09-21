// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PASS 2 of the greedy sampler: fold the per-chunk (key, index) records pass 1 wrote into one
// index per batch row.
//
// ONE CORE PER BATCH ROW, and a row's records are ONE page, so each core pulls its whole fold in
// a single transfer. Both matter: folding every row on one core turns a few microseconds of work
// into the same order of time as the scan itself, and reading the records chunk by chunk turns
// the transfer count into the cost. The rows are independent and each core owns one output page,
// so there is nothing to synchronise and no semaphore anywhere.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t part_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);

    constexpr uint32_t src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb = get_compile_time_arg_val(1);
    constexpr uint32_t chunks = get_compile_time_arg_val(2);
    constexpr uint32_t part_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(4);

    constexpr auto part_args = TensorAccessorArgs<5>();
    constexpr auto out_args = TensorAccessorArgs<part_args.next_compile_time_args_offset()>();

    const auto part = TensorAccessor(part_args, part_addr, part_page_bytes);
    const auto out = TensorAccessor(out_args, out_addr, out_page_bytes);

    const uint32_t scratch = get_write_ptr(src_cb);
    noc_async_read_page(row, part, scratch);
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    const uint32_t* p = reinterpret_cast<const uint32_t*>(scratch);
    uint32_t best = p[0];
    uint32_t best_index = p[1];
    for (uint32_t c = 1; c < chunks; ++c) {
        const uint32_t k = p[c * 4];
        // Pass 1 numbers chunks in ASCENDING vocab order, so a forward scan with a strict `>`
        // keeps the lowest index among equal maxima -- the same tie rule the within-chunk scan
        // uses, and the one `torch.argmax` and the stock op follow.
        if (k > best) {
            best = k;
            best_index = p[c * 4 + 1];
        }
    }

    const uint32_t out_l1 = get_write_ptr(dst_cb);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1) = best_index;
    noc_async_write_page(row, out, out_l1, out_page_bytes);
    noc_async_write_barrier();
}
