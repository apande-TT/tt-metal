// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The small-weight linear's reader: this core's tile rows of the bf16 input, one row of kt tiles at a time, into
// c_0; before the first row of each batch entry, that entry's whole bf16 weight block (kt x nt tiles, its pages
// in order) into c_1, held until the entry's rows are done (a weight shared by every row is one entry).
// Compile-time: the input's accessor args, the weight's, then kt, nt, the tile rows per entry and the weight
// pages per entry (0: one weight for all entries). Runtime: input addr, weight addr, first row, row count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t row_count = get_arg_val<uint32_t>(3);

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = w_args.next_compile_time_args_offset();
    constexpr uint32_t kt = get_compile_time_arg_val(base);
    constexpr uint32_t nt = get_compile_time_arg_val(base + 1);
    constexpr uint32_t rows_per_entry = get_compile_time_arg_val(base + 2);
    constexpr uint32_t w_entry_pages = get_compile_time_arg_val(base + 3);
    const auto a = TensorAccessor(a_args, a_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb_a(0), cb_w(1);
    uint32_t cur = 0xFFFFFFFF;
    for (uint32_t r = row_start; r < row_start + row_count; ++r) {
        const uint32_t e = r / rows_per_entry;
        if (e != cur) {
            cb_w.reserve_back(kt * nt);  // (free once the compute has released the previous entry's block)
            for (uint32_t t = 0; t < kt * nt; ++t) {
                noc.async_read(w, cb_w, bytes, {.page_id = e * w_entry_pages + t}, {.offset_bytes = t * bytes});
            }
            noc.async_read_barrier();
            cb_w.push_back(kt * nt);
            cur = e;
        }
        cb_a.reserve_back(kt);
        for (uint32_t k = 0; k < kt; ++k) {
            noc.async_read(a, cb_a, bytes, {.page_id = r * kt + k}, {.offset_bytes = k * bytes});
        }
        noc.async_read_barrier();
        cb_a.push_back(kt);
    }
}
