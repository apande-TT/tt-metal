// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// masked_lane_bmm's reader: masked_mm_reader.cpp for a batched product whose b differs per entry. The 8 column
// masks into c_2 once; then per tile row of a (one at a time: each has its own b entry), per K tile, a's tile into
// c_0 and b's nt tiles of that K row into c_1. Row `row` belongs to entry row / mte, whose b entry is that / b_rep.
// Compile-time: a, b, masks accessor args, kt, nt, mte, b_rep. Runtime: a, b, masks addr, first row, row count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t m_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = m_args.next_compile_time_args_offset();
    constexpr uint32_t kt = get_compile_time_arg_val(base);
    constexpr uint32_t nt = get_compile_time_arg_val(base + 1);
    constexpr uint32_t mte = get_compile_time_arg_val(base + 2);
    constexpr uint32_t b_rep = get_compile_time_arg_val(base + 3);
    constexpr uint32_t lanes = 8;
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);
    const auto masks = TensorAccessor(m_args, m_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb_a(0), cb_b(1), cb_m(2);
    cb_m.reserve_back(lanes);
    for (uint32_t r = 0; r < lanes; ++r) {
        noc.async_read(masks, cb_m, bytes, {.page_id = r}, {.offset_bytes = r * bytes});
    }
    noc.async_read_barrier();
    cb_m.push_back(lanes);

    for (uint32_t m = 0; m < rows; ++m) {
        const uint32_t row = row_start + m;
        const uint32_t b_base = (row / mte / b_rep) * kt * nt;
        for (uint32_t k = 0; k < kt; ++k) {
            cb_a.reserve_back(1);
            noc.async_read(a, cb_a, bytes, {.page_id = row * kt + k}, {.offset_bytes = 0});
            cb_b.reserve_back(nt);
            for (uint32_t j = 0; j < nt; ++j) {
                noc.async_read(b, cb_b, bytes, {.page_id = b_base + k * nt + j}, {.offset_bytes = j * bytes});
            }
            noc.async_read_barrier();
            cb_a.push_back(1);
            cb_b.push_back(nt);
        }
    }
}
