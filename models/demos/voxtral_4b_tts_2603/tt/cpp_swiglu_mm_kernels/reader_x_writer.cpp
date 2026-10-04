// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The activation stream and output writer of tt/cpp_swiglu_mm.py, on the other RISC / NoC from the weight
// stream: every core reads the whole [MT x KT] activation one K block (MT rows x KB tiles) at a time, then
// writes its MT x W/2 gated output tiles, a row at a time.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);  // first output column tile
    const uint32_t rev = get_arg_val<uint32_t>(3);  // K blocks last to first (the weight stream's order)

    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t PW = get_compile_time_arg_val(3);  // output tiles a row (W / 2)
    constexpr uint32_t KT = get_compile_time_arg_val(4);
    constexpr uint32_t NT = get_compile_time_arg_val(5);  // output column tiles
    constexpr auto ax = TensorAccessorArgs<6>();
    constexpr auto ay = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_y = 16;
    const uint32_t x_bytes = get_tile_size(cb_x);
    const uint32_t y_bytes = get_tile_size(cb_y);

    for (uint32_t i = 0; i < NB; ++i) {
        const uint32_t b = rev ? NB - 1 - i : i;
        cb_reserve_back(cb_x, MT * KB);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t r = 0; r < MT; ++r) {
            const uint32_t t = r * KT + b * KB;
            for (uint32_t k = 0; k < KB; ++k) {
                noc_async_read_page(t + k, sx, l1);
                l1 += x_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, MT * KB);
    }

    for (uint32_t m = 0; m < MT; ++m) {
        cb_wait_front(cb_y, PW);
        uint32_t l1 = get_read_ptr(cb_y);
        for (uint32_t j = 0; j < PW; ++j) {
            noc_async_write_page(m * NT + n0 + j, sy, l1);
            l1 += y_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, PW);
    }
}
