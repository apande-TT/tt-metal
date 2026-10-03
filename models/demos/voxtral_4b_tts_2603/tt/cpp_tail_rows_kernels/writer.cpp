// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the C++ tail-row regroup (see tt/cpp_tail_rows.py): unit u's DT bf8_b tiles to tile row
// (h * B + b) of its pair's [H * B, 1, R, D] output.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y0 = get_arg_val<uint32_t>(0);
    const uint32_t y1 = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t HB = get_compile_time_arg_val(1);  // H * B tile rows a pair
    constexpr auto acc0 = TensorAccessorArgs<2>();
    constexpr auto acc1 = TensorAccessorArgs<acc0.next_compile_time_args_offset()>();
    const auto s0 = TensorAccessor(acc0, y0);
    const auto s1 = TensorAccessor(acc1, y1);

    constexpr uint32_t cb_out = 16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t p = u / HB;
        const uint32_t row = u % HB;
        cb_wait_front(cb_out, DT);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < DT; ++c) {
            if (p == 0) {
                noc_async_write_page(row * DT + c, s0, l1);
            } else {
                noc_async_write_page(row * DT + c, s1, l1);
            }
            l1 += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, DT);
    }
}
