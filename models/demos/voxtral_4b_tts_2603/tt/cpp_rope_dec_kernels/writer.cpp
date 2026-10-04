// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The output stream of the C++ decode RoPE (see tt/cpp_rope_dec.py): this core's tile rows of y.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t nrows = get_arg_val<uint32_t>(2);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr auto ay = TensorAccessorArgs<1>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_y = 16;
    const uint32_t bytes = get_tile_size(cb_y);
    for (uint32_t r = row0; r < row0 + nrows; ++r) {
        cb_wait_front(cb_y, DT);
        uint32_t l1 = get_read_ptr(cb_y);
        for (uint32_t c = 0; c < DT; ++c) {
            noc_async_write_page(r * DT + c, sy, l1);
            l1 += bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, DT);
    }
}
