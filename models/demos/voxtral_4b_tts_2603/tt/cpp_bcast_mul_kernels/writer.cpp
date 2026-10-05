// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the C++ column-broadcast multiply (see tt/cpp_bcast_mul.py): a core's output tiles, CH a
// chunk with one barrier, to the same tile ids it read.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t CH = get_compile_time_arg_val(0);
    constexpr auto ay = TensorAccessorArgs<1>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_y = 16;
    const uint32_t y_bytes = get_tile_size(cb_y);

    for (uint32_t c0 = 0; c0 < nt; c0 += CH) {
        const uint32_t n = (nt - c0 < CH) ? (nt - c0) : CH;
        cb_wait_front(cb_y, CH);
        uint32_t l1 = get_read_ptr(cb_y);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_write_page(t0 + c0 + j, sy, l1);
            l1 += y_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, CH);
    }
}
