// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The output writer of the C++ decode attention scores (see tt/cpp_scores_dec.py): each unit's ST
// score tiles, in order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr uint32_t ST = get_compile_time_arg_val(0);
    constexpr auto ay = TensorAccessorArgs<1>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_y = 16;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        for (uint32_t j = 0; j < ST; ++j) {
            cb_wait_front(cb_y, 1);
            noc_async_write_page(u * ST + j, sy, get_read_ptr(cb_y));
            noc_async_write_barrier();
            cb_pop_front(cb_y, 1);
        }
    }
}
