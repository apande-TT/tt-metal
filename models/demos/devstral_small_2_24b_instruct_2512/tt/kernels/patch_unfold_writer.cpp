// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of tt/patch_unfold.py: each unit's 4 assembled output tiles (sub-positions j = 0..3 of output tile row R,
// channel tile c) to output column tiles j * DT + c of tile row img * RT + R.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t TILE = 2048;
    constexpr uint32_t cb_out = 16;
    constexpr auto y_args = TensorAccessorArgs<2>();

    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u % DT;
        const uint32_t row = u / DT;  // img * RT + R
        cb_wait_front(cb_out, 4);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_write_page(row * 4 * DT + j * DT + c, y, l1 + j * TILE);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 4);
    }
}
