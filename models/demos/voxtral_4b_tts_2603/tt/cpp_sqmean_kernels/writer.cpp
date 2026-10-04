// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the C++ split square-mean (see tt/cpp_sqmean.py): per unit, the first 2 KB of the packed
// result tile (faces 0 and 1: the half's 16 row means in column 0) into its half of the output tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t k0 = get_arg_val<uint32_t>(1);
    const uint32_t nk = get_arg_val<uint32_t>(2);

    constexpr auto ay = TensorAccessorArgs<0>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_y = 16;
    constexpr uint32_t half_bytes = 2048;

    for (uint32_t k = k0; k < k0 + nk; ++k) {
        cb_wait_front(cb_y, 1);
        noc_async_write(get_read_ptr(cb_y), sy.get_noc_addr(k >> 1, (k & 1) * half_bytes), half_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_y, 1);
    }
}
