// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writes tiles from circular buffers to a list of places, in list order per buffer (see tiles_in.cpp).
// runtime args: n, then n x (cb, dst address, dst tile)
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto acc = TensorAccessorArgs<0>();
    const uint32_t n = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t cb = get_arg_val<uint32_t>(1 + 3 * i);
        const uint32_t page = get_tile_size(cb);
        cb_wait_front(cb, 1);
        noc_async_write(
            get_read_ptr(cb),
            TensorAccessor(acc, get_arg_val<uint32_t>(2 + 3 * i)).get_noc_addr(get_arg_val<uint32_t>(3 + 3 * i)),
            page);
        noc_async_write_barrier();
        cb_pop_front(cb, 1);
    }
}
