// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Writes the tiles of circular buffers, in order, as runs into other cores' L1 (a width-sharded tensor's shards).
// runtime args: n, then n x (cb, noc x, noc y, L1 address, tiles)
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(get_arg_addr(1 + 5 * i));
        const uint32_t cb = a[0], page = get_tile_size(cb);
        uint32_t addr = a[3];
        for (uint32_t t = 0; t < a[4]; ++t, addr += page) {
            cb_wait_front(cb, 1);
            noc_async_write(get_read_ptr(cb), get_noc_addr(a[1], a[2], addr), page);
            noc_async_writes_flushed();
            cb_pop_front(cb, 1);
        }
    }
    noc_async_write_barrier();
}
