// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reads a list of whole tiles into circular buffers, in list order per buffer; a tile may also be copied on to a
// second place (from L1, so the copy holds the value read even if that place is read here too). Every tensor is
// interleaved with the same page size, so one accessor configuration serves them all.
// runtime args: n, then n x (cb, src address, src tile, dst address or 0, dst tile)
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto acc = TensorAccessorArgs<0>();
    constexpr uint32_t MAX_CB = 8;
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t page = get_tile_size(0);

    uint32_t count[MAX_CB] = {0};
    for (uint32_t i = 0; i < n; ++i) {
        ++count[get_arg_val<uint32_t>(1 + 5 * i)];
    }
    uint32_t next[MAX_CB];
    for (uint32_t c = 0; c < MAX_CB; ++c) {
        if (count[c]) {
            cb_reserve_back(c, count[c]);
            next[c] = get_write_ptr(c);
        }
    }
    uint32_t l1[64];
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(get_arg_addr(1 + 5 * i));
        const uint32_t cb = a[0];
        l1[i] = next[cb];
        next[cb] += page;
        noc_async_read(TensorAccessor(acc, a[1]).get_noc_addr(a[2]), l1[i], page);
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(get_arg_addr(1 + 5 * i));
        if (a[3]) {
            noc_async_write(l1[i], TensorAccessor(acc, a[3]).get_noc_addr(a[4]), page);
        }
    }
    for (uint32_t c = 0; c < MAX_CB; ++c) {
        if (count[c]) {
            cb_push_back(c, count[c]);
        }
    }
    noc_async_write_barrier();
}
