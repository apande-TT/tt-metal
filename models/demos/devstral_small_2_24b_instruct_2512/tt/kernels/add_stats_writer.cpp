// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of tt/add_stats.py: for each of this core's tile rows the BLK-tile blocks of h = x + d, then the row's
// sum-of-squares tile. With L1_COPY each block is also copied into this core's shard of a height-sharded L1 copy of h
// (runtime arg 4: its address; the core's rows are exactly its shard).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t h_addr = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t r0 = get_arg_val<uint32_t>(2);
    const uint32_t nr = get_arg_val<uint32_t>(3);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);
    constexpr auto ah = TensorAccessorArgs<2>();
    constexpr auto as = TensorAccessorArgs<ah.next_compile_time_args_offset()>();
    const auto sh = TensorAccessor(ah, h_addr);
    const auto ss = TensorAccessor(as, s_addr);

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_s = 17;
    const uint32_t h_bytes = get_tile_size(cb_out);
#ifdef L1_COPY
    uint32_t hl1 = get_arg_val<uint32_t>(4);
#endif

    for (uint32_t r = r0; r < r0 + nr; ++r) {
        for (uint32_t t = 0; t < WT; t += BLK) {
            cb_wait_front(cb_out, BLK);
            uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < BLK; ++j) {
                noc_async_write_page(r * WT + t + j, sh, l1);
                l1 += h_bytes;
            }
#ifdef L1_COPY
            noc_async_write(get_read_ptr(cb_out), get_noc_addr(hl1), BLK * h_bytes);
            hl1 += BLK * h_bytes;
#endif
            noc_async_write_barrier();
            cb_pop_front(cb_out, BLK);
        }
        cb_wait_front(cb_s, 1);
        noc_async_write_page(r, ss, get_read_ptr(cb_s));
        noc_async_write_barrier();
        cb_pop_front(cb_s, 1);
    }
}
