// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of tt/norm_scale.py: hands the compute this core's x shard (the CB is the shard) and reads the core's ROWS
// factor tiles (tile rows t0 .. t0 + ROWS - 1 of r, one tile a tile row) into cb_r.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t r_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);  // this core's first tile row

    constexpr uint32_t ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_r = 1;
    constexpr auto a_r = TensorAccessorArgs<2>();
    const auto s_r = TensorAccessor(a_r, r_addr);

    cb_reserve_back(cb_x, ROWS * WT);
    cb_push_back(cb_x, ROWS * WT);

    cb_reserve_back(cb_r, ROWS);
    const uint32_t l1 = get_write_ptr(cb_r);
    const uint32_t page = get_tile_size(cb_r);
    for (uint32_t t = 0; t < ROWS; ++t) {
        noc_async_read(s_r.get_noc_addr(t0 + t), l1 + t * page, page);
    }
    noc_async_read_barrier();
    cb_push_back(cb_r, ROWS);
}
