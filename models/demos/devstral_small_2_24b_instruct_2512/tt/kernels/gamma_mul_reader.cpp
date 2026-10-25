// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of tt/gamma_mul.py: this core's GW gamma tiles (columns tw0 .. tw0 + GW - 1, one at a time into cb_g for the
// compute to broadcast), then for each tile row th0 .. th0 + nrows - 1 the row's GW y tiles of those columns.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t th0 = get_arg_val<uint32_t>(2);
    const uint32_t nrows = get_arg_val<uint32_t>(3);
    const uint32_t tw0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t GW = get_compile_time_arg_val(1);
    constexpr auto a_y = TensorAccessorArgs<2>();
    constexpr auto a_g = TensorAccessorArgs<a_y.next_compile_time_args_offset()>();
    const auto s_y = TensorAccessor(a_y, y_addr);
    const auto s_g = TensorAccessor(a_g, g_addr);
    constexpr uint32_t cb_y = 0, cb_g = 1;
    const uint32_t page = get_tile_size(cb_y);

    for (uint32_t w = 0; w < GW; ++w) {
        cb_reserve_back(cb_g, 1);
        noc_async_read(s_g.get_noc_addr(tw0 + w), get_write_ptr(cb_g), page);
        noc_async_read_barrier();
        cb_push_back(cb_g, 1);
    }
    for (uint32_t r = 0; r < nrows; ++r) {
        cb_reserve_back(cb_y, GW);
        const uint32_t l1 = get_write_ptr(cb_y);
        const uint32_t p0 = (th0 + r) * WT + tw0;
        for (uint32_t j = 0; j < GW; ++j) {
            noc_async_read(s_y.get_noc_addr(p0 + j), l1 + j * page, page);
        }
        noc_async_read_barrier();
        cb_push_back(cb_y, GW);
    }
}
