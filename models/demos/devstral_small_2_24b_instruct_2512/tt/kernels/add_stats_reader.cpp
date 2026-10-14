// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of tt/add_stats.py: the unit reduce scaler once, then for each of this core's tile rows the fp32 residual
// x and its delta d, BLK tiles at a time.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_arg_val<uint32_t>(1);
    const uint32_t r0 = get_arg_val<uint32_t>(2);  // first tile row this core owns
    const uint32_t nr = get_arg_val<uint32_t>(3);  // tile rows this core owns

    constexpr uint32_t WT = get_compile_time_arg_val(0);   // tiles a row
    constexpr uint32_t BLK = get_compile_time_arg_val(1);  // tiles a block
    constexpr auto ax = TensorAccessorArgs<2>();
    constexpr auto ad = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sd = TensorAccessor(ad, d_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_d = 1;
    constexpr uint32_t cb_scaler = 2;
    const uint32_t x_bytes = get_tile_size(cb_x);
    const uint32_t d_bytes = get_tile_size(cb_d);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (uint32_t r = r0; r < r0 + nr; ++r) {
        for (uint32_t t = 0; t < WT; t += BLK) {
            cb_reserve_back(cb_x, BLK);
            cb_reserve_back(cb_d, BLK);
            uint32_t lx = get_write_ptr(cb_x);
            uint32_t ld = get_write_ptr(cb_d);
            for (uint32_t j = 0; j < BLK; ++j) {
                const uint32_t page = r * WT + t + j;
                noc_async_read_page(page, sx, lx);
                noc_async_read_page(page, sd, ld);
                lx += x_bytes;
                ld += d_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, BLK);
            cb_push_back(cb_d, BLK);
        }
    }
}
