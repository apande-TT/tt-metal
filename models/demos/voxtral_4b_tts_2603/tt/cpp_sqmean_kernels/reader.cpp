// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ split square-mean (see tt/cpp_sqmean.py). A unit is one half of one tile row:
// half 0 = the tile row's rows 0-15 (faces 0 and 1, the first 2 KB of each float32 tile), half 1 = rows
// 16-31 (faces 2 and 3, the last 2 KB). Each of the row's WT tiles contributes its half, landed in the first
// 2 KB of a CB slot -- the compute reads it as faces 0 and 1 of a tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t k0 = get_arg_val<uint32_t>(1);
    const uint32_t nk = get_arg_val<uint32_t>(2);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);  // tiles a read barrier (divides WT)
    constexpr auto ax = TensorAccessorArgs<2>();
    const auto sx = TensorAccessor(ax, x_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t half_bytes = 2048;
    constexpr uint32_t tile_bytes = 4096;

    for (uint32_t k = k0; k < k0 + nk; ++k) {
        const uint32_t row = k >> 1;
        const uint32_t off = (k & 1) * half_bytes;
        for (uint32_t w = 0; w < WT; w += BATCH) {
            cb_reserve_back(cb_x, BATCH);
            uint32_t l1 = get_write_ptr(cb_x);
            for (uint32_t i = 0; i < BATCH; ++i) {
                noc_async_read(sx.get_noc_addr(row * WT + w + i, off), l1, half_bytes);
                l1 += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, BATCH);
        }
    }
}
