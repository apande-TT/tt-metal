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
#ifdef FUSED_SUM
    constexpr uint32_t ND = get_compile_time_arg_val(2);  // devices gathered
    constexpr auto a_r = TensorAccessorArgs<3>();
#else
    constexpr auto a_r = TensorAccessorArgs<2>();
#endif
    const auto s_r = TensorAccessor(a_r, r_addr);

    cb_reserve_back(cb_x, ROWS * WT);
    cb_push_back(cb_x, ROWS * WT);

#ifdef FUSED_SUM
    // r is the all_gather of tt/add_stats' COMPACT stats (ND devices): row (k % 32) of its tile (k / 32, d) holds
    // device d's 32 row sums of tile row k. For each of this core's tile rows and each device, that 128-byte row
    // is dropped into column 0 of a tile in cb_r (rows 0..15 at face 0, 16..31 at face 2 -- the stats tile
    // tt/stats_expand.py rebuilds; the compute reads column 0 only, so the other columns are left as they are).
    constexpr uint32_t cb_rows = 3;
    const uint32_t rows_l1 = get_write_ptr(cb_rows);
    cb_reserve_back(cb_r, ROWS * ND);
    const uint32_t l1 = get_write_ptr(cb_r);
    for (uint32_t t = 0; t < ROWS; ++t) {
        const uint32_t k = t0 + t, c = k % 32, f = (c / 16) * 2, fr = c % 16;
        for (uint32_t d = 0; d < ND; ++d) {
            const uint32_t pg = (k / 32) * ND + d, j = t * ND + d;
            noc_async_read(s_r.get_noc_addr(pg, f * 1024 + fr * 64), rows_l1 + j * 128, 64);
            noc_async_read(s_r.get_noc_addr(pg, (f + 1) * 1024 + fr * 64), rows_l1 + j * 128 + 64, 64);
        }
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < ROWS * ND; ++j) {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + j * 128);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + j * 4096);
        for (uint32_t i = 0; i < 16; ++i) {
            dst[i * 16] = src[i];
            dst[512 + i * 16] = src[16 + i];
        }
    }
    cb_push_back(cb_r, ROWS * ND);
#else
    cb_reserve_back(cb_r, ROWS);
    const uint32_t l1 = get_write_ptr(cb_r);
    const uint32_t page = get_tile_size(cb_r);
    for (uint32_t t = 0; t < ROWS; ++t) {
        noc_async_read(s_r.get_noc_addr(t0 + t), l1 + t * page, page);
    }
    noc_async_read_barrier();
    cb_push_back(cb_r, ROWS);
#endif
}
