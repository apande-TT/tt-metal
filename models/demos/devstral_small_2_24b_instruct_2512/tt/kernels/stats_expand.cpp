// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt/stats_expand.py: rebuilds the norm's gathered stats tensor (..., S, ND * 32) -- tile (k, d) is device d's stats
// tile k: its 32 row sums in column 0, exact zeros elsewhere -- from the all_gather of the compacted stats, where row
// (k % 32) of tile (k / 32, d) holds those 32 sums. This core writes output pages [p0, p0 + np): every page is a
// staging tile that stays zero outside column 0 (zeroed once) with the page's 32 sums dropped into column 0.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t g_addr = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t p0 = get_arg_val<uint32_t>(2);
    const uint32_t np = get_arg_val<uint32_t>(3);

    constexpr uint32_t ND = get_compile_time_arg_val(0);    // devices gathered (tile columns)
    constexpr uint32_t NMAX = get_compile_time_arg_val(1);  // most pages a core writes
    constexpr auto a_g = TensorAccessorArgs<2>();
    constexpr auto a_o = TensorAccessorArgs<a_g.next_compile_time_args_offset()>();
    const auto s_g = TensorAccessor(a_g, g_addr);
    const auto s_o = TensorAccessor(a_o, o_addr);

    constexpr uint32_t cb_rows = 0, cb_stage = 1;  // NMAX gathered rows (128 B each); two zeroed staging tiles
    constexpr uint32_t TILE_BYTES = 4096, ROW_BYTES = 128;
    const uint32_t rows_l1 = get_write_ptr(cb_rows);
    const uint32_t stage_l1 = get_write_ptr(cb_stage);

    // every page's 32 sums: row c = k % 32 of gathered tile (k / 32, d), two 64-byte face-row halves
    for (uint32_t j = 0; j < np; ++j) {
        const uint32_t p = p0 + j, k = p / ND, d = p % ND;
        const uint32_t t = k / 32, c = k % 32, f = (c / 16) * 2, fr = c % 16;
        const uint32_t page = t * ND + d;
        noc_async_read(s_g.get_noc_addr(page, f * 1024 + fr * 64), rows_l1 + j * ROW_BYTES, 64);
        noc_async_read(s_g.get_noc_addr(page, (f + 1) * 1024 + fr * 64), rows_l1 + j * ROW_BYTES + 64, 64);
    }
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_l1);
    for (uint32_t w = 0; w < 2 * TILE_BYTES / 4; ++w) {
        stage[w] = 0;
    }
    noc_async_read_barrier();

    for (uint32_t j = 0; j < np; ++j) {
        const uint32_t buf = stage_l1 + (j & 1) * TILE_BYTES;
        if (j >= 2) {
            noc_async_writes_flushed();  // (buffer j & 1 was last sent two pages ago)
        }
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + j * ROW_BYTES);
        // column 0 of an fp32 tile: rows 0..15 at face 0 words 16 * i, rows 16..31 at face 2 (word 512)
        for (uint32_t i = 0; i < 16; ++i) {
            dst[i * 16] = src[i];
            dst[512 + i * 16] = src[16 + i];
        }
        noc_async_write(buf, s_o.get_noc_addr(p0 + j), TILE_BYTES);
    }
    noc_async_write_barrier();
}
