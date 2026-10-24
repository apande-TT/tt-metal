// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of tt/add_stats.py: for each of this core's tile rows the BLK-tile blocks of h = x + d, then the row's
// sum-of-squares tile. With L1_COPY each block is also copied into this core's shard of a height-sharded L1 copy of h
// (runtime arg 4: its address; the core's rows are exactly its shard).
// COMPACT_STATS: instead of the whole stats tile (sums in column 0, zeros elsewhere), only its column -- the 32 row
// sums -- is written, as row (r % 32) of compact stats tile r / 32 (two 64-byte face-row halves), so the norm's stats
// all_gather moves 1/32 of the bytes (tt/stats_expand.py rebuilds the gathered tiles).

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
#ifdef COMPACT_STATS
        {
            // column 0 of the fp32 stats tile: rows 0..15 at face 0 words 16 * i, rows 16..31 at face 2 (word 512)
            constexpr uint32_t cb_row = 19;
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_s));
            const uint32_t row_l1 = get_write_ptr(cb_row);
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
            for (uint32_t i = 0; i < 16; ++i) {
                dst[i] = src[i * 16];
                dst[16 + i] = src[512 + i * 16];
            }
            const uint32_t t = r / 32, c = r % 32, f = (c / 16) * 2, fr = c % 16;
            noc_async_write(row_l1, ss.get_noc_addr(t, f * 1024 + fr * 64), 64);
            noc_async_write(row_l1 + 64, ss.get_noc_addr(t, (f + 1) * 1024 + fr * 64), 64);
        }
#else
        noc_async_write_page(r, ss, get_read_ptr(cb_s));
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_s, 1);
    }
}
