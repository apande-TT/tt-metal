// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill routed experts, token-gathered (writer). Each output row of a unit of expert e belongs to one token t;
// it goes to row t of plane k of the fp32 [K, M, hidden] partial buffer, where k is e's rank among t's routed
// local experts (how many of them precede e). Ranks 0 .. cnt_t - 1 of every token are written exactly once, so
// the combine sums planes k < cnt_t and never reads a stale row. Rows move as two 64 B face rows per tile.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // output tiles per row
    constexpr uint32_t M = get_compile_time_arg_val(1);
    constexpr uint32_t wrow_bytes = get_compile_time_arg_val(2);  // one row of the [M, E] fp32 weights
    constexpr auto w_args = TensorAccessorArgs<3>();
    constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto wr = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(1));

    constexpr uint32_t cb_list = 5, cb_rows = 9, cb_y = 18;
    constexpr uint32_t MT = M / 32;
    constexpr uint32_t face = 1024, frow = 64, tile = 4096;  // fp32 tile geometry

    cb_wait_front(cb_list, 1);
    const uint32_t units = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_list));
    cb_pop_front(cb_list, 1);

    cb_reserve_back(cb_rows, 1);
    const uint32_t rows_l1 = get_write_ptr(cb_rows);
    uint32_t page_row[64];  // per token: plane-k tile row index (k * MT + t / 32) * Kt
    uint32_t dst_off[64];   // per token: offset of its row within a tile
    for (uint32_t i = 0; i < units; ++i) {
        cb_wait_front(cb_list, 1);
        volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_list));
        const uint32_t e = list[0], n = list[1];
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(list[2 + j], wr, rows_l1 + j * wrow_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < n; ++j) {
            volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + j * wrow_bytes);
            uint32_t k = 0;
            for (uint32_t q = 0; q < e; ++q) {
                k += (r[q] & 0x7fffffffu) != 0;
            }
            const uint32_t t = list[2 + j], rt = t % 32;
            page_row[j] = (k * MT + t / 32) * Kt;
            dst_off[j] = (rt / 16) * 2 * face + (rt % 16) * frow;
        }
        for (uint32_t nn = 0; nn < Kt; ++nn) {
            cb_wait_front(cb_y, 2);
            const uint32_t l1 = get_read_ptr(cb_y);
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t src = l1 + (j / 32) * tile + ((j % 32) / 16) * 2 * face + (j % 16) * frow;
                const uint32_t page = page_row[j] + nn;
                noc_async_write(src, y.get_noc_addr(page, dst_off[j]), frow);
                noc_async_write(src + face, y.get_noc_addr(page, dst_off[j] + face), frow);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_y, 2);
        }
        cb_pop_front(cb_list, 1);
    }
    noc_async_write_barrier();
}
