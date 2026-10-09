// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode gate/up over the ROUTED experts only (reader). The [1, E] mask says which local experts any of
// the step's tokens routed to; their (expert, column) weight tiles are enumerated in expert order and dealt
// round-robin over the cores, so every core streams an equal share of the weights that matter and none of
// the rest. Hands the column list to the writer and the count to compute.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);           // K tiles
    constexpr uint32_t Nt = get_compile_time_arg_val(1);           // N tiles of one weight row
    constexpr uint32_t E = get_compile_time_arg_val(2);            // local experts
    constexpr uint32_t CPE = get_compile_time_arg_val(3);          // N tiles per expert
    constexpr uint32_t NC = get_compile_time_arg_val(4);           // cores
    constexpr uint32_t KB = get_compile_time_arg_val(5);           // K tiles per weight push
    constexpr uint32_t MAX_SLOTS = get_compile_time_arg_val(6);    // max columns one core can own
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto wg_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wu_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto wg = TensorAccessor(wg_args, get_arg_val<uint32_t>(1));
    const auto wu = TensorAccessor(wu_args, get_arg_val<uint32_t>(2));
    const auto mask_acc = TensorAccessor(m_args, get_arg_val<uint32_t>(3));
    const uint32_t core_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = 0, cb_g = 1, cb_u = 2, cb_mask = 3, cb_count = 4, cb_list = 5;

    cb_reserve_back(cb_mask, 1);
    const uint32_t mask_l1 = get_write_ptr(cb_mask);
    noc_async_read_page(0, mask_acc, mask_l1);

    cb_reserve_back(cb_x, Kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < Kt; ++k) {
        noc_async_read_page(k, x, x_l1 + k * x_tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, Kt);

    // An expert is active when its mask word is non-zero (+-0.0 both count as inactive).
    volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1);
    uint32_t cols[MAX_SLOTS];
    uint32_t n = 0, turn = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if ((mask[e] & 0x7fffffffu) == 0) {
            continue;
        }
        for (uint32_t j = 0; j < CPE; ++j) {
            if (turn == core_id) {
                cols[n++] = e * CPE + j;
            }
            if (++turn == NC) {
                turn = 0;
            }
        }
    }

    cb_reserve_back(cb_list, 1);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_list));
    list[0] = n;
    for (uint32_t s = 0; s < n; ++s) {
        list[1 + s] = cols[s];
    }
    cb_push_back(cb_list, 1);
    cb_reserve_back(cb_count, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_count)) = n;
    cb_push_back(cb_count, 1);

    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t col = cols[s];
        for (uint32_t kb = 0; kb < Kt; kb += KB) {
            cb_reserve_back(cb_g, KB);
            cb_reserve_back(cb_u, KB);
            const uint32_t g_l1 = get_write_ptr(cb_g);
            const uint32_t u_l1 = get_write_ptr(cb_u);
            for (uint32_t k = 0; k < KB; ++k) {
                const uint32_t page = (kb + k) * Nt + col;
                noc_async_read_page(page, wg, g_l1 + k * w_tile_bytes);
                noc_async_read_page(page, wu, u_l1 + k * w_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_g, KB);
            cb_push_back(cb_u, KB);
        }
    }
}
