// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill routed experts, combine (reader). Units of (token tile row, CW output tiles) are dealt round-robin over the
// cores, so the work splits evenly whatever the row count (whole rows left 18 of 110 cores doing two). For each row,
// cnt_j = how many local experts token j routed to; planes 0 .. max_j cnt_j - 1 of the fp32 partial buffer hold the
// row's partial sums. A unit's planes x CW tiles are read in ONE batch (one barrier), and while they land, one fp32
// mask tile per plane is built: row j = 1.0 where cnt_j > k, else 0. Rows a plane never received keep an earlier
// call's finite values (the buffer persists, zeroed once); compute multiplies them by the 0. Per plane, bit k of the
// `full` word says every row is valid, so compute skips its mask. Pushes are always KMAX * CW and KMAX tiles (the
// planes used first), so a reservation never wraps the circular buffer.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // output tiles per row
    constexpr uint32_t MT = get_compile_time_arg_val(1);  // token tile rows
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);  // cores
    constexpr uint32_t wrow_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t KMAX = get_compile_time_arg_val(5);  // most routed local experts a token can have
    constexpr uint32_t CW = get_compile_time_arg_val(6);    // output tiles per unit (divides Kt)
    constexpr auto w_args = TensorAccessorArgs<7>();
    constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto wr = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(1));
    const uint32_t core = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_p = 0, cb_cnt = 4, cb_mask = 5, cb_rows = 9;
    constexpr uint32_t face = 1024, tile = 4096;  // fp32 tile geometry
    constexpr uint32_t one = 0x3f800000u;         // 1.0f

    cb_reserve_back(cb_rows, 1);
    const uint32_t rows_l1 = get_write_ptr(cb_rows);
    uint32_t cnt[32];
    for (uint32_t unit = core; unit < MT * (Kt / CW); unit += NC) {
        const uint32_t r = unit / (Kt / CW), n0 = (unit % (Kt / CW)) * CW;
        for (uint32_t j = 0; j < 32; ++j) {
            noc_async_read_page(r * 32 + j, wr, rows_l1 + j * wrow_bytes);
        }
        noc_async_read_barrier();
        uint32_t planes = 1;
        for (uint32_t j = 0; j < 32; ++j) {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + j * wrow_bytes);
            uint32_t c = 0;
            for (uint32_t q = 0; q < E; ++q) {
                c += (w[q] & 0x7fffffffu) != 0;
            }
            cnt[j] = c;
            planes = c > planes ? c : planes;
        }
        planes = planes < KMAX ? planes : KMAX;

        cb_reserve_back(cb_p, KMAX * CW);
        const uint32_t p_l1 = get_write_ptr(cb_p);
        for (uint32_t k = 0; k < planes; ++k) {
            for (uint32_t c = 0; c < CW; ++c) {
                noc_async_read_page((k * MT + r) * Kt + n0 + c, y, p_l1 + (k * CW + c) * tile);
            }
        }

        cb_reserve_back(cb_mask, KMAX);
        const uint32_t m_l1 = get_write_ptr(cb_mask);
        uint32_t full = 0;
        for (uint32_t k = 0; k < planes; ++k) {
            uint32_t valid = 0;
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t v = cnt[j] > k ? one : 0u;
                valid += v != 0;
                volatile tt_l1_ptr uint32_t* p =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_l1 + k * tile + (j / 16) * 2 * face + (j % 16) * 64);
                for (uint32_t i = 0; i < 16; ++i) {
                    p[i] = v;
                    p[face / 4 + i] = v;
                }
            }
            full |= (valid == 32 ? 1u : 0u) << k;
        }
        cb_reserve_back(cb_cnt, 1);
        volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cnt));
        hdr[0] = planes;
        hdr[1] = full;
        cb_push_back(cb_cnt, 1);
        cb_push_back(cb_mask, KMAX);
        noc_async_read_barrier();
        cb_push_back(cb_p, KMAX * CW);
    }
}
