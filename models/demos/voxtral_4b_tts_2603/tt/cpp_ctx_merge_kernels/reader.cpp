// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ decode context merge (see tt/cpp_ctx_merge.py). Unit u = output tile (batch tile
// row bt, column tile c) of the merged [1, 1, B, KV * G * D] operand: column tile c is query row r = (c / DT) % G
// of kv head g = c / (DT * G), d-tile dt = c % DT. Output row b (= 32 bt + i) is P@V row r of (b, g) -- two
// 64 B face-row reads from tile (b, g, 0, dt) -- and its divisor is that row's sum, column 0 of row r of the
// row-sum tile (b, g), filled across the row (binary_ng's reader-side column broadcast). Rows b >= B are zero.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t pv_addr = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t B = get_compile_time_arg_val(0);   // batch rows
    constexpr uint32_t KV = get_compile_time_arg_val(1);  // kv heads
    constexpr uint32_t G = get_compile_time_arg_val(2);   // query rows a kv head
    constexpr uint32_t DT = get_compile_time_arg_val(3);  // d tiles a head
    // PACKED: the row sums are the packed [B, 1, 32, 1] (kv head g's G rows at rows g * G ..; see cpp_scores_dec).
    constexpr uint32_t PACKED = get_compile_time_arg_val(4);
    constexpr auto apv = TensorAccessorArgs<5>();
    constexpr auto as = TensorAccessorArgs<apv.next_compile_time_args_offset()>();
    const auto spv = TensorAccessor(apv, pv_addr);
    const auto ss = TensorAccessor(as, s_addr);
    constexpr uint32_t CT = KV * G * DT;  // output column tiles

    constexpr uint32_t cb_num = 0;
    constexpr uint32_t cb_den = 1;
    constexpr uint32_t cb_scratch = 2;
    constexpr uint32_t face = 1024;
    constexpr uint32_t seg = 64;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // 32 row-sum segments

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u % CT;
        const uint32_t bt = u / CT;
        const uint32_t g = c / (DT * G);
        const uint32_t r = (c / DT) % G;
        const uint32_t dt = c % DT;
        const uint32_t rf = (r >> 4) * 2;  // the face pair holding row r
        const uint32_t ro = (r & 15) * seg;

        cb_reserve_back(cb_num, 1);
        cb_reserve_back(cb_den, 1);
        const uint32_t num = get_write_ptr(cb_num);
        const uint32_t den = get_write_ptr(cb_den);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t b = 32 * bt + i;
            const uint32_t drow = (i >> 4) * 2 * face + (i & 15) * seg;
            if (b < B) {
                const uint32_t bg = b * KV + g;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(spv.get_noc_addr(bg * DT + dt, (rf + f) * face + ro), num + drow + f * face, seg);
                }
                if constexpr (PACKED) {
                    const uint32_t R = g * G + r;
                    noc_async_read(ss.get_noc_addr(b, (R >> 4) * 2 * face + (R & 15) * seg), scratch + i * seg, seg);
                } else {
                    noc_async_read(ss.get_noc_addr(bg, rf * face + ro), scratch + i * seg, seg);
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t b = 32 * bt + i;
            const uint32_t drow = (i >> 4) * 2 * face + (i & 15) * seg;
            volatile tt_l1_ptr uint32_t* nrow0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num + drow);
            volatile tt_l1_ptr uint32_t* nrow1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num + drow + face);
            volatile tt_l1_ptr uint32_t* drow0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(den + drow);
            volatile tt_l1_ptr uint32_t* drow1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(den + drow + face);
            if (b < B) {
                const uint32_t v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + i * seg)[0];
                for (uint32_t w = 0; w < 16; ++w) {
                    drow0[w] = v;
                    drow1[w] = v;
                }
            } else {
                // A padding row: 0 / 1 (no NaN in the pad).
                for (uint32_t w = 0; w < 16; ++w) {
                    nrow0[w] = 0;
                    nrow1[w] = 0;
                    drow0[w] = 0x3f800000;
                    drow1[w] = 0x3f800000;
                }
            }
        }
        cb_push_back(cb_num, 1);
        cb_push_back(cb_den, 1);
    }
}
