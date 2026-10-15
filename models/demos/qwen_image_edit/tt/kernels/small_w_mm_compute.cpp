// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The small-weight linear: each row of kt input tiles (c_0) against its batch entry's whole weight block held in
// c_1 (kt x nt tiles row-major, or nt x kt with each tile transposed when TRANSPOSE_B), the K reduction
// accumulated in the float32 DST (k ascending, as the stock matmul's single K block), `chunk` output tiles per
// DST pass, packed float32 to c_16 in row-major tile order. The weight block is waited for at each entry's first
// row and released after its last.
// Compile-time: kt, nt, chunk (divides nt, at most the 8 float32 DST tiles), tile rows per entry, transpose_b.
// Runtime: first row, row count.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t row_start = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t chunk = get_compile_time_arg_val(2);
    constexpr uint32_t rows_per_entry = get_compile_time_arg_val(3);
    constexpr uint32_t transpose_b = get_compile_time_arg_val(4);
    constexpr uint32_t cb_a = 0, cb_w = 1, cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);
    matmul_init(cb_a, cb_w, transpose_b);
    uint32_t cur = 0xFFFFFFFF;
    for (uint32_t r = row_start; r < row_start + rows; ++r) {
        const uint32_t e = r / rows_per_entry;
        if (e != cur) {
            if (cur != 0xFFFFFFFF) {
                cb_pop_front(cb_w, kt * nt);
            }
            cb_wait_front(cb_w, kt * nt);
            cur = e;
        }
        cb_wait_front(cb_a, kt);
        for (uint32_t n0 = 0; n0 < nt; n0 += chunk) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < kt; ++k) {
                for (uint32_t j = 0; j < chunk; ++j) {
                    const uint32_t w_tile = transpose_b ? (n0 + j) * kt + k : k * nt + n0 + j;
                    matmul_tiles(cb_a, cb_w, k, w_tile, j);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, chunk);
            tile_regs_wait();
            for (uint32_t j = 0; j < chunk; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, chunk);
        }
        cb_pop_front(cb_a, kt);
    }
    if (cur != 0xFFFFFFFF) {
        cb_pop_front(cb_w, kt * nt);
    }
}
