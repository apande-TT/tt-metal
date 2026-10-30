// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row sums of squares of a bf16 tile tensor, float32 out: per tile row, each of its Kt tiles is squared on the
// SFPU (c_0 -> DST -> c_2, rounded to bf16) and row-reduced into a float32 running sum (c_3, reloaded exactly
// with UnpackToDestFp32); the last tile's sum goes to c_16 (column 0 of the output tile). The square + reduce
// round trip is the stock reduce_w_neg pattern with the negation replaced by a square.
// Compile-time: Kt. Runtime: tile row count.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_sq = 2, cb_acc = 3, cb_out = 16;
    constexpr uint32_t DST = 0;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < kt; ++k) {
            // square: c_0 (bf16) -> c_2 (bf16)
            cb_wait_front(cb_in, 1);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_in);
            copy_init(cb_in);
            copy_tile(cb_in, 0, DST);
            square_tile_init();
            square_tile(DST);
            tile_regs_commit();
            cb_pop_front(cb_in, 1);
            cb_reserve_back(cb_sq, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_sq);
            pack_tile(DST, cb_sq);
            tile_regs_release();
            cb_push_back(cb_sq, 1);

            // running row sum: (k > 0: the previous sum from c_3) + reduce_row(c_2) -> c_3, or c_16 at the last tile
            const uint32_t dst_cb = k + 1 == kt ? cb_out : cb_acc;
            tile_regs_acquire();
            if (k > 0) {
                cb_wait_front(cb_acc, 1);
                reconfig_data_format_srca(cb_acc);
                copy_init(cb_acc);
                copy_tile(cb_acc, 0, DST);
            }
            cb_wait_front(cb_sq, 1);
            reconfig_data_format(cb_scaler, cb_sq);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, dst_cb);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, 0, 0, DST);
            reduce_uninit();
            tile_regs_commit();
            cb_pop_front(cb_sq, 1);
            if (k > 0) {
                cb_pop_front(cb_acc, 1);
            }
            cb_reserve_back(dst_cb, 1);
            tile_regs_wait();
            pack_reconfig_data_format(dst_cb);
            pack_tile(DST, dst_cb);
            tile_regs_release();
            cb_push_back(dst_cb, 1);
        }
    }
}
