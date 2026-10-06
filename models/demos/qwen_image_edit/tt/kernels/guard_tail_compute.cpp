// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The guarded exact-lane tail of the precise linears, one float32 tile at a time:
//
//     y = where(|ex - dn| > tol, median(ex, dn, -nn), ex) + lo
//
// with the same SFPU calls ttnn's float32 binary_ng / ternary ops make for the ttnn spelling (sub, abs,
// gt, min, max, negative, where<Float32>, add), on operands unpacked straight to the float32 DST, so the
// result is bit-identical to it. Seven DST slots: dst_full_sync_en gives eight float32 tiles.

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_ex = 0, cb_dn = 1, cb_nn = 2, cb_tol = 3, cb_lo = 4, cb_out = 16;
    // DST slots
    constexpr uint32_t EX = 0, DN = 1, NN = 2, TOL = 3, LO = 4, FAR = 5, TMP = 6;

    compute_kernel_hw_startup(cb_ex, cb_out);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_ex, 1);
        cb_wait_front(cb_dn, 1);
        cb_wait_front(cb_nn, 1);
        cb_wait_front(cb_tol, 1);
        cb_wait_front(cb_lo, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_init(cb_ex);
        copy_tile(cb_ex, 0, EX);
        copy_init(cb_dn);
        copy_tile(cb_dn, 0, DN);
        copy_init(cb_nn);
        copy_tile(cb_nn, 0, NN);
        copy_init(cb_tol);
        copy_tile(cb_tol, 0, TOL);
        copy_init(cb_lo);
        copy_tile(cb_lo, 0, LO);

        sub_binary_tile_init();
        sub_binary_tile(EX, DN, FAR);  // ex - dn
        abs_tile_init();
        abs_tile(FAR);  // |ex - dn|
        gt_binary_tile_init();
        gt_binary_tile(FAR, TOL, FAR);  // 1 where |ex - dn| > tol (the guard trips), else 0
        binary_min_tile_init();
        binary_min_tile(EX, DN, TOL);  // lo = min(ex, dn)       (tol no longer needed)
        binary_max_tile_init();
        binary_max_tile(EX, DN, TMP);  // hi = max(ex, dn)
        negative_tile_init();
        negative_tile(NN);  // -nn
        binary_min_tile_init();
        binary_min_tile(TMP, NN, TMP);  // min(hi, -nn)
        binary_max_tile_init();
        binary_max_tile(TOL, TMP, TOL);  // median(ex, dn, -nn)
        where_tile_init();
        where_tile<DataFormat::Float32>(FAR, TOL, EX, EX);  // far ? median : ex
        add_binary_tile_init();
        add_binary_tile(EX, LO, EX);  // + lo
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(EX, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_ex, 1);
        cb_pop_front(cb_dn, 1);
        cb_pop_front(cb_nn, 1);
        cb_pop_front(cb_tol, 1);
        cb_pop_front(cb_lo, 1);
    }
}
