// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// guard_tail_compute.cpp with its tolerance formed in DST from the two broadcast norm tiles:
//
//     tol = a * b                      (c_3: a's column-broadcast tile, held for a whole tile row; c_5: b's row tile)
//     y = where(|ex - dn| > tol, median(ex, dn, -nn), ex) + lo
//
// tol is the same SFPU float32 multiply ttnn's binary_ng makes for the [.., M, 1] x [.., 1, N] outer product, on
// the same values, so y is bit-identical to guard_tail fed that product.
// Runtime: tile count, first tile index, Nt (a tile row starts at every multiple of Nt, and at the first tile).

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
    const uint32_t count = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    // 0: no negated-input product -- where the guard trips the tail takes dn (the dense product) for ex
    constexpr uint32_t has_nn = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ex = 0, cb_dn = 1, cb_nn = 2, cb_a = 3, cb_lo = 4, cb_b = 5, cb_out = 16;
    constexpr uint32_t EX = 0, DN = 1, NN = 2, TOL = 3, LO = 4, FAR = 5, TMP = 6, BN = 7;

    compute_kernel_hw_startup(cb_ex, cb_out);
    for (uint32_t i = start; i < start + count; ++i) {
        const uint32_t c = i % nt;
        if (i == start || c == 0) {
            cb_wait_front(cb_a, 1);
        }
        cb_wait_front(cb_ex, 1);
        cb_wait_front(cb_dn, 1);
        if constexpr (has_nn) {
            cb_wait_front(cb_nn, 1);
        }
        cb_wait_front(cb_lo, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_init(cb_ex);
        copy_tile(cb_ex, 0, EX);
        copy_init(cb_dn);
        copy_tile(cb_dn, 0, DN);
        if constexpr (has_nn) {
            copy_init(cb_nn);
            copy_tile(cb_nn, 0, NN);
        }
        copy_init(cb_a);
        copy_tile(cb_a, 0, TOL);
        copy_init(cb_b);
        copy_tile(cb_b, 0, BN);
        copy_init(cb_lo);
        copy_tile(cb_lo, 0, LO);

        mul_binary_tile_init();
        mul_binary_tile(TOL, BN, TOL);  // tol = a * b
        sub_binary_tile_init();
        sub_binary_tile(EX, DN, FAR);  // ex - dn
        abs_tile_init();
        abs_tile(FAR);  // |ex - dn|
        gt_binary_tile_init();
        gt_binary_tile(FAR, TOL, FAR);  // 1 where |ex - dn| > tol (the guard trips), else 0
        if constexpr (has_nn) {
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
        } else {
            where_tile_init();
            where_tile<DataFormat::Float32>(FAR, DN, EX, EX);  // far ? dn : ex
        }
        add_binary_tile_init();
        add_binary_tile(EX, LO, EX);  // + lo
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(EX, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_ex, 1);
        cb_pop_front(cb_dn, 1);
        if constexpr (has_nn) {
            cb_pop_front(cb_nn, 1);
        }
        cb_pop_front(cb_lo, 1);
        cb_pop_front(cb_b, 1);
        if (i + 1 == start + count || c + 1 == nt) {
            cb_pop_front(cb_a, 1);
        }
    }
}
