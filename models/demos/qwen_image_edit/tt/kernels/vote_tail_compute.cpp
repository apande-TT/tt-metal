// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The VAE encoder's exact-conv vote tail, one float32 tile at a time:
//
//     e_x = acc - bn;  y = where(|e_x - e_neg| > tol, median(e_x, f0, e_neg), e_x)
//
// with the same SFPU calls ttnn's float32 binary_ng / ternary ops make for the ttnn spelling in
// precise_affine (sub, abs, gt, min, max, where<Float32>), on operands unpacked straight to the float32 DST,
// so the result is bit-identical to it. Seven DST slots: dst_full_sync_en gives eight float32 tiles.

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
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
    constexpr uint32_t cb_acc = 0, cb_bn = 1, cb_f0 = 2, cb_en = 3, cb_tol = 4, cb_out = 16;
    // DST slots
    constexpr uint32_t EX = 0, BN = 1, F0 = 2, EN = 3, TOL = 4, FAR = 5, TMP = 6;

    compute_kernel_hw_startup(cb_acc, cb_out);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_acc, 1);
        cb_wait_front(cb_bn, 1);
        cb_wait_front(cb_f0, 1);
        cb_wait_front(cb_en, 1);
        cb_wait_front(cb_tol, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_init(cb_acc);
        copy_tile(cb_acc, 0, EX);
        copy_init(cb_bn);
        copy_tile(cb_bn, 0, BN);
        copy_init(cb_f0);
        copy_tile(cb_f0, 0, F0);
        copy_init(cb_en);
        copy_tile(cb_en, 0, EN);
        copy_init(cb_tol);
        copy_tile(cb_tol, 0, TOL);

        sub_binary_tile_init();
        sub_binary_tile(EX, BN, EX);   // e_x = acc - bn
        sub_binary_tile(EX, EN, FAR);  // e_x - e_neg
        abs_tile_init();
        abs_tile(FAR);  // |e_x - e_neg|
        gt_binary_tile_init();
        gt_binary_tile(FAR, TOL, FAR);  // 1 where |e_x - e_neg| > tol (the vote overrides), else 0
        binary_min_tile_init();
        binary_min_tile(EX, F0, BN);  // min(e_x, f0)        (bn no longer needed)
        binary_max_tile_init();
        binary_max_tile(EX, F0, TMP);  // max(e_x, f0)
        binary_min_tile_init();
        binary_min_tile(TMP, EN, TMP);  // min(max(e_x, f0), e_neg)
        binary_max_tile_init();
        binary_max_tile(BN, TMP, BN);  // median(e_x, f0, e_neg)
        where_tile_init();
        where_tile<DataFormat::Float32>(FAR, BN, EX, EX);  // far ? median : e_x
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(EX, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_acc, 1);
        cb_pop_front(cb_bn, 1);
        cb_pop_front(cb_f0, 1);
        cb_pop_front(cb_en, 1);
        cb_pop_front(cb_tol, 1);
    }
}
