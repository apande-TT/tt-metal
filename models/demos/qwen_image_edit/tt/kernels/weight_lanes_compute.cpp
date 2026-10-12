// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bf16 weight tile -> its 8 lane-masked copies (c_16..c_23): copy r = w * mask_r, mask_r the 0/1 tile keeping the
// rows i % 8 == r. The SFPU multiply on the widened bf16 values is exact (x * 1 = x, x * 0 = +-0), and so is the
// pack back to bf16: each copy equals ttnn.multiply(w, row mask r).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_w = 0, cb_m = 1, cb_out0 = 16, lanes = 8;
    constexpr uint32_t W = 0, M = 1;

    compute_kernel_hw_startup(cb_w, cb_out0);
    cb_wait_front(cb_m, lanes);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_w, 1);
        for (uint32_t r = 0; r < lanes; ++r) {
            cb_reserve_back(cb_out0 + r, 1);
            tile_regs_acquire();
            copy_init(cb_w);
            copy_tile(cb_w, 0, W);
            copy_init(cb_m);
            copy_tile(cb_m, r, M);
            mul_binary_tile_init();
            mul_binary_tile(W, M, W);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(W, cb_out0 + r);
            tile_regs_release();
            cb_push_back(cb_out0 + r, 1);
        }
        cb_pop_front(cb_w, 1);
    }
    cb_pop_front(cb_m, lanes);
}
