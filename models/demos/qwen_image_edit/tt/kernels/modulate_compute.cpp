// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The modulated LayerNorm tail, one float32 tile at a time: y = ((xc * r) * s) + h -- the centred input times its row's
// 1/std, times (1 + scale), plus shift -- with the SFPU float32 multiply / add ttnn's binary_ng uses on operands
// unpacked straight to DST, in the ttnn spelling's order (bit-identical to it).

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
    constexpr uint32_t cb_x = 0, cb_r = 1, cb_s = 2, cb_h = 3, cb_out = 16;
    constexpr uint32_t ACC = 0, OPD = 1;

    compute_kernel_hw_startup(cb_x, cb_out);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_r, 1);
        cb_wait_front(cb_s, 1);
        cb_wait_front(cb_h, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, 0, ACC);
        copy_init(cb_r);
        copy_tile(cb_r, 0, OPD);
        mul_binary_tile_init();
        mul_binary_tile(ACC, OPD, ACC);  // xc * (1 / std)
        copy_init(cb_s);
        copy_tile(cb_s, 0, OPD);
        mul_binary_tile_init();
        mul_binary_tile(ACC, OPD, ACC);  // * (1 + scale)
        copy_init(cb_h);
        copy_tile(cb_h, 0, OPD);
        add_binary_tile_init();
        add_binary_tile(ACC, OPD, ACC);  // + shift
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(ACC, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_h, 1);
    }
}
