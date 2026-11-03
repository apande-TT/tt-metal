// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// y = ((t_0 + t_1) + t_2) + ... + t_{L-1}: the L lane tiles of each output tile, L at a time from c_0 (lane order),
// summed left to right with the SFPU float32 add ttnn's binary_ng uses, on operands unpacked straight to the
// float32 DST -- the same values as the ttnn.add chain over L separate lane outputs.
// Compile-time: L. Runtime: num_tiles.

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
    constexpr uint32_t lanes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = 0, cb_out = 16;
    constexpr uint32_t ACC = 0, PART = 1;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_in, lanes);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_init(cb_in);
        copy_tile(cb_in, 0, ACC);
        for (uint32_t r = 1; r < lanes; ++r) {
            copy_init(cb_in);
            copy_tile(cb_in, r, PART);
            add_binary_tile_init();
            add_binary_tile(ACC, PART, ACC);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(ACC, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, lanes);
    }
}
