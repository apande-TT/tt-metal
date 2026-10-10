// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// y = ((p0 + p1) + p2) + ...: the float32 sum of N (compile-time, 2..6) operands from c_0.., one tile at a time,
// left to right with the SFPU add ttnn's float32 binary_ng uses, on operands unpacked straight to the float32
// DST -- bit-identical to the chain of ttnn.add calls in that order.

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
    constexpr uint32_t n_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t ACC = 0, PART = 1;

    compute_kernel_hw_startup(0, cb_out);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        for (uint32_t k = 0; k < n_in; ++k) {
            cb_wait_front(k, 1);
        }
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_init(0);
        copy_tile(0, 0, ACC);
        for (uint32_t k = 1; k < n_in; ++k) {
            copy_init(k);
            copy_tile(k, 0, PART);
            add_binary_tile_init();
            add_binary_tile(ACC, PART, ACC);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(ACC, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
        for (uint32_t k = 0; k < n_in; ++k) {
            cb_pop_front(k, 1);
        }
    }
}
