// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode gate/up -> act over the ROUTED experts only (compute). For each column the reader streams: the one tile
// row of tokens [1 x Kt] times that gate and up weight column [Kt x 1], accumulated over all of K in fp32 dest,
// then on the SFPU in fp32 silu(gate) * up * the tokens' routing weights for the column's expert (unpacked
// straight to dest), packed once as act.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_g = 1, cb_u = 2, cb_count = 4, cb_scale = 7, cb_act = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_g, cb_act);

    cb_wait_front(cb_count, 1);
    const uint32_t n = read_tile_value(cb_count, 0, 0);
    cb_pop_front(cb_count, 1);

    cb_wait_front(cb_x, Kt);
    for (uint32_t s = 0; s < n; ++s) {
        reconfig_data_format(cb_g, cb_x);
        pack_reconfig_data_format(cb_act);
        matmul_init(cb_x, cb_g);
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < Kt; kb += KB) {
            cb_wait_front(cb_g, KB);
            cb_wait_front(cb_u, KB);
            for (uint32_t k = 0; k < KB; ++k) {
                matmul_tiles(cb_x, cb_g, kb + k, k, 0);
                matmul_tiles(cb_x, cb_u, kb + k, k, 1);
            }
            cb_pop_front(cb_g, KB);
            cb_pop_front(cb_u, KB);
        }
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        cb_wait_front(cb_scale, 1);
        reconfig_data_format_srca(cb_scale);
        copy_init(cb_scale);
        copy_tile(cb_scale, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_scale, 1);
        tile_regs_wait();
        cb_reserve_back(cb_act, 1);
        pack_tile(0, cb_act);
        cb_push_back(cb_act, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_x, Kt);
}
