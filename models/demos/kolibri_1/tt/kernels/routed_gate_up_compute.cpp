// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode gate/up over the ROUTED experts only (compute). For each column the reader streams: the one tile
// row of tokens [1 x Kt] times that gate and up weight column [Kt x 1], accumulated over all of K in fp32
// dest, packed once.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_g = 1, cb_u = 2, cb_count = 4, cb_out_g = 16, cb_out_u = 17;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_g, cb_out_g);
    matmul_init(cb_x, cb_g);

    cb_wait_front(cb_count, 1);
    const uint32_t n = read_tile_value(cb_count, 0, 0);
    cb_pop_front(cb_count, 1);

    cb_wait_front(cb_x, Kt);
    for (uint32_t s = 0; s < n; ++s) {
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
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out_g, 1);
        cb_reserve_back(cb_out_u, 1);
        pack_tile(0, cb_out_g);
        pack_tile(1, cb_out_u);
        cb_push_back(cb_out_g, 1);
        cb_push_back(cb_out_u, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_x, Kt);
}
