// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// masked_lane_bmm's compute: masked_mm_compute.cpp one tile row at a time (each row has its own b entry). Per K
// tile: the 8 lanes of a's tile (a * mask_r, exact) into c_3, then the 8 lane products against b's nt tiles of that
// K row, accumulated over the lanes in DST and over K by the packer in L1 (float32), written once per row.
// Compile-time: kt, nt, sbw. Runtime: row count.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t sbw = get_compile_time_arg_val(2);
    constexpr uint32_t lanes = 8;
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_m = 2, cb_lane = 3, cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_lane, cb_b, cb_out);
    cb_wait_front(cb_m, lanes);
    for (uint32_t m = 0; m < rows; ++m) {
        cb_reserve_back(cb_out, nt);
        for (uint32_t k = 0; k < kt; ++k) {
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, nt);

            mul_init(cb_a, cb_m, false);
            pack_reconfig_data_format(cb_lane);
            pack_reconfig_l1_acc(0);
            cb_reserve_back(cb_lane, lanes);
            for (uint32_t r0 = 0; r0 < lanes; r0 += 4) {
                tile_regs_acquire();
                for (uint32_t r = 0; r < 4; ++r) {
                    mul_tiles(cb_a, cb_m, 0, r0 + r, r);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < 4; ++r) {
                    pack_tile<true>(r, cb_lane, r0 + r);
                }
                tile_regs_release();
            }
            cb_push_back(cb_lane, lanes);
            cb_pop_front(cb_a, 1);

            cb_wait_front(cb_lane, lanes);
            pack_reconfig_data_format(cb_out);
            pack_reconfig_l1_acc(k > 0 ? 1 : 0);
            matmul_block_init(cb_lane, cb_b, 0, sbw, 1, lanes);
            for (uint32_t j0 = 0; j0 < nt; j0 += sbw) {
                tile_regs_acquire();
                for (uint32_t r = 0; r < lanes; ++r) {
                    matmul_block(cb_lane, cb_b, r, j0, 0, 0, sbw, 1, lanes);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < sbw; ++j) {
                    pack_tile<true>(j, cb_out, j0 + j);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_lane, lanes);
            cb_pop_front(cb_b, nt);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_out, nt);
    }
    cb_pop_front(cb_m, lanes);
}
