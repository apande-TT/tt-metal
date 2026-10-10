// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill routed experts, token-gathered (compute). Per unit (RB 32-token row blocks of one expert's tokens; dest holds
// 2 * RB fp32 tiles, so RB = 4 needs full-sync dest):
// tilize the gathered x rows; per intermediate column, gate and up over all of K in fp32 dest, then on the
// SFPU in fp32 silu(gate) * up * the row's routing weight (unpacked straight to dest), packed fp32 as act;
// act times the expert's down rows, each output column over the expert's K in fp32 dest, packed fp32 for the
// writer to scatter. No activation is rounded to bf16 on the way.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t CPE = get_compile_time_arg_val(2);
    constexpr uint32_t RB = get_compile_time_arg_val(3);  // 32-token row blocks per unit
    constexpr uint32_t cb_rm = 0, cb_g = 1, cb_u = 2, cb_scale = 3, cb_count = 4, cb_wd = 7;
    constexpr uint32_t cb_x = 8, cb_nrb = 11, cb_act = 17, cb_y = 18;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_g, cb_act);

    cb_wait_front(cb_count, 1);
    const uint32_t groups = read_tile_value(cb_count, 0, 0);
    cb_pop_front(cb_count, 1);

    for (uint32_t grp = 0; grp < groups; ++grp) {
        reconfig_data_format_srca(cb_rm);
        pack_reconfig_data_format(cb_x);
        tilize_init(cb_rm, Kt, cb_x);
        // Only the unit's nrb row blocks hold tokens; every buffer still moves RB blocks per unit, so reservations
        // stay aligned and never wrap.
        cb_wait_front(cb_nrb, 1);
        const uint32_t nrb = read_tile_value(cb_nrb, 0, 0);
        cb_pop_front(cb_nrb, 1);
        for (uint32_t rr = 0; rr < nrb; ++rr) {
            cb_wait_front(cb_rm, Kt);
            cb_reserve_back(cb_x, Kt);
            tilize_block(cb_rm, Kt, cb_x);
            cb_push_back(cb_x, Kt);
            cb_pop_front(cb_rm, Kt);
        }
        tilize_uninit(cb_rm, cb_x);
        cb_reserve_back(cb_x, (RB - nrb) * Kt);
        cb_push_back(cb_x, (RB - nrb) * Kt);
        cb_wait_front(cb_x, RB * Kt);
        cb_wait_front(cb_scale, RB);

        // act[rr, c] = silu(x[rr] @ Wg[:, c]) * (x[rr] @ Wu[:, c]) * w[rr]; dest 2rr = gate, 2rr + 1 = up, then w.
        for (uint32_t c = 0; c < CPE; ++c) {
            reconfig_data_format(cb_g, cb_x);
            pack_reconfig_data_format(cb_act);
            matmul_init(cb_x, cb_g);
            tile_regs_acquire();
            for (uint32_t kb = 0; kb < Kt; kb += KB) {
                cb_wait_front(cb_g, KB);
                cb_wait_front(cb_u, KB);
                for (uint32_t k = 0; k < KB; ++k) {
                    for (uint32_t rr = 0; rr < nrb; ++rr) {
                        matmul_tiles(cb_x, cb_g, rr * Kt + kb + k, k, 2 * rr);
                        matmul_tiles(cb_x, cb_u, rr * Kt + kb + k, k, 2 * rr + 1);
                    }
                }
                cb_pop_front(cb_g, KB);
                cb_pop_front(cb_u, KB);
            }
            silu_tile_init();
            for (uint32_t rr = 0; rr < nrb; ++rr) {
                silu_tile(2 * rr);
            }
            mul_binary_tile_init();
            for (uint32_t rr = 0; rr < nrb; ++rr) {
                mul_binary_tile(2 * rr, 2 * rr + 1, 2 * rr);
            }
            reconfig_data_format_srca(cb_scale);
            copy_init(cb_scale);
            for (uint32_t rr = 0; rr < nrb; ++rr) {
                copy_tile(cb_scale, rr, 2 * rr + 1);
            }
            mul_binary_tile_init();
            for (uint32_t rr = 0; rr < nrb; ++rr) {
                mul_binary_tile(2 * rr, 2 * rr + 1, 2 * rr);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_act, RB);
            for (uint32_t rr = 0; rr < nrb; ++rr) {
                pack_tile(2 * rr, cb_act);
            }
            cb_push_back(cb_act, RB);
            tile_regs_release();
        }
        cb_pop_front(cb_x, RB * Kt);
        cb_pop_front(cb_scale, RB);

        // y[rr, n] = act[rr] @ Wd[:, n] over the expert's CPE K tiles (act tile RB * k + rr).
        cb_wait_front(cb_act, RB * CPE);
        reconfig_data_format(cb_wd, cb_act);
        pack_reconfig_data_format(cb_y);
        matmul_init(cb_act, cb_wd);
        for (uint32_t nn = 0; nn < Kt; ++nn) {
            cb_wait_front(cb_wd, CPE);
            tile_regs_acquire();
            for (uint32_t k = 0; k < CPE; ++k) {
                for (uint32_t rr = 0; rr < nrb; ++rr) {
                    matmul_tiles(cb_act, cb_wd, RB * k + rr, k, rr);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_y, RB);
            for (uint32_t rr = 0; rr < nrb; ++rr) {
                pack_tile(rr, cb_y);
            }
            cb_push_back(cb_y, RB);
            tile_regs_release();
            cb_pop_front(cb_wd, CPE);
        }
        cb_pop_front(cb_act, RB * CPE);
    }
}
