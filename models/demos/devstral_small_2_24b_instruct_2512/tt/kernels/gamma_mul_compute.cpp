// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/gamma_mul.py: binary_ng's SFPU row-broadcast multiply (eltwise_binary_sfpu_row_bcast.cpp, MUL, bf16
// in / out, fp32 DEST off) with its per-tile work hoisted: this core's GW gamma tiles are row-broadcast
// (unary_bcast<ROW>, packed to cb_gb) once, then a tile row's GW tiles a DEST acquire: y's tile copied to slot 2j, the
// broadcast gamma tile of its column to slot 2j + 1, mul_binary_tile(2j, 2j + 1, 2j), slot 2j packed. Same calls per
// tile as the stock kernel.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t nrows = get_arg_val<uint32_t>(0);
    constexpr uint32_t GW = get_compile_time_arg_val(0);  // this core's columns (= tiles a DEST acquire)
    constexpr uint32_t cb_y = 0, cb_g = 1, cb_out = 2, cb_gb = 6;

    compute_kernel_hw_startup(cb_y, cb_out);
    copy_init(cb_y);
    mul_binary_tile_init();

    // this core's gamma tiles row-broadcast once (the stock kernel does this for every output tile)
    pack_reconfig_data_format(cb_out, cb_gb);
    reconfig_data_format(cb_g, cb_g);
    unary_bcast_init<BroadcastType::ROW>(cb_g);
    cb_reserve_back(cb_gb, GW);
    for (uint32_t w = 0; w < GW; ++w) {
        cb_wait_front(cb_g, 1);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_g, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gb);
        tile_regs_release();
        cb_pop_front(cb_g, 1);
    }
    cb_push_back(cb_gb, GW);
    cb_wait_front(cb_gb, GW);
    pack_reconfig_data_format(cb_gb, cb_out);

    for (uint32_t r = 0; r < nrows; ++r) {
        constexpr uint32_t m = GW;
        cb_wait_front(cb_y, m);
        cb_reserve_back(cb_out, m);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_gb, cb_y);
        copy_init(cb_y);
        for (uint32_t j = 0; j < m; ++j) {
            copy_tile(cb_y, j, 2 * j);
        }
        reconfig_data_format_srca(cb_y, cb_gb);
        copy_init(cb_gb);
        for (uint32_t j = 0; j < m; ++j) {
            copy_tile(cb_gb, j, 2 * j + 1);
            mul_binary_tile(2 * j, 2 * j + 1, 2 * j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < m; ++j) {
            pack_tile(2 * j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, m);
        cb_pop_front(cb_y, m);
    }
}
