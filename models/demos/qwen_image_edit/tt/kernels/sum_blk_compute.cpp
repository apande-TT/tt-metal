// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sum_parts_compute.cpp blocked: y = ((p0 + p1) + p2) + ... for BLK tiles per DST acquire (slots 0..BLK-1 hold the
// running sums, BLK..2 BLK-1 the next operand's tiles), each init once per block instead of once per tile. Same SFPU
// float32 adds in the same order per tile, on operands unpacked straight to the float32 DST: bit-identical.
// Compile time: n, BLK (2 BLK <= 8 float32 DST tiles with dst_full_sync_en).  Runtime: num_tiles.

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
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = 16;

    compute_kernel_hw_startup(0, cb_out);
    for (uint32_t i = 0; i < num_tiles; i += BLK) {
        const uint32_t blk = (num_tiles - i) < BLK ? (num_tiles - i) : BLK;
        for (uint32_t k = 0; k < n; ++k) {
            cb_wait_front(k, blk);
        }
        cb_reserve_back(cb_out, blk);

        tile_regs_acquire();
        copy_init(0);
        for (uint32_t t = 0; t < blk; ++t) {
            copy_tile(0, t, t);
        }
        for (uint32_t k = 1; k < n; ++k) {
            copy_init(k);
            for (uint32_t t = 0; t < blk; ++t) {
                copy_tile(k, t, BLK + t);
            }
            add_binary_tile_init();
            for (uint32_t t = 0; t < blk; ++t) {
                add_binary_tile(t, BLK + t, t);
            }
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t t = 0; t < blk; ++t) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, blk);
        for (uint32_t k = 0; k < n; ++k) {
            cb_pop_front(k, blk);
        }
    }
}
