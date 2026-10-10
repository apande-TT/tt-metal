// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill routed experts, combine (compute): out tile = sum over the row's planes of plane * mask, every operand
// unpacked straight to fp32 dest and multiplied / added on the SFPU in fp32, packed once (bf16). A plane whose rows
// are all valid (bit k of `full`) is added unmasked.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t KMAX = get_compile_time_arg_val(3);  // planes per push; the first `planes` are summed
    constexpr uint32_t CW = get_compile_time_arg_val(4);    // output tiles per unit
    const uint32_t core = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_p = 0, cb_cnt = 4, cb_mask = 5, cb_out = 16;

    compute_kernel_hw_startup(cb_p, cb_out);
    for (uint32_t unit = core; unit < MT * (Kt / CW); unit += NC) {
        cb_wait_front(cb_cnt, 1);
        const uint32_t planes = read_tile_value(cb_cnt, 0, 0);
        const uint32_t full = read_tile_value(cb_cnt, 0, 1);
        cb_pop_front(cb_cnt, 1);
        cb_wait_front(cb_mask, KMAX);
        cb_wait_front(cb_p, KMAX * CW);
        for (uint32_t c = 0; c < CW; ++c) {
            tile_regs_acquire();
            copy_init(cb_p);
            copy_tile(cb_p, c, 0);
            if (!(full & 1u)) {
                copy_init(cb_mask);
                copy_tile(cb_mask, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
            }
            for (uint32_t k = 1; k < planes; ++k) {
                copy_init(cb_p);
                copy_tile(cb_p, k * CW + c, 1);
                if (!((full >> k) & 1u)) {
                    copy_init(cb_mask);
                    copy_tile(cb_mask, k, 2);
                    mul_binary_tile_init();
                    mul_binary_tile(1, 2, 1);
                }
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_p, KMAX * CW);
        cb_pop_front(cb_mask, KMAX);
    }
}
