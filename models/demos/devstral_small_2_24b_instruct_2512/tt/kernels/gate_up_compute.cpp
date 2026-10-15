// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/gate_up_decode.py: one tile row of tokens times this core's NC gate columns (cb_wg) and NC up
// columns (cb_wu), each accumulated over the whole K in fp32 DEST (full-sync DEST holds the 2 * NC tiles, so no
// partial ever leaves it): gate in DEST 0..NC-1, up in NC..2NC-1. Then silu(gate) * up on the SFPU in fp32 and ONE
// rounding to bf16 at pack (ttnn's chain rounds gate and up to bf16 before the multiply).
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t NC = get_compile_time_arg_val(1);  // output columns (gate and up each)
    constexpr uint32_t RB = get_compile_time_arg_val(2);  // K rows a push
    constexpr uint32_t cb_x = 0, cb_wg = 1, cb_wu = 2, cb_out = 16;
    static_assert(Kt % RB == 0, "gate_up blocking");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wg, cb_out);
    matmul_block_init(cb_x, cb_wg, 0, NC, 1, RB);
    cb_wait_front(cb_x, Kt);

    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < Kt; k0 += RB) {
        cb_wait_front(cb_wg, RB * NC);
        for (uint32_t i = 0; i < RB; ++i) {
            matmul_block(cb_x, cb_wg, k0 + i, i * NC, 0, 0, NC, 1, RB);
        }
        cb_pop_front(cb_wg, RB * NC);
        cb_wait_front(cb_wu, RB * NC);
        for (uint32_t i = 0; i < RB; ++i) {
            matmul_block(cb_x, cb_wu, k0 + i, i * NC, NC, 0, NC, 1, RB);
        }
        cb_pop_front(cb_wu, RB * NC);
    }
    silu_tile_init();
    for (uint32_t c = 0; c < NC; ++c) {
        silu_tile(c);
    }
    mul_binary_tile_init();
    for (uint32_t c = 0; c < NC; ++c) {
        mul_binary_tile(c, NC + c, c);
    }
    tile_regs_commit();

    cb_reserve_back(cb_out, NC);
    tile_regs_wait();
    for (uint32_t c = 0; c < NC; ++c) {
        pack_tile(c, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, NC);
    cb_pop_front(cb_x, Kt);
}
