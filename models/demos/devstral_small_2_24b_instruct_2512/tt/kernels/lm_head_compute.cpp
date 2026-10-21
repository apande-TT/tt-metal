// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/lm_head_decode.py: one tile row of tokens times this core's columns, PASSES groups of 2 * NC: in each
// pass the NC columns streamed into cb_wa (RISCV_0) land in DEST 0..NC-1 and the NC of cb_wb (RISCV_1) in NC..2NC-1,
// each accumulated over the whole K in K order in fp32 DEST (full-sync DEST holds the 2 * NC tiles, so no partial
// ever leaves it) and rounded to bf16 once at pack -- the result of ttnn's 1D-multicast matmul, whose fp32 partials
// between K blocks are reloaded exactly.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t NC = get_compile_time_arg_val(1);  // columns a stream a pass
    constexpr uint32_t RB = get_compile_time_arg_val(2);  // K rows a push
    constexpr uint32_t PASSES = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = 0, cb_wa = 1, cb_wb = 2, cb_out = 16;
    static_assert(Kt % RB == 0, "lm_head blocking");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wa, cb_out);
    matmul_block_init(cb_x, cb_wa, 0, NC, 1, RB);
    cb_wait_front(cb_x, Kt);

    for (uint32_t p = 0; p < PASSES; ++p) {
        tile_regs_acquire();
        for (uint32_t k0 = 0; k0 < Kt; k0 += RB) {
            cb_wait_front(cb_wa, RB * NC);
            for (uint32_t i = 0; i < RB; ++i) {
                matmul_block(cb_x, cb_wa, k0 + i, i * NC, 0, 0, NC, 1, RB);
            }
            cb_pop_front(cb_wa, RB * NC);
            cb_wait_front(cb_wb, RB * NC);
            for (uint32_t i = 0; i < RB; ++i) {
                matmul_block(cb_x, cb_wb, k0 + i, i * NC, NC, 0, NC, 1, RB);
            }
            cb_pop_front(cb_wb, RB * NC);
        }
        tile_regs_commit();

        cb_reserve_back(cb_out, 2 * NC);
        tile_regs_wait();
        for (uint32_t c = 0; c < 2 * NC; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, 2 * NC);
    }
    cb_pop_front(cb_x, Kt);
}
