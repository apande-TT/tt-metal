// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The row scale rs = rsqrt(mean(x * x, -1) + eps) of the precise float32 RMSNorm, one tile row (Wt tiles of
// 32 rows) at a time, bit-identical to the stock chain it replaces (_stubs/encoder_stack.py TtRMSNorm):
//   sq = x * x               binary_ng SFPU (both float32): copy a, copy b, mul_binary_tile
//   ms = mean(sq, -1)        the accurate float32 reduce W (SFPU): copy + add_binary_tile fold over the Wt
//                            tiles in order, sfpu_reduce<SUM, REDUCE_ROW>, then the 1 / W post-mul
//                            (mul_unary_tile with the scaler's bits) before the pack
//   ms + eps                 binary_ng scalar SFPU: the scalar tile (filled with eps's bits) and
//                            add_binary_tile<NearestEven>
//   rsqrt(ms + eps)          unary rsqrt_tile<RsqrtMode::Default>, math_approx_mode off
// Each square goes straight into the fold instead of through a float32 tensor, and ms stays in DEST instead of
// a float32 tensor: the same float32 values reach the same SFPU ops in the same order. Only column 0 of rs is
// valid (as the stock ms); the multiply that uses it broadcasts that column.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0;     // x tiles, streamed by the reader
constexpr uint32_t cb_out = 16;  // rs, one tile per row

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(1);  // float32 bits of 1 / W
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);    // float32 bits of eps

    CircularBuffer x(cb_x);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_x, cb_out);
    for (uint32_t r = 0; r < num_rows; ++r) {
        tile_regs_acquire();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // sq into DST 0 for the first tile (the fold's seed), else into DST 2 and folded into DST 0
            const uint32_t d = wt == 0 ? 0 : 2;
            x.wait_front(1);
            copy_init(cb_x);
            copy_tile(cb_x, 0, d);
            copy_tile(cb_x, 0, d + 1);
            x.pop_front(1);
            mul_binary_tile_init();
            mul_binary_tile(d, d + 1, d);
            if (wt > 0) {
                add_binary_tile_init();
                add_binary_tile(0, 2, 0);
            }
        }
        sfpu_reduce_init<ckernel::PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<ckernel::PoolType::SUM, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_w_bits);
        fill_tile_init();
        fill_tile_bitcast(1, eps_bits);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        rsqrt_tile_init();
        rsqrt_tile<RsqrtMode::Default>(0);
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
    }
}
