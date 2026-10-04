// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ split square-mean (see tt/cpp_sqmean.py): `mean(square(x), -1)` of a float32
// x, replaying the stock ops' own float32 SFPU primitives in the stock order, per element:
//   x^2                  unary SQUARE           calculate_square (unpacked straight to DEST, float32 DEST)
//   fold over the row    reduce W SUM (SFPU)    add_binary_tile over the row's tiles, first tile seeding
//   row sums             sfpu_reduce<SUM, REDUCE_ROW>   (each 16-row face pair reduced by the same code)
//   * 1 / dim            reduce's post-multiply  mul_unary_tile
// on HALF the tile: a unit holds one 16-row half of a tile row in faces 0 and 1, and the elementwise steps
// run VectorMode::R (faces 0 and 1 only). The fold is elementwise and the row reduce treats each face pair
// alike, so every row's mean is the stock op's, bit for bit; two cores share a tile row.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

constexpr uint32_t WT = get_compile_time_arg_val(0);
constexpr uint32_t BATCH = get_compile_time_arg_val(1);
constexpr uint32_t INV_N_BITS = get_compile_time_arg_val(2);

constexpr auto cb_x = tt::CBIndex::c_0;
constexpr auto cb_y = tt::CBIndex::c_16;

inline void square_half(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_square, (APPROX), idst, VectorMode::R));
}

inline void add_half(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::R)));
}

void kernel_main() {
    const uint32_t nk = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_y);

    for (uint32_t k = 0; k < nk; ++k) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < WT; w += BATCH) {
            cb_wait_front(cb_x, BATCH);
            for (uint32_t i = 0; i < BATCH; ++i) {
                const uint32_t dst = (w + i == 0) ? 0 : 1;
                copy_init(cb_x);
                copy_tile(cb_x, i, dst);
                square_tile_init();
                square_half(dst);
                if (w + i > 0) {
                    add_binary_tile_init();
                    add_half(0, 1, 0);
                }
            }
            cb_pop_front(cb_x, BATCH);
        }
        sfpu_reduce_init<ckernel::PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<ckernel::PoolType::SUM, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, INV_N_BITS);
        tile_regs_commit();
        cb_reserve_back(cb_y, 1);
        tile_regs_wait();
        pack_tile(0, cb_y);
        tile_regs_release();
        cb_push_back(cb_y, 1);
    }
}
