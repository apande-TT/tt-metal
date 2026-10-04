// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the 4-way split square-mean (see tt/cpp_sqmean.py). Every core folds one face of a
// tile row, its tiles' faces arriving four to a CB tile: copied straight to DEST, squared by the stock
// calculate_square, and added into the accumulator face one after another in tile order (fold_faces_add, the
// stock fold's float32 SFPU add) -- per element exactly the stock square + fold.
// The folded face is packed for the writer. A core of role 0 (the even face) then takes the assembled
// [its face | the odd face, sent over by its partner core] tile and runs the stock sfpu_reduce<SUM,
// REDUCE_ROW> (which reduces each 16-row face pair with the same code) and the float32 1/dim post-multiply.

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
constexpr auto cb_part = tt::CBIndex::c_1;
constexpr auto cb_asm = tt::CBIndex::c_2;
constexpr auto cb_y = tt::CBIndex::c_16;

#ifdef TRISC_MATH
#include "sfpi.h"
namespace ckernel::sfpu {
// acc (face 0 of the call's tile) += faces 0, 1, 2, 3 of the next tile, one after another: the stock fold's
// float32 SFPU adds, in tile order (the four faces are four consecutive tiles' squared face). SEED: the first
// face starts the accumulator, as the stock fold's first tile does. Adds only -- the squares are the stock
// calculate_square, stored to DEST before this runs, so nothing here can fuse a multiply into the add.
template <bool SEED>
inline void fold_faces_add() {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat f0 = sfpi::dst_reg[32];
        const sfpi::vFloat f1 = sfpi::dst_reg[40];
        const sfpi::vFloat f2 = sfpi::dst_reg[48];
        const sfpi::vFloat f3 = sfpi::dst_reg[56];
        sfpi::vFloat acc;
        if constexpr (SEED) {
            acc = f0;
        } else {
            const sfpi::vFloat prev = sfpi::dst_reg[0];
            acc = prev + f0;
        }
        acc = acc + f1;
        acc = acc + f2;
        acc = acc + f3;
        sfpi::dst_reg[0] = acc;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);  // 0: even face (reduces); 1: odd face (sends)

    compute_kernel_hw_startup(cb_x, cb_part);

    tile_regs_acquire();
    copy_init(cb_x);
    square_tile_init();
    for (uint32_t w = 0; w < WT; w += 4 * BATCH) {
        cb_wait_front(cb_x, BATCH);
        for (uint32_t j = 0; j < BATCH; ++j) {
            copy_tile(cb_x, j, 1);
            MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_square, (APPROX), 1, VectorMode::RC));
            if (w == 0 && j == 0) {
                MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, fold_faces_add, (true), 0, VectorMode::None));
            } else {
                MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, fold_faces_add, (false), 0, VectorMode::None));
            }
        }
        cb_pop_front(cb_x, BATCH);
    }
    tile_regs_commit();
    cb_reserve_back(cb_part, 1);
    tile_regs_wait();
    pack_tile(0, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 1);

    if (role == 0) {
        cb_wait_front(cb_asm, 1);
        tile_regs_acquire();
        copy_init(cb_asm);
        copy_tile(cb_asm, 0, 0);
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
        cb_pop_front(cb_asm, 1);
    }
}
