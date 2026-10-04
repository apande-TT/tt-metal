// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ codec residual add + RMS norm (see tt/cpp_addnorm.py). Per unit (a tile row):
//   s = h + r                    binary_ng ADD's float32 SFPU add (NearestEven): the same residual stream
//   z = sum over the row of s^2  square_tile + add_binary_tile fold + sfpu_reduce<SUM, REDUCE_ROW>
//   inv = rsqrt(z / dim + eps)   mul_unary / add_unary by scalar, exact (non-approx) rsqrt
//   y = s * inv                  mul_binary_tile against the column-filled inv tile (filled by the writer)
// every operand unpacked straight to DEST (float32 SFPU throughout); s is kept for the row in L1.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t WT = get_compile_time_arg_val(0);
constexpr uint32_t INV_N_BITS = get_compile_time_arg_val(1);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);
constexpr uint32_t Y_NARROW = get_compile_time_arg_val(3);  // 1: y is narrower than float32

constexpr auto cb_h = tt::CBIndex::c_0;
constexpr auto cb_r = tt::CBIndex::c_1;
constexpr auto cb_s = tt::CBIndex::c_2;
constexpr auto cb_inv = tt::CBIndex::c_3;
constexpr auto cb_invf = tt::CBIndex::c_4;
constexpr auto cb_hn = tt::CBIndex::c_16;
constexpr auto cb_y = tt::CBIndex::c_17;

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_h, cb_r, cb_hn);

    for (uint32_t u = 0; u < nu; ++u) {
        // s = h + r: packed twice -- to the writer (the new residual stream) and to the row buffer.
        for (uint32_t w = 0; w < WT; ++w) {
            cb_wait_front(cb_h, 1);
            cb_wait_front(cb_r, 1);
            cb_reserve_back(cb_hn, 1);
            cb_reserve_back(cb_s, 1);
            tile_regs_acquire();
            copy_init(cb_h);
            copy_tile(cb_h, 0, 0);
            copy_init(cb_r);
            copy_tile(cb_r, 0, 1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_hn);
            pack_tile(0, cb_s);
            tile_regs_release();
            cb_push_back(cb_hn, 1);
            cb_push_back(cb_s, 1);
            cb_pop_front(cb_h, 1);
            cb_pop_front(cb_r, 1);
        }

        // inv = rsqrt(sum(s^2) / dim + eps), the row sums in column 0
        cb_wait_front(cb_s, WT);
        cb_reserve_back(cb_inv, 1);
        tile_regs_acquire();
        copy_init(cb_s);
        copy_tile(cb_s, 0, 0);
        square_tile_init();
        square_tile(0);
        for (uint32_t w = 1; w < WT; ++w) {
            copy_tile(cb_s, w, 1);
            square_tile_init();
            square_tile(1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        sfpu_reduce_init<ckernel::PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<ckernel::PoolType::SUM, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, INV_N_BITS);
        add_unary_tile(0, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_inv);
        tile_regs_release();
        cb_push_back(cb_inv, 1);

        // y = s * inv (column-filled)
        cb_wait_front(cb_invf, 1);
        for (uint32_t w = 0; w < WT; ++w) {
            cb_reserve_back(cb_y, 1);
            tile_regs_acquire();
            copy_init(cb_s);
            copy_tile(cb_s, w, 0);
            copy_init(cb_invf);
            copy_tile(cb_invf, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (Y_NARROW) {
                pack_reconfig_data_format(cb_hn, cb_y);
            }
            pack_tile(0, cb_y);
            if constexpr (Y_NARROW) {
                pack_reconfig_data_format(cb_y, cb_hn);
            }
            tile_regs_release();
            cb_push_back(cb_y, 1);
        }
        cb_pop_front(cb_s, WT);
        cb_pop_front(cb_invf, 1);
    }
}
