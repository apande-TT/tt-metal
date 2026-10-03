// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ acoustic attention softmax (see tt/cpp_softmax.py). Per unit it runs
// the stock chain's own float32 SFPU primitives in the stock order, every operand unpacked straight
// to DEST (UnpackToDestFp32), so each value is the one the stock ops produce:
//   s = raw + mask                   binary_ng ADD     add_binary_tile<NearestEven>
//   m = max over the row             reduce W MAX      binary_max_tile fold + sfpu_reduce<MAX, REDUCE_ROW>
//   e = exp(s - m)                   binary_ng SUB+EXP sub_binary_tile<NearestEven>, exp_tile
//   z = sum over the row             reduce W SUM      add_binary_tile fold + sfpu_reduce<SUM, REDUCE_ROW>
//   y = e / z                        binary_ng DIV     div_binary_tile
// m and z come back column-filled by the writer (binary_ng's reader-side column broadcast).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

constexpr uint32_t ST = get_compile_time_arg_val(0);

constexpr auto cb_raw = tt::CBIndex::c_0;
constexpr auto cb_mask = tt::CBIndex::c_1;
constexpr auto cb_s = tt::CBIndex::c_2;
constexpr auto cb_max = tt::CBIndex::c_3;
constexpr auto cb_maxf = tt::CBIndex::c_4;
constexpr auto cb_e = tt::CBIndex::c_5;
constexpr auto cb_sum = tt::CBIndex::c_6;
constexpr auto cb_sumf = tt::CBIndex::c_7;
constexpr auto cb_y = tt::CBIndex::c_16;

template <bool IS_MAX>
inline void reduce_row(uint32_t cb_in, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_init(cb_in);
    if constexpr (IS_MAX) {
        binary_max_tile_init();
    } else {
        add_binary_tile_init();
    }
    copy_tile(cb_in, 0, 0);
    for (uint32_t w = 1; w < ST; ++w) {
        copy_tile(cb_in, w, 1);
        if constexpr (IS_MAX) {
            binary_max_tile(0, 1, 0);
        } else {
            add_binary_tile(0, 1, 0);
        }
    }
    if constexpr (IS_MAX) {
        sfpu_reduce_init<ckernel::PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<ckernel::PoolType::MAX, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
    } else {
        sfpu_reduce_init<ckernel::PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<ckernel::PoolType::SUM, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_raw, cb_y);
    copy_init(cb_raw);

    for (uint32_t u = 0; u < nu; ++u) {
        // s = raw + mask
        for (uint32_t w = 0; w < ST; ++w) {
            cb_wait_front(cb_raw, 1);
            cb_wait_front(cb_mask, 1);
            cb_reserve_back(cb_s, 1);
            tile_regs_acquire();
            copy_init(cb_raw);
            copy_tile(cb_raw, 0, 0);
            copy_init(cb_mask);
            copy_tile(cb_mask, 0, 1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_s);
            tile_regs_release();
            cb_push_back(cb_s, 1);
            cb_pop_front(cb_raw, 1);
            cb_pop_front(cb_mask, 1);
        }

        // m = row max
        cb_wait_front(cb_s, ST);
        reduce_row<true>(cb_s, cb_max);

        // e = exp(s - m)
        cb_wait_front(cb_maxf, 1);
        for (uint32_t w = 0; w < ST; ++w) {
            cb_reserve_back(cb_e, 1);
            tile_regs_acquire();
            copy_init(cb_s);
            copy_tile(cb_s, w, 0);
            copy_init(cb_maxf);
            copy_tile(cb_maxf, 0, 1);
            sub_binary_tile_init();
            sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            exp_tile_init();
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_e);
            tile_regs_release();
            cb_push_back(cb_e, 1);
        }
        cb_pop_front(cb_s, ST);
        cb_pop_front(cb_maxf, 1);

        // z = row sum
        cb_wait_front(cb_e, ST);
        reduce_row<false>(cb_e, cb_sum);

        // y = e / z
        cb_wait_front(cb_sumf, 1);
        for (uint32_t w = 0; w < ST; ++w) {
            cb_reserve_back(cb_y, 1);
            tile_regs_acquire();
            copy_init(cb_e);
            copy_tile(cb_e, w, 0);
            copy_init(cb_sumf);
            copy_tile(cb_sumf, 0, 1);
            div_binary_tile_init();
            div_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_y);
            tile_regs_release();
            cb_push_back(cb_y, 1);
        }
        cb_pop_front(cb_e, ST);
        cb_pop_front(cb_sumf, 1);
    }
}
