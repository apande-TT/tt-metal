// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/norm_decode.py: ttnn's sharded RMSNorm (layernorm_sharded.cpp, RMSNORM, no gamma / beta, fp32 in,
// fp32 DEST, HiFi4) step for step, so the normed values are the stock ones:
//   x^2 = mul_tiles(x, x) -> cb_x2; partial = reduce<AVG, ROW>(x^2, scaler 1/128 (bf16)) -> cb_part
//   row workers (core x = 0 of each row): reduce<AVG, ROW> over the row's 8 partials with the global scaler (1.0)
//   -> cb_ex2, which the root (0, 0) reduces with its own row's 8 partials (12 tiles, scaler bf16(1/40));
//   root: + eps (bf16, column 0), rsqrt -> cb_ex2pe, sent to every core's cb_exg;
//   out = mul_tiles_bcast_cols(x, factor).
// TRUNC_BF16: each output value's low 16 bits cleared before the (bf16) pack -- the exact bf16 truncation, which a
// HiFi2 matmul multiplies exactly as it would the fp32 value.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);       // tiles of this core's shard
    constexpr uint32_t ROLE = get_compile_time_arg_val(1);     // 0 plain core, 1 row worker, 2 root
    constexpr uint32_t NBLOCKS = get_compile_time_arg_val(2);  // tiles the worker reduces (8, or 12 on the root)

    constexpr uint32_t cb_x = 0, cb_x2 = 1, cb_scaler = 2, cb_part = 3, cb_ext = 4, cb_scg = 5;
    constexpr uint32_t cb_ex2 = 6, cb_eps = 7, cb_ex2pe = 8, cb_exg = 9, cb_out = 16;
    constexpr uint32_t dst0 = 0;

    compute_kernel_hw_startup(cb_x, cb_x, cb_x2);

    // x^2
    cb_wait_front(cb_x, WT);
    mul_tiles_init(cb_x, cb_x);
    cb_reserve_back(cb_x2, WT);
    tile_regs_acquire();
    for (uint32_t w = 0; w < WT; ++w) {
        mul_tiles(cb_x, cb_x, w, w, w);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < WT; ++w) {
        pack_tile(w, cb_x2);
    }
    tile_regs_release();
    cb_push_back(cb_x2, WT);

    reconfig_data_format(cb_x, cb_x2, cb_x, cb_scaler);
    cb_wait_front(cb_x2, WT);
    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        cb_x2,
        cb_scaler,
        cb_part,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(1, WT, 1),
        compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(WT));
    reconfig_data_format(cb_x2, cb_scaler);
    cb_pop_front(cb_x2, WT);

    if constexpr (ROLE != 0) {
        reconfig_data_format(cb_scg, cb_ext);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_ext, cb_scg, cb_ex2);
        cb_reserve_back(cb_ex2, 1);
        cb_wait_front(cb_scg, 1);
        tile_regs_acquire();
        for (uint32_t w = 0; w < NBLOCKS; ++w) {
            cb_wait_front(cb_ext, 1);
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_ext, cb_scg, 0, 0, dst0);
            cb_pop_front(cb_ext, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex2);
        tile_regs_release();
        reduce_uninit();
        cb_push_back(cb_ex2, 1);
        reconfig_data_format(cb_x2, cb_scaler);

        if constexpr (ROLE == 2) {
            cb_wait_front(cb_ex2, 1);
            cb_reserve_back(cb_ex2pe, 1);
            cb_wait_front(cb_eps, 1);
            tile_regs_acquire();
            add_tiles_init(cb_ex2, cb_eps);
            add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
            tile_regs_wait();
            rsqrt_tile_init();
            rsqrt_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            cb_push_back(cb_ex2pe, 1);
            tile_regs_release();
            cb_pop_front(cb_ex2, 1);
        }
    }

    // x * factor (factor broadcast along each row)
    pack_reconfig_data_format(cb_out);
    reconfig_data_format(cb_x, cb_exg);
    mul_bcast_cols_init(cb_x, cb_exg);
    cb_reserve_back(cb_out, WT);
    cb_wait_front(cb_exg, 1);
    tile_regs_acquire();
    for (uint32_t w = 0; w < WT; ++w) {
        mul_tiles_bcast_cols(cb_x, cb_exg, w, 0, w);
    }
#ifdef TRUNC_BF16
    bitwise_and_tile_init();
    for (uint32_t w = 0; w < WT; ++w) {
        bitwise_and_tile<DataFormat::Int32>(w, 0xFFFF0000u);
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < WT; ++w) {
        pack_tile(w, cb_out);
    }
    tile_regs_release();
    cb_pop_front(cb_exg, 1);
    cb_push_back(cb_out, WT);
    cb_pop_front(cb_x, WT);
    if constexpr (ROLE != 0) {
        cb_pop_front(cb_scg, 1);
    }
    cb_pop_front(cb_scaler, 1);
    if constexpr (ROLE == 2) {
        cb_pop_front(cb_eps, 1);
    }
}
