// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode RMSNorm compute, shared-statistic variant of compute_rms_norm.cpp with the same arithmetic
// (ttnn.rms_norm's layernorm.cpp RMSNORM + FUSE_GAMMA, fp32 DEST, HiFi4), but the row statistic is computed once:
//   every core: x2 = x * x for its own num_tiles tiles (packed fp32, sent to the reducer by the writer)
//   reducer:    ex2 = sum_row(x2) over all Wt tiles in order * (1 / W); rstd = rsqrt(ex2 + eps) -> cb_ex2pe_send
//               (the writer copies it into every core's cb_ex2pe)
//   every core: out[t] = (x[t] * rstd col-broadcast) * gamma[t] row-broadcast for its own tiles.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(4);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(5);
    constexpr uint32_t cb_x2 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ex2 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_ex2pe = get_compile_time_arg_val(8);
    constexpr uint32_t cb_fusion = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t cb_x2all = get_compile_time_arg_val(11);
    constexpr uint32_t cb_ex2pe_send = get_compile_time_arg_val(12);
    constexpr uint32_t blk = 4;  // fp32 DEST, half sync

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t is_reducer = get_arg_val<uint32_t>(2);
    (void)t_start;

    compute_kernel_hw_startup(cb_in, cb_in, cb_x2);

    // x^2 for this core's slice
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_x2);
    cb_wait_front(cb_in, num_tiles);
    for (uint32_t b0 = 0; b0 < num_tiles; b0 += blk) {
        const uint32_t n = (num_tiles - b0) < blk ? (num_tiles - b0) : blk;
        mul_init(cb_in, cb_in, false);
        cb_reserve_back(cb_x2, n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            mul_tiles(cb_in, cb_in, b0 + j, b0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(j, cb_x2);
        }
        tile_regs_release();
        cb_push_back(cb_x2, n);
    }

    if (is_reducer) {
        cb_wait_front(cb_eps, 1);
        // mean(x^2): row reduce over all tiles in order, then * 1/W
        reconfig_data_format(cb_in, cb_x2all, cb_in, cb_scaler);
        cb_wait_front(cb_scaler, 1);
        reconfig_data_format(cb_x2all, cb_scaler);
        tile_regs_acquire();
        reconfig_data_format(cb_scaler, cb_x2all);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2all, cb_scaler, cb_ex2);
        cb_wait_front(cb_x2all, Wt);
        for (uint32_t t = 0; t < Wt; ++t) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2all, cb_scaler, t, 0, 0);
        }
        cb_pop_front(cb_x2all, Wt);
        reduce_uninit();
        reconfig_data_format(cb_x2all, cb_scaler);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_w_bits);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_ex2, 1);
        pack_reconfig_data_format(cb_ex2);
        pack_tile(0, cb_ex2);
        tile_regs_release();
        cb_push_back(cb_ex2, 1);

        // rsqrt(mean + eps)
        reconfig_data_format(cb_ex2, cb_eps);
        pack_reconfig_data_format(cb_ex2pe_send);
        cb_wait_front(cb_ex2, 1);
        cb_reserve_back(cb_ex2pe_send, 1);
        add_init(cb_ex2, cb_eps, false);
        tile_regs_acquire();
        add_tiles(cb_ex2, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_ex2pe_send);
        tile_regs_release();
        cb_push_back(cb_ex2pe_send, 1);
        cb_pop_front(cb_ex2, 1);
        cb_pop_front(cb_eps, 1);
        cb_pop_front(cb_scaler, 1);
    }
    cb_wait_front(cb_ex2pe, 1);

    // This core's slice: (x * rstd) * gamma
    cb_wait_front(cb_gamma, num_tiles);
    for (uint32_t b0 = 0; b0 < num_tiles; b0 += blk) {
        const uint32_t n = (num_tiles - b0) < blk ? (num_tiles - b0) : blk;
        reconfig_data_format(cb_in, cb_ex2pe);
        pack_reconfig_data_format(cb_fusion);
        bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_in, cb_ex2pe);
        cb_reserve_back(cb_fusion, n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            mul_tiles_bcast_cols(cb_in, cb_ex2pe, b0 + j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(j, cb_fusion);
        }
        tile_regs_release();
        cb_push_back(cb_fusion, n);

        reconfig_data_format(cb_fusion, cb_gamma);
        pack_reconfig_data_format(cb_out);
        bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(cb_fusion, cb_gamma);
        cb_wait_front(cb_fusion, n);
        cb_reserve_back(cb_out, n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            mul_tiles_bcast_rows(cb_fusion, cb_gamma, j, b0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, n);
        cb_pop_front(cb_fusion, n);
    }
    cb_pop_front(cb_ex2pe, 1);
    cb_pop_front(cb_gamma, num_tiles);
    cb_pop_front(cb_in, num_tiles);
}
