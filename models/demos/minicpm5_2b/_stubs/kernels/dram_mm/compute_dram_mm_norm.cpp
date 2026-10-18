// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode projection with the RMSNorm of its input fused in front. The norm is
// kernels/rms_norm_shared/compute_rms_norm_shared.cpp's arithmetic call for call (this core's tpc-tile slice is
// squared, the reducer row-reduces all Kt x^2 tiles in order, every core normalizes its slice and packs it bf16);
// the writer all-gathers the normalized slices into every core's cb_in0, and the projection is
// compute_dram_mm.cpp's (one uninterrupted fp32 DEST accumulation over K per group of group_w output tiles).
// The program's math fidelity is the norm's; the projection runs at its own (mm_fid) through llk calls that take
// the fidelity as a template argument.

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
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

// matmul_block_init / matmul_block (api/compute/matmul.h, Blackhole path) with the math fidelity as a template
// argument instead of the program's MATH_FIDELITY, so the projection keeps its own fidelity beside the norm's.
namespace ckernel {
template <MathFidelity FID>
ALWI void mm_block_init_fid(uint32_t in0, uint32_t in1, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<FID, MM_THROTTLE>(in0, in1, 0, ct_dim, rt_dim)));
    MATH((throttled_mop_status = 0));
}

template <MathFidelity FID>
ALWI void mm_block_fid(
    uint32_t in0, uint32_t in1, uint32_t in0_index, uint32_t in1_index, uint32_t idst, uint32_t ct_dim,
    uint32_t rt_dim, uint32_t kt_dim) {
    state_configure(in1, in0, __builtin_LINE());
    UNPACK((llk_unpack_AB_matmul(in0, in1, in0_index, in1_index, ct_dim, rt_dim, kt_dim)));
#ifdef TRISC_MATH
    const uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            llk_math_matmul_init<FID, MM_THROTTLE_MAX>(in0, in1, 0, ct_dim, rt_dim);
            throttled_mop_status = 1;
        }
        llk_math_matmul<FID, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim);
    } else {
        if (throttled_mop_status != 0) {
            llk_math_matmul_init<FID, MM_THROTTLE>(in0, in1, 0, ct_dim, rt_dim);
            throttled_mop_status = 0;
        }
        llk_math_matmul<FID, MM_THROTTLE>(idst, ct_dim, rt_dim);
    }
#endif
}
}  // namespace ckernel

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t group_w = get_compile_time_arg_val(1);
    constexpr uint32_t k_block = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(6);
    constexpr uint32_t tpc = get_compile_time_arg_val(7);
    constexpr uint32_t cb_xin = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(9);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(10);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(11);
    constexpr uint32_t cb_x2 = get_compile_time_arg_val(12);
    constexpr uint32_t cb_ex2 = get_compile_time_arg_val(13);
    constexpr uint32_t cb_ex2pe = get_compile_time_arg_val(14);
    constexpr uint32_t cb_fusion = get_compile_time_arg_val(15);
    constexpr uint32_t cb_x2all = get_compile_time_arg_val(16);
    constexpr uint32_t cb_ex2pe_send = get_compile_time_arg_val(17);
    constexpr uint32_t cb_normout = get_compile_time_arg_val(18);
    constexpr auto mm_fid = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(19));  // projection
    constexpr uint32_t block_tiles = k_block * group_w;
    constexpr uint32_t blk = 4;  // fp32 DEST, half sync

    const uint32_t num_groups = get_arg_val<uint32_t>(4);
    const uint32_t is_reducer = get_arg_val<uint32_t>(7);
    const uint32_t does_norm = get_arg_val<uint32_t>(8);

    // ---------------- RMSNorm (as compute_rms_norm_shared.cpp), on the norm cores ----------------
    compute_kernel_hw_startup(cb_xin, cb_xin, cb_x2);
    if (does_norm) {
        reconfig_data_format(cb_xin, cb_xin);
        pack_reconfig_data_format(cb_x2);
        cb_wait_front(cb_xin, tpc);
        for (uint32_t b0 = 0; b0 < tpc; b0 += blk) {
            const uint32_t n = (tpc - b0) < blk ? (tpc - b0) : blk;
            mul_init(cb_xin, cb_xin, false);
            cb_reserve_back(cb_x2, n);
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles(cb_xin, cb_xin, b0 + j, b0 + j, j);
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
            reconfig_data_format(cb_xin, cb_x2all, cb_xin, cb_scaler);
            cb_wait_front(cb_scaler, 1);
            reconfig_data_format(cb_x2all, cb_scaler);
            tile_regs_acquire();
            reconfig_data_format(cb_scaler, cb_x2all);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2all, cb_scaler, cb_ex2);
            cb_wait_front(cb_x2all, Kt);
            for (uint32_t t = 0; t < Kt; ++t) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2all, cb_scaler, t, 0, 0);
            }
            cb_pop_front(cb_x2all, Kt);
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

        cb_wait_front(cb_gamma, tpc);
        for (uint32_t b0 = 0; b0 < tpc; b0 += blk) {
            const uint32_t n = (tpc - b0) < blk ? (tpc - b0) : blk;
            reconfig_data_format(cb_xin, cb_ex2pe);
            pack_reconfig_data_format(cb_fusion);
            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_xin, cb_ex2pe);
            cb_reserve_back(cb_fusion, n);
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles_bcast_cols(cb_xin, cb_ex2pe, b0 + j, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_fusion);
            }
            tile_regs_release();
            cb_push_back(cb_fusion, n);

            reconfig_data_format(cb_fusion, cb_gamma);
            pack_reconfig_data_format(cb_normout);
            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(cb_fusion, cb_gamma);
            cb_wait_front(cb_fusion, n);
            cb_reserve_back(cb_normout, n);
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles_bcast_rows(cb_fusion, cb_gamma, j, b0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_normout);
            }
            tile_regs_release();
            cb_push_back(cb_normout, n);
            cb_pop_front(cb_fusion, n);
        }
        cb_pop_front(cb_ex2pe, 1);
        cb_pop_front(cb_gamma, tpc);
        cb_pop_front(cb_xin, tpc);
    }

    // ---------------- projection (as compute_dram_mm.cpp) ----------------
    reconfig_data_format(cb_in1, cb_in0);
    pack_reconfig_data_format(cb_out);
    ckernel::mm_block_init_fid<mm_fid>(cb_in0, cb_in1, group_w, 1, k_block);
    cb_wait_front(cb_in0, Kt);
    for (uint32_t g = 0; g < num_groups; ++g) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < Kt; kb += k_block) {
            cb_wait_front(cb_in1, block_tiles);
            uint32_t in1_index = 0;
            for (uint32_t k = 0; k < k_block; ++k) {
                ckernel::mm_block_fid<mm_fid>(cb_in0, cb_in1, kb + k, in1_index, 0, group_w, 1, k_block);
                in1_index += group_w;
            }
            cb_pop_front(cb_in1, block_tiles);
        }
        tile_regs_commit();

        cb_reserve_back(cb_out, group_w);
        tile_regs_wait();
        for (uint32_t j = 0; j < group_w; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, group_w);
    }
    cb_pop_front(cb_in0, Kt);
}
