// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The transformer blocks' float32 LayerNorm (no affine) + modulation, ((x - mu) * rsqrt(var + eps)) * (1 + scale)
// + shift, one tile row (Wt tiles) at a time, bit-identical to the stock chain it replaces
// (_stubs/encoder_stack.py _TtBlock._ln / _modulate; every operand float32, so every binary_ng op is SFPU):
//   mu  = mean(x, -1)          accurate float32 reduce W: copy + add_binary_tile fold over the Wt tiles in order,
//                              sfpu_reduce<SUM, REDUCE_ROW>, then the 1 / W post-mul (as rms_scale.cpp)
//   xc  = x - mu               unary_bcast<COL> + sub_binary_tile<NearestEven>
//   var = mean(xc * xc, -1)    mul_binary_tile, then the same reduce
//   r   = rsqrt(var + eps)     scalar tile + add_binary_tile<NearestEven>, rsqrt_tile<Default>
//   y   = ((xc * r) * s1) + sh unary_bcast<COL> / <ROW> + mul_binary_tile, mul_binary_tile, add_binary_tile<NE>
// xc is recomputed per pass from x and mu (the same subtraction, the same value) instead of stored; mu and r are
// one tile each per row (column 0 valid, as the stock reduce's output). s1 = 1 + scale and sh arrive per tile
// (row 0 valid). With `split` (compile-time arg 3) the output is y's bf16 limbs, made as kernels/split_bf16.cpp
// makes them (hi = the packer's bf16 narrowing of y, lo = bf16(y - float32(hi))), instead of y. Each x row arrives once and stays in cb_x (Wt tiles, two rows deep) for the three passes.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_mu = 1;
constexpr uint32_t cb_r = 2;
constexpr uint32_t cb_ss = 3;  // (1 + scale, shift) per apply tile
constexpr uint32_t cb_out = 16;  // y (float32), or with `split` its bf16 hi limb
constexpr uint32_t cb_lo = 17;   // split: bf16(y - float32(hi))
constexpr uint32_t cb_tmp = 24;  // split: scratch hi (bf16)
constexpr uint32_t cb_y = 25;    // split: scratch y (float32)

template <bool centered>
inline void row_mean_of(CircularBuffer& x, uint32_t inv_w_bits) {
    // DST 0 accumulates, DST 2 takes each later tile, DST 3 holds mu (centered pass)
    if constexpr (centered) {
        unary_bcast_init<ckernel::BroadcastType::COL>(cb_mu);
        unary_bcast<ckernel::BroadcastType::COL>(cb_mu, 0, 3);
    }
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    reconfig_data_format_srca(cb_x);
    for (uint32_t wt = 0; wt < Wt; ++wt) {
        const uint32_t d = wt == 0 ? 0 : 2;
        x.wait_front(wt + 1);  // the row stays in cb_x (popped after the apply pass)
        copy_init(cb_x);
        copy_tile(cb_x, wt, d);
        if constexpr (centered) {
            sub_binary_tile_init();
            sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(d, 3, d);
            mul_binary_tile_init();
            mul_binary_tile(d, d, d);
        }
        if (wt > 0) {
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
        }
    }
    sfpu_reduce_init<ckernel::PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<ckernel::PoolType::SUM, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_w_bits);
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(1);  // float32 bits of 1 / W
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);    // float32 bits of eps
    constexpr bool split = get_compile_time_arg_val(3) != 0;      // write y's bf16 limbs (hi, lo) instead of y

    CircularBuffer x(cb_x);
    CircularBuffer mu(cb_mu);
    CircularBuffer r(cb_r);
    CircularBuffer ss(cb_ss);
    CircularBuffer out(cb_out);
    CircularBuffer lo(cb_lo);
    CircularBuffer tmp(cb_tmp);
    CircularBuffer yb(cb_y);

    compute_kernel_hw_startup(cb_x, cb_out);
    for (uint32_t row = 0; row < num_rows; ++row) {
        // mu
        tile_regs_acquire();
        row_mean_of<false>(x, inv_w_bits);
        tile_regs_commit();
        mu.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_mu);
        pack_tile(0, cb_mu);
        tile_regs_release();
        mu.push_back(1);
        mu.wait_front(1);

        // r = rsqrt(mean((x - mu)^2) + eps)
        tile_regs_acquire();
        row_mean_of<true>(x, inv_w_bits);
        fill_tile_init();
        fill_tile_bitcast(1, eps_bits);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        rsqrt_tile_init();
        rsqrt_tile<RsqrtMode::Default>(0);
        tile_regs_commit();
        r.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_r);
        pack_tile(0, cb_r);
        tile_regs_release();
        r.push_back(1);
        r.wait_front(1);

        // y = ((x - mu) * r) * (1 + scale) + shift, two tiles per DEST acquire (DST 0 / 1 and 2 / 3): the same ops per
        // tile, each init and broadcast setup done once per pair
        for (uint32_t wt = 0; wt < Wt; wt += 2) {
            ss.wait_front(4);  // (1 + scale, shift) of columns wt and wt + 1
            tile_regs_acquire();
            reconfig_data_format_srca(cb_x);
            copy_init(cb_x);
            copy_tile(cb_x, wt, 0);
            copy_tile(cb_x, wt + 1, 2);
            unary_bcast_init<ckernel::BroadcastType::COL>(cb_mu);
            unary_bcast<ckernel::BroadcastType::COL>(cb_mu, 0, 1);
            unary_bcast<ckernel::BroadcastType::COL>(cb_mu, 0, 3);
            sub_binary_tile_init();
            sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(2, 3, 2);
            unary_bcast_init<ckernel::BroadcastType::COL>(cb_r);
            unary_bcast<ckernel::BroadcastType::COL>(cb_r, 0, 1);
            unary_bcast<ckernel::BroadcastType::COL>(cb_r, 0, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(2, 3, 2);
            unary_bcast_init<ckernel::BroadcastType::ROW>(cb_ss);
            unary_bcast<ckernel::BroadcastType::ROW>(cb_ss, 0, 1);
            unary_bcast<ckernel::BroadcastType::ROW>(cb_ss, 2, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(2, 3, 2);
            unary_bcast_init<ckernel::BroadcastType::ROW>(cb_ss);
            unary_bcast<ckernel::BroadcastType::ROW>(cb_ss, 1, 1);
            unary_bcast<ckernel::BroadcastType::ROW>(cb_ss, 3, 3);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(2, 3, 2);
            tile_regs_commit();
            out.reserve_back(2);
            if constexpr (!split) {
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                pack_tile(0, cb_out);
                pack_tile(2, cb_out);
                tile_regs_release();
                out.push_back(2);
            } else {
                // as split_bf16.cpp: hi = bf16(y) (packer narrowing), lo = bf16(y - float32(hi))
                tmp.reserve_back(2);
                yb.reserve_back(2);
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                pack_tile(0, cb_out);
                pack_tile(2, cb_out);
                pack_tile(0, cb_tmp);
                pack_tile(2, cb_tmp);
                pack_reconfig_data_format(cb_y);
                pack_tile(0, cb_y);
                pack_tile(2, cb_y);
                tile_regs_release();
                out.push_back(2);
                tmp.push_back(2);
                yb.push_back(2);

                tmp.wait_front(2);
                yb.wait_front(2);
                lo.reserve_back(2);
                tile_regs_acquire();
                reconfig_data_format_srca(cb_y);
                copy_init(cb_y);
                copy_tile(cb_y, 0, 0);
                copy_tile(cb_y, 1, 2);
                reconfig_data_format_srca(cb_tmp);
                copy_init(cb_tmp);
                copy_tile(cb_tmp, 0, 1);
                copy_tile(cb_tmp, 1, 3);
                sub_binary_tile_init();
                sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
                sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(2, 3, 2);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_lo);
                pack_tile(0, cb_lo);
                pack_tile(2, cb_lo);
                tile_regs_release();
                lo.push_back(2);
                tmp.pop_front(2);
                yb.pop_front(2);
            }
            ss.pop_front(4);
        }
        x.pop_front(Wt);
        mu.pop_front(1);
        r.pop_front(1);
    }
}
