// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode q/k prep (compute), one q / k head of a 32-user tile row (W tiles, one user per row):
//   n = x * rsqrt(mean(x^2) + eps) * w        per row over the head's D columns (fp32 dest), rounded to bf16
// and on RoPE layers rotate-half RoPE as tile pairs (the head's first half pairs with its second half, t <-> t + W/2):
//   out_t = n_t * cos_t - n_{t+W/2} * sin_t   (t < W/2),    out_t = n_t * cos_t + n_{t-W/2} * sin_t   (t >= W/2)
// with each product rounded to bf16 before the add (as the rotate-half matmul, two multiplies and an add gave).
// A v unit (kind 1) has no compute.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t ROPE = get_compile_time_arg_val(1);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);  // fp32 bits (EPS is an LLK macro)
    const uint32_t kind = get_arg_val<uint32_t>(0);
    if (kind == 1) {
        return;
    }
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_cos = 2, cb_sin = 3, cb_scaler = 4;
    constexpr uint32_t cb_sq = 5, cb_rinv = 6, cb_xr = 7, cb_n = 8, cb_a = 9, cb_b = 10, cb_out = 16;
    constexpr uint32_t H = W / 2;

    compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    cb_wait_front(cb_x, W);
    cb_wait_front(cb_w, W);
    cb_wait_front(cb_scaler, 1);

    // x^2 (fp32)
    reconfig_data_format(cb_x, cb_x);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_x, cb_x);
    tile_regs_acquire();
    for (uint32_t t = 0; t < W; ++t) {
        mul_tiles(cb_x, cb_x, t, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_sq, W);
    for (uint32_t t = 0; t < W; ++t) {
        pack_tile(t, cb_sq);
    }
    cb_push_back(cb_sq, W);
    tile_regs_release();

    // rinv = rsqrt(mean over the head's columns + eps), in column 0
    cb_wait_front(cb_sq, W);
    reconfig_data_format(cb_scaler, cb_sq);
    pack_reconfig_data_format(cb_rinv);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_rinv);
    tile_regs_acquire();
    for (uint32_t t = 0; t < W; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, t, 0, 0);
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_rinv, 1);
    pack_tile(0, cb_rinv);
    cb_push_back(cb_rinv, 1);
    tile_regs_release();
    reduce_uninit();
    cb_pop_front(cb_sq, W);

    // x * rinv (fp32)
    cb_wait_front(cb_rinv, 1);
    reconfig_data_format(cb_x, cb_rinv);
    pack_reconfig_data_format(cb_xr);
    mul_bcast_cols_init(cb_x, cb_rinv);
    tile_regs_acquire();
    for (uint32_t t = 0; t < W; ++t) {
        mul_tiles_bcast_cols(cb_x, cb_rinv, t, 0, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_xr, W);
    for (uint32_t t = 0; t < W; ++t) {
        pack_tile(t, cb_xr);
    }
    cb_push_back(cb_xr, W);
    tile_regs_release();
    cb_pop_front(cb_rinv, 1);
    cb_pop_front(cb_x, W);

    // n = (x * rinv) * w, bf16 (the output itself on a layer without RoPE)
    constexpr uint32_t cb_norm = ROPE ? cb_n : cb_out;
    cb_wait_front(cb_xr, W);
    reconfig_data_format(cb_xr, cb_w);
    pack_reconfig_data_format(cb_norm);
    mul_init(cb_xr, cb_w);
    tile_regs_acquire();
    for (uint32_t t = 0; t < W; ++t) {
        mul_tiles(cb_xr, cb_w, t, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_norm, W);
    for (uint32_t t = 0; t < W; ++t) {
        pack_tile(t, cb_norm);
    }
    cb_push_back(cb_norm, W);
    tile_regs_release();
    cb_pop_front(cb_xr, W);
    cb_pop_front(cb_w, W);

    if constexpr (ROPE) {
        cb_wait_front(cb_n, W);
        cb_wait_front(cb_cos, W);
        cb_wait_front(cb_sin, W);
        // a = n * cos (bf16)
        reconfig_data_format(cb_n, cb_cos);
        pack_reconfig_data_format(cb_a);
        mul_init(cb_n, cb_cos);
        tile_regs_acquire();
        for (uint32_t t = 0; t < W; ++t) {
            mul_tiles(cb_n, cb_cos, t, t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_a, W);
        for (uint32_t t = 0; t < W; ++t) {
            pack_tile(t, cb_a);
        }
        cb_push_back(cb_a, W);
        tile_regs_release();
        // b = n_partner * sin (bf16)
        reconfig_data_format(cb_n, cb_sin);
        pack_reconfig_data_format(cb_b);
        mul_init(cb_n, cb_sin);
        tile_regs_acquire();
        for (uint32_t t = 0; t < W; ++t) {
            mul_tiles(cb_n, cb_sin, t < H ? t + H : t - H, t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_b, W);
        for (uint32_t t = 0; t < W; ++t) {
            pack_tile(t, cb_b);
        }
        cb_push_back(cb_b, W);
        tile_regs_release();
        cb_pop_front(cb_n, W);
        cb_pop_front(cb_cos, W);
        cb_pop_front(cb_sin, W);
        // out = a -/+ b (bf16)
        cb_wait_front(cb_a, W);
        cb_wait_front(cb_b, W);
        reconfig_data_format(cb_a, cb_b);
        pack_reconfig_data_format(cb_out);
        tile_regs_acquire();
        sub_init(cb_a, cb_b);
        for (uint32_t t = 0; t < H; ++t) {
            sub_tiles(cb_a, cb_b, t, t, t);
        }
        add_init(cb_a, cb_b);
        for (uint32_t t = H; t < W; ++t) {
            add_tiles(cb_a, cb_b, t, t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, W);
        for (uint32_t t = 0; t < W; ++t) {
            pack_tile(t, cb_out);
        }
        cb_push_back(cb_out, W);
        tile_regs_release();
        cb_pop_front(cb_a, W);
        cb_pop_front(cb_b, W);
    }
}
