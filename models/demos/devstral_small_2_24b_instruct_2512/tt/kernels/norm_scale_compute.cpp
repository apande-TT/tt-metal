// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/norm_scale.py: y = bf16(x * r) for this core's height shard of the fp32 residual x (ROWS tile rows x
// WT tiles) and the fp32 per-row factor r (one tile a tile row, the factor in column 0) -- the calls ttnn.multiply's
// SFPU column-broadcast program (binary_ng eltwise_binary_sfpu_col_bcast.cpp, MUL + its fp32 -> bf16 TYPECAST post
// activation, fp32 DEST, x / r / the broadcast tile unpacked straight to DEST) makes for every tile:
//   r's tile column-broadcast into DEST (unary_bcast<COL>, unpack-to-dest for fp32) and packed to cb_rb;
//   x's tile and cb_rb's tile copied into DEST, mul_binary_tile, typecast_tile<Float32, Float16_b>, pack to bf16.
// The stock loop handles ONE x tile a DEST acquire (copying the broadcast tile back in, reconfiguring srcA twice and
// re-running both inits for every tile); here CHUNK x tiles share an acquire, with the broadcast tile copied once into
// the slot after them. Same unpacks, same SFPU arithmetic, same pack: bit-identical.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#ifdef NORM_CHAIN
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#endif
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#if defined(FUSED_RNE) && defined(TRISC_MATH)
namespace ckernel::sfpu {
// mul_binary_tile then typecast_tile<Float32, Float16_b> in ONE pass over the rows (FUSED_RNE, cpp try 2): the
// product in0 * in1 (calculate_sfpu_binary_mul's fp32-DEST path) stays in a register and is rounded there exactly as
// calculate_typecast_fp32_to_fp16b rounds it after reloading it -- bits + 0x7FFF + bit 16, low half cleared, stored
// as fp32 -- instead of a store, a reload and a second pass. The constants live in LREGs across the rows.
template <int ITERATIONS>
inline void ns_mul_rne(const uint32_t in0, const uint32_t in1, const uint32_t out) {
    constexpr uint32_t T = 32;  // DEST rows a tile in sfpi addressing
    const sfpi::vUInt c_one = 1;
    const sfpi::vUInt c_half = 0x7fff;
    const sfpi::vUInt c_hi = 0xffff0000;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[in0 * T];
        sfpi::vFloat b = sfpi::dst_reg[in1 * T];
        sfpi::vFloat p = a * b;
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(p);
        sfpi::vUInt lsb = (u >> 16) & c_one;
        u = u + c_half;
        u = u + lsb;
        u = u & c_hi;
        sfpi::dst_reg[out * T] = sfpi::as<sfpi::vFloat>(u);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(0);   // tile rows of this core's shard
    constexpr uint32_t WT = get_compile_time_arg_val(1);     // tiles across a row
    constexpr uint32_t CHUNK = get_compile_time_arg_val(2);  // x tiles a DEST acquire (+ the broadcast tile)
#ifdef NORM_CHAIN
    constexpr uint32_t MUL_BITS = get_compile_time_arg_val(3);  // fp32 bits of 1 / N
    constexpr uint32_t ADD_BITS = get_compile_time_arg_val(4);  // fp32 bits of eps
#endif

    constexpr uint32_t cb_x = 0, cb_r = 1, cb_out = 2, cb_rb = 6;
    constexpr uint32_t dst_b = CHUNK;  // the broadcast factor's DEST slot

    compute_kernel_hw_startup(cb_x, cb_out);
    copy_init(cb_x);
    cb_wait_front(cb_x, ROWS * WT);
    cb_wait_front(cb_r, ROWS);
#ifdef NORM_CHAIN
    // cb_r holds the row SUMS: each core's ROWS factor tiles rsqrt(sum * (1/N) + eps) first, by the SFPU functions,
    // fp32 DEST and approx-off config of the ttnn.unary_chain (MUL_UNARY_SFPU, ADD_UNARY_SFPU, RSQRT) they replace,
    // on faces 0 and 2 only (VectorMode::C: column 0, all the broadcast reads) -- same per-element arithmetic
    constexpr uint32_t cb_f = 7;
    reconfig_data_format_srca(cb_r);
    copy_init(cb_r);
    pack_reconfig_data_format(cb_f);
    cb_reserve_back(cb_f, ROWS);
    for (uint32_t t = 0; t < ROWS; ++t) {
        tile_regs_acquire();
        copy_tile(cb_r, t, 0);
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE), 0,
            VectorMode::C, MUL_BITS));
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE), 0,
            VectorMode::C, ADD_BITS));
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8, DST_ACCUM_MODE, false), 0, VectorMode::C));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_f);
        tile_regs_release();
    }
    cb_push_back(cb_f, ROWS);
    cb_wait_front(cb_f, ROWS);
    pack_reconfig_data_format(cb_out);
    constexpr uint32_t cb_bc = cb_f;  // the factors the rows broadcast
#else
    constexpr uint32_t cb_bc = cb_r;
#endif

    for (uint32_t t = 0; t < ROWS; ++t) {
        // the row's factor, column-broadcast (as the stock kernel's process_tile prologue)
        pack_reconfig_data_format(cb_out, cb_rb);
        reconfig_data_format(cb_bc, cb_bc);
        unary_bcast_init<BroadcastType::COL>(cb_bc);
        cb_reserve_back(cb_rb, 1);
        tile_regs_acquire();
        unary_bcast<BroadcastType::COL>(cb_bc, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rb);
        cb_push_back(cb_rb, 1);
        tile_regs_release();
        pack_reconfig_data_format(cb_rb, cb_out);
        cb_wait_front(cb_rb, 1);

        for (uint32_t w0 = 0; w0 < WT; w0 += CHUNK) {
            const uint32_t n = WT - w0 < CHUNK ? WT - w0 : CHUNK;
            cb_reserve_back(cb_out, n);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_rb, cb_x);
            copy_init(cb_x);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_x, t * WT + w0 + i, i);
            }
            reconfig_data_format_srca(cb_x, cb_rb);
            copy_init(cb_rb);
            copy_tile(cb_rb, 0, dst_b);
            mul_binary_tile_init();
#ifdef FUSED_RNE
            for (uint32_t i = 0; i < n; ++i) {
                MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, ns_mul_rne, (8), i, dst_b, i, VectorMode::RC)));
            }
#else
            for (uint32_t i = 0; i < n; ++i) {
                mul_binary_tile(i, dst_b, i);
            }
            typecast_tile_init<0u, 5u>();
            for (uint32_t i = 0; i < n; ++i) {
                typecast_tile<0u, 5u>(i);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, n);
        }
        cb_pop_front(cb_rb, 1);
    }
#ifdef NORM_CHAIN
    cb_pop_front(cb_f, ROWS);
#endif
    cb_pop_front(cb_r, ROWS);
    cb_pop_front(cb_x, ROWS * WT);
}
