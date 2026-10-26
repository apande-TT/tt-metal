// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of tt/heads_rope.py, per (tile row, head) unit, on its q tiles and k tiles x (2 * HT): for every
// output tile j of a head (partner p = (j + HT / 2) % HT)
//     out_j = x_j * cos_j + x_p * signed_sin_j
// = x * cos + rotate_half(x) * sin. Each step replays the op the rotate-half chain ran for bf16 operands: ttnn.multiply's
// accurate SFPU path (both operands copied into a 16-bit DEST, mul_binary_tile -- the product rounded to bf16 nearest-
// even -- packed to a bf16 tile) and ttnn.add's default FPU path (add_tiles of the two bf16 product tiles in a 16-bit
// DEST). (-x_p) * s and x_p * (-s) round alike, so the output matches the chain bit for bit.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#if defined(ROPE_FUSED_MUL) && defined(TRISC_MATH)
#include "ckernel_sfpu_square.h"  // float32_to_bf16_rne_prgm
namespace ckernel::sfpu {
// The constants float32_to_bf16_rne_prgm reads (as calculate_square holds them): its 1, 0x7fff and 0xffff0000.
inline void rope_products_init() {
    sfpi::vConstIntPrgm0 = 1;
    sfpi::vConstIntPrgm1 = 0x7fff;
    sfpi::vConstIntPrgm2 = 0xffff0000;
}

// Both products of a slot in one pass over the rows: x * cos into tile base, x_p * signed_sin into tile base + 2 (x, cos,
// x_p, sin in tiles base .. base + 3). Each is calculate_sfpu_binary_mul's 16-bit-DEST body -- the fp32 product, rounded
// to bf16 nearest-even, zero when either operand is -- with the rounding constants in registers instead of three
// SFPLOADIs a product, so the results are the same bits.
template <int ITERATIONS>
inline void calculate_rope_products(const std::uint32_t base, const std::uint32_t, const std::uint32_t) {
    constexpr std::uint32_t T = 32;  // DEST rows a tile in sfpi addressing
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[base * T];
        sfpi::vFloat in1 = sfpi::dst_reg[(base + 1) * T];
        sfpi::vFloat a = float32_to_bf16_rne_prgm(in0 * in1);
#ifndef ROPE_SIGNED_ZEROS
        v_if(in0 == 0 || in1 == 0) { a = 0.0f; }
        v_endif;
#endif
        sfpi::dst_reg[base * T] = a;
        sfpi::vFloat in2 = sfpi::dst_reg[(base + 2) * T];
        sfpi::vFloat in3 = sfpi::dst_reg[(base + 3) * T];
        sfpi::vFloat b = float32_to_bf16_rne_prgm(in2 * in3);
#ifndef ROPE_SIGNED_ZEROS
        v_if(in2 == 0 || in3 == 0) { b = 0.0f; }
        v_endif;
#endif
        sfpi::dst_reg[(base + 2) * T] = b;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t ST = get_compile_time_arg_val(2);
    constexpr uint32_t N = 2 * HT;  // q and k tiles of one unit
    constexpr uint32_t HALF = HT / 2;
    constexpr uint32_t PER = 2;      // product pairs a DEST acquire (4 slots each)
    constexpr uint32_t SUM_PER = (N % 8 == 0) ? 8 : (N % 4 == 0) ? 4 : 2;  // sums a DEST acquire
    static_assert(N % PER == 0 && N % SUM_PER == 0, "heads_rope blocking");

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_cos = tt::CBIndex::c_1;
    constexpr uint32_t cb_sin = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_ab = tt::CBIndex::c_24;  // the unit's products (a_i, b_i interleaved)

    compute_kernel_hw_startup(cb_x, cb_cos, cb_out);

    uint32_t cur_st = 0xFFFFFFFF;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t st = (u / H) % ST;
        if (st != cur_st) {
            if (cur_st != 0xFFFFFFFF) {
                cb_pop_front(cb_cos, HT);
                cb_pop_front(cb_sin, HT);
            }
            cb_wait_front(cb_cos, HT);
            cb_wait_front(cb_sin, HT);
            cur_st = st;
        }
        cb_wait_front(cb_x, N);

        // the products, as BinaryNg's multiply ops write them: a_i = x_i * cos_j, b_i = x_p * signed_sin_j (bf16)
        cb_reserve_back(cb_ab, 2 * N);
#ifdef HOIST_INITS
        // x, cos and sin are all bf16 tiles of one shape, so one datacopy init serves the three; the SFPU init programs
        // ADDR_MOD_6/7 only (the datacopy uses ADDR_MOD_0/2): both once a unit instead of once a DEST acquire / CB
        copy_init(cb_x);
        mul_binary_tile_init();
#ifdef ROPE_FUSED_MUL
        MATH((ckernel::sfpu::rope_products_init()));
#endif
#endif
        for (uint32_t i0 = 0; i0 < N; i0 += PER) {
            tile_regs_acquire();
#ifndef HOIST_INITS
            copy_init(cb_x);
#endif
            for (uint32_t q = 0; q < PER; ++q) {
                const uint32_t i = i0 + q;
                const uint32_t t = i / HT;
                const uint32_t j = i % HT;
                copy_tile(cb_x, i, 4 * q);
                copy_tile(cb_x, t * HT + (j + HALF) % HT, 4 * q + 2);
            }
#ifndef HOIST_INITS
            copy_init(cb_cos);
#endif
            for (uint32_t q = 0; q < PER; ++q) {
                copy_tile(cb_cos, (i0 + q) % HT, 4 * q + 1);
            }
#ifndef HOIST_INITS
            copy_init(cb_sin);
#endif
            for (uint32_t q = 0; q < PER; ++q) {
                copy_tile(cb_sin, (i0 + q) % HT, 4 * q + 3);
            }
#ifndef HOIST_INITS
            mul_binary_tile_init();
#endif
            for (uint32_t q = 0; q < PER; ++q) {
#ifdef ROPE_FUSED_MUL
                MATH((SFPU_BINARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_rope_products,
                    (8),
                    4 * q,
                    4 * q + 1,
                    4 * q + 2,
                    VectorMode::RC)));
#else
                mul_binary_tile(4 * q, 4 * q + 1, 4 * q);
                mul_binary_tile(4 * q + 2, 4 * q + 3, 4 * q + 2);
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t q = 0; q < PER; ++q) {
                pack_tile(4 * q, cb_ab);      // a_i at 2i
                pack_tile(4 * q + 2, cb_ab);  // b_i at 2i + 1
            }
            tile_regs_release();
        }
        cb_push_back(cb_ab, 2 * N);
        cb_pop_front(cb_x, N);

        // out_i = a_i + b_i, as ttnn.add runs it by default: the FPU add (16-bit DEST) of the two bf16 product tiles
        cb_wait_front(cb_ab, 2 * N);
        cb_reserve_back(cb_out, N);
        for (uint32_t i0 = 0; i0 < N; i0 += SUM_PER) {
            add_init(cb_ab, cb_ab);
            tile_regs_acquire();
            for (uint32_t q = 0; q < SUM_PER; ++q) {
                add_tiles(cb_ab, cb_ab, 2 * (i0 + q), 2 * (i0 + q) + 1, q);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t q = 0; q < SUM_PER; ++q) {
                pack_tile(q, cb_out);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_ab, 2 * N);
        cb_push_back(cb_out, N);
    }
    if (cur_st != 0xFFFFFFFF) {
        cb_pop_front(cb_cos, HT);
        cb_pop_front(cb_sin, HT);
    }
}
