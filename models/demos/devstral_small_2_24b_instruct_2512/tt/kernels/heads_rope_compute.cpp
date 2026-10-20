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
                mul_binary_tile(4 * q, 4 * q + 1, 4 * q);
                mul_binary_tile(4 * q + 2, 4 * q + 3, 4 * q + 2);
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
