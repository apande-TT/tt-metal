// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ column-broadcast multiply (see tt/cpp_bcast_mul.py). Per tile it runs
// binary_ng's float32 SFPU multiply (eltwise_binary_sfpu with a reader-filled column-broadcast b): both
// operands unpacked straight to DEST (UnpackToDestFp32), mul_binary_tile, packed to the output's format.
// Same operands, same SFPU function, same packer: the stock bits. Up to G a tiles of one row share one copy
// of their b tile in DEST (slot G); every product is still mul_binary_tile on the same two values.
// A bf16 output (TC) also takes binary_ng's post-activation TYPECAST float32 -> bf16 in DEST before the pack
// (binary_op_utils::is_typecast): the SFPU rounds to nearest-even there, where the packer alone would round
// ties away from zero. A bf8_b output has no such step in binary_ng and is narrowed by the packer.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t nt = get_arg_val<uint32_t>(1);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t CH = get_compile_time_arg_val(1);
    constexpr uint32_t NBR = get_compile_time_arg_val(2);
    constexpr bool TC = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t F32 = 0u, F16B = 5u;  // DataFormat::Float32, DataFormat::Float16_b
    constexpr uint32_t G = 3;  // a tiles a DEST acquire: half-sync float32 DEST holds 4, one is b's

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_a, cb_y);
    copy_init(cb_a);
    mul_binary_tile_init();

    for (uint32_t c0 = 0; c0 < nt; c0 += CH) {
        const uint32_t n = (nt - c0 < CH) ? (nt - c0) : CH;
        const uint32_t g0 = t0 + c0;
        const uint32_t r0 = g0 / WT;
        cb_wait_front(cb_a, CH);
        cb_reserve_back(cb_y, CH);
        uint32_t j = 0;
        while (j < n) {
            const uint32_t rb = (g0 + j) / WT - r0;
            // this group: up to G tiles, all in row rb
            const uint32_t row_end = (r0 + rb + 1) * WT - g0;
            uint32_t m = n - j;
            if (m > G) {
                m = G;
            }
            if (m > row_end - j) {
                m = row_end - j;
            }
            cb_wait_front(cb_b, rb + 1);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_a, cb_b);
            copy_init(cb_b);
            copy_tile(cb_b, rb, G);
            reconfig_data_format_srca(cb_b, cb_a);
            copy_init(cb_a);
            for (uint32_t i = 0; i < m; ++i) {
                copy_tile(cb_a, j + i, i);
            }
            // Each slot's product, then its narrowing: elementwise on its own slot, so the group's products
            // run before its typecasts with one init each.
            if constexpr (TC) {
                mul_binary_tile_init();
            }
            for (uint32_t i = 0; i < m; ++i) {
                mul_binary_tile(i, G, i);
            }
            if constexpr (TC) {
                typecast_tile_init<F32, F16B>();
                for (uint32_t i = 0; i < m; ++i) {
                    typecast_tile<F32, F16B>(i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < m; ++i) {
                pack_tile(i, cb_y);
            }
            tile_regs_release();
            j += m;
        }
        cb_push_back(cb_y, CH);
        cb_pop_front(cb_a, CH);
        cb_wait_front(cb_b, NBR);
        cb_pop_front(cb_b, NBR);
    }
}
