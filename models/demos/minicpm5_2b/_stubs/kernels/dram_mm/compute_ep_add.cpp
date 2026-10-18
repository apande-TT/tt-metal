// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode projection epilogue compute: out = lhs + rhs for bf16 tiles, the calls of ttnn.add's
// binary_ng FPU kernel (eltwise_binary_no_bcast.cpp, no activations) under its compute config (16-bit DEST,
// HiFi4, no approximation), so the sum is the stand-alone add's.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t cb_lhs = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rhs = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t per_cycle = 8;  // binary_ng's tiles per cycle under 16-bit DEST
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_lhs, cb_rhs, cb_out);
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_lhs, cb_rhs);
    cb_wait_front(cb_lhs, n_tiles);
    cb_wait_front(cb_rhs, n_tiles);
    for (uint32_t t0 = 0; t0 < n_tiles; t0 += per_cycle) {
        const uint32_t n = (n_tiles - t0) < per_cycle ? (n_tiles - t0) : per_cycle;
        cb_reserve_back(cb_out, n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            add_tiles(cb_lhs, cb_rhs, t0 + i, t0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, n);
    }
    cb_pop_front(cb_lhs, n_tiles);
    cb_pop_front(cb_rhs, n_tiles);
}
