// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Mamba2 decode, after the recurrence, one tile of the inner dimension per core (fp32, inputs unpacked straight to
// DST): cb 0 = the R row-range partials of y = C S (ssm_step_*.cpp); cb 1 = x, D (rows replicated), the gate
//   ->  (sum of the partials + D x) silu(gate)
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_p = 0, cb_k = 1, cb_out = 16;

ALWI void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t R = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(cb_p, cb_out);
    cb_wait_front(cb_p, R);
    cb_wait_front(cb_k, 3);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    load(cb_p, 0, 0);
    for (uint32_t r = 1; r < R; ++r) {
        load(cb_p, r, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    load(cb_k, 0, 1);
    load(cb_k, 1, 2);
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    load(cb_k, 2, 1);
    silu_tile_init();
    silu_tile(1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_p, R);
    cb_pop_front(cb_k, 3);
}
