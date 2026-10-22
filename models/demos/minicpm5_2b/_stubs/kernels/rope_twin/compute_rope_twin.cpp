// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode rope (C++ twin of _stubs/ttl_rope.py's compute, the same calls tt-lang generates for
// y = x * cos + xr * sin): FPU mul_tiles into DEST 0 and 1, SFPU add_binary_tile, one pack (16-bit DEST, HiFi4).

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(3);
    constexpr uint32_t cb_y = get_compile_time_arg_val(4);

    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_xr, 1);
    cb_wait_front(cb_cos, 1);
    cb_wait_front(cb_sin, 1);
    cb_reserve_back(cb_y, 1);
    binary_op_init_common(cb_x, cb_cos, cb_y);
    tile_regs_acquire();
    mul_tiles_init(cb_x, cb_cos);
    mul_tiles(cb_x, cb_cos, 0, 0, 0);
    mul_tiles_init(cb_xr, cb_sin);
    mul_tiles(cb_xr, cb_sin, 0, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_y, 0);
    tile_regs_release();
    cb_push_back(cb_y, 1);
    cb_pop_front(cb_sin, 1);
    cb_pop_front(cb_cos, 1);
    cb_pop_front(cb_xr, 1);
    cb_pop_front(cb_x, 1);
}
