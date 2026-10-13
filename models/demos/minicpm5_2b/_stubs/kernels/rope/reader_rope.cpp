// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B rope reader. Input x [B, H, St * 32, D]; cos / signed-sin [B, 1, St * 32, D] (decode: St = 1,
// broadcast over the tile rows; prefill: one row per position). For output tile (b, h, st, dt) it pushes
// x(b, h, st, dt), cos(b, st, dt), x(b, h, st, perm(dt)), sin_signed(b, st, dt) where perm rotates the D
// column tiles by half (rotate_half: tile dt takes tile (dt + Dt/2) % Dt; its sign is folded into sin_signed).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t Dt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(3);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(4);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto cs_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto cos = TensorAccessor(cs_args, get_common_arg_val<uint32_t>(1));
    const auto sin = TensorAccessor(cs_args, get_common_arg_val<uint32_t>(2));

    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        const uint32_t dt = t % Dt;
        const uint32_t row = t / Dt;  // (b, h, st) tile row
        const uint32_t st = row % St;
        const uint32_t b = row / (St * H);
        const uint32_t cs_page = (b * St + st) * Dt + dt;
        const uint32_t rot = (dt + Dt / 2) % Dt;
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_cos, 1);
        cb_reserve_back(cb_xr, 1);
        cb_reserve_back(cb_sin, 1);
        noc_async_read_page(t, x, get_write_ptr(cb_x));
        noc_async_read_page(cs_page, cos, get_write_ptr(cb_cos));
        noc_async_read_page(row * Dt + rot, x, get_write_ptr(cb_xr));
        noc_async_read_page(cs_page, sin, get_write_ptr(cb_sin));
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
        cb_push_back(cb_cos, 1);
        cb_push_back(cb_xr, 1);
        cb_push_back(cb_sin, 1);
    }
}
