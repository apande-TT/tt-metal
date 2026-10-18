// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode rope reader on the QKV projection's own layout (one tile row: row b = batch b): output tile c
// of the q / k columns is head c / Dt, column tile dt = c % Dt. Pushes the operands compute_rope.cpp consumes:
// x tile c, the step's cos tile dt (row b = batch b's position), x's rotate-half partner tile (same head,
// (dt + Dt / 2) % Dt) and the signed sin tile dt. Every element sees the same operands as with the head-split
// layout (reader_rope.cpp), so the result is identical.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto cs_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto cos = TensorAccessor(cs_args, get_common_arg_val<uint32_t>(1));
    const auto sin = TensorAccessor(cs_args, get_common_arg_val<uint32_t>(2));

    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        const uint32_t dt = t % Dt;
        const uint32_t rot = (t - dt) + (dt + Dt / 2) % Dt;
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_cos, 1);
        cb_reserve_back(cb_xr, 1);
        cb_reserve_back(cb_sin, 1);
        noc_async_read_page(t, x, get_write_ptr(cb_x));
        noc_async_read_page(dt, cos, get_write_ptr(cb_cos));
        noc_async_read_page(rot, x, get_write_ptr(cb_xr));
        noc_async_read_page(dt, sin, get_write_ptr(cb_sin));
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
        cb_push_back(cb_cos, 1);
        cb_push_back(cb_xr, 1);
        cb_push_back(cb_sin, 1);
    }
}
