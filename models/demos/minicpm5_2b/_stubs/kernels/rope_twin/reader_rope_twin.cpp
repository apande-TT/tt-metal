// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode rope (C++ twin of _stubs/ttl_rope.py's reader): tile c = this core's row-major index in the
// grid_x-wide grid; reads x[c], its rotate-half partner x[rot], cos[dt], sin[dt] (dt = c % Dt) -- all four under one
// barrier (the ttl reader waits on each in turn).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(3);
    constexpr uint32_t grid_x = get_compile_time_arg_val(4);
    constexpr uint32_t Dt = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto cs = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto sn = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(2), tile_bytes);

    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t dt = c % Dt;
    const uint32_t rot = c - dt + (dt + Dt / 2) % Dt;

    cb_reserve_back(cb_x, 1);
    cb_reserve_back(cb_xr, 1);
    cb_reserve_back(cb_cos, 1);
    cb_reserve_back(cb_sin, 1);
    noc_async_read_tile(c, x, get_write_ptr(cb_x));
    noc_async_read_tile(rot, x, get_write_ptr(cb_xr));
    noc_async_read_tile(dt, cs, get_write_ptr(cb_cos));
    noc_async_read_tile(dt, sn, get_write_ptr(cb_sin));
    noc_async_read_barrier();
    cb_push_back(cb_sin, 1);
    cb_push_back(cb_cos, 1);
    cb_push_back(cb_xr, 1);
    cb_push_back(cb_x, 1);
}
