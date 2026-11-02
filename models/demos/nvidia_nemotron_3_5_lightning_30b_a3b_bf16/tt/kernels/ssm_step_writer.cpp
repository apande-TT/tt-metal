// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Mamba2 decode step, writer (layout in ssm_step_reader.cpp): each new state tile back in place, then this core's
// partial y tiles into row slab r of y_part (R, 32, H P): tile r * (G * 16) + g * 16 + c.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t CPG = get_compile_time_arg_val(2);  // column tiles per group
    constexpr uint32_t YT = get_compile_time_arg_val(3);   // column tiles of y (G CPG)
    constexpr auto a_s = TensorAccessorArgs<4>();
    constexpr auto a_y = TensorAccessorArgs<a_s.next_compile_time_args_offset()>();
    const auto S = TensorAccessor(a_s, get_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(a_y, get_arg_val<uint32_t>(1));
    const uint32_t g = get_arg_val<uint32_t>(2), c0 = get_arg_val<uint32_t>(3);
    const uint32_t n0 = get_arg_val<uint32_t>(4), n1 = get_arg_val<uint32_t>(5), r = get_arg_val<uint32_t>(6);
    constexpr uint32_t TILE_BYTES = 4096;
    constexpr uint32_t cb_so = 16, cb_y = 17;

    for (uint32_t n = n0; n < n1; ++n) {
        cb_wait_front(cb_so, NC);
        const uint32_t l1 = get_read_ptr(cb_so), row = g * N + n;
        for (uint32_t i = 0; i < NC; ++i) {
            noc_async_write(l1 + i * TILE_BYTES, S.get_noc_addr(row * CPG + c0 + i), TILE_BYTES);
        }
        noc_async_writes_flushed();  // the tiles have left L1: the buffer may be refilled
        cb_pop_front(cb_so, NC);
    }
    cb_wait_front(cb_y, NC);
    const uint32_t l1 = get_read_ptr(cb_y);
    for (uint32_t i = 0; i < NC; ++i) {
        noc_async_write(l1 + i * TILE_BYTES, y.get_noc_addr(r * YT + g * CPG + c0 + i), TILE_BYTES);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_y, NC);
}
