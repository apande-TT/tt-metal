// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ decode RoPE (see tt/cpp_rope_dec.py). Reads the DT column tiles of the full cos
// and signed-sin tile rows once (every row the same: cpp_rope_dec.full_rows), then this core's tile rows of x,
// DT tiles at a time.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t row0 = get_arg_val<uint32_t>(3);
    const uint32_t nrows = get_arg_val<uint32_t>(4);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr auto ax = TensorAccessorArgs<1>();
    constexpr auto ac = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    constexpr auto as = TensorAccessorArgs<ac.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sc = TensorAccessor(ac, cos_addr);
    const auto ss = TensorAccessor(as, sin_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_cos = 1;
    constexpr uint32_t cb_sin = 2;
    const uint32_t bytes = get_tile_size(cb_x);

    cb_reserve_back(cb_cos, DT);
    cb_reserve_back(cb_sin, DT);
    const uint32_t lc = get_write_ptr(cb_cos);
    const uint32_t ls = get_write_ptr(cb_sin);
    for (uint32_t c = 0; c < DT; ++c) {
        noc_async_read_page(c, sc, lc + c * bytes);
        noc_async_read_page(c, ss, ls + c * bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_cos, DT);
    cb_push_back(cb_sin, DT);

    for (uint32_t r = row0; r < row0 + nrows; ++r) {
        cb_reserve_back(cb_x, DT);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t c = 0; c < DT; ++c) {
            noc_async_read_page(r * DT + c, sx, l1);
            l1 += bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, DT);
    }
}
