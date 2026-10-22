// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode rope (C++ twin of _stubs/ttl_rope.py's writer): stores tile c of y.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_y = get_compile_time_arg_val(0);
    constexpr uint32_t grid_x = get_compile_time_arg_val(1);
    constexpr auto y_args = TensorAccessorArgs<2>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    const auto y = TensorAccessor(y_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    cb_wait_front(cb_y, 1);
    noc_async_write_tile(c, y, get_read_ptr(cb_y));
    noc_async_write_barrier();
    cb_pop_front(cb_y, 1);
}
