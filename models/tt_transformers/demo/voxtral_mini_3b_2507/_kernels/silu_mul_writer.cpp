// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the SwiGLU product kernel: drain the output circular buffer to this core's run of the
// product tensor, BLOCK tiles at a time.  See silu_mul_reader.cpp for why the stream is flat.

#include <cstdint>

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_y = tt::CBIndex::c_16;
    constexpr uint32_t block = get_compile_time_arg_val(0);

    constexpr auto y_args = TensorAccessorArgs<1>();
    const auto y = TensorAccessor(y_args, y_addr);

    const uint32_t tile_bytes = get_tile_size(cb_y);

    for (uint32_t i = 0; i < n_tiles; i += block) {
        cb_wait_front(cb_y, block);
        uint32_t r = get_read_ptr(cb_y);
        for (uint32_t j = 0; j < block; ++j) {
            noc_async_write_page(start + i + j, y, r);
            r += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, block);
    }
}
