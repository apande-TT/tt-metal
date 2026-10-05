// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt/cpp_row_tile.py: one row-major float32 row `[..., 1, W]` (one page) as a TILE `[1, 1, 1, W]`: output tile j
// takes columns j*32 .. j*32+31 of the row as its row 0 (face 0 / face 1, 64 bytes each), every other row zero.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t j0 = get_arg_val<uint32_t>(2);
    const uint32_t nj = get_arg_val<uint32_t>(3);

    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto ay = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb = 0;
    constexpr uint32_t bytes = 4096;  // float32 tile
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t j = j0; j < j0 + nj; ++j) {
        cb_reserve_back(cb, 1);
        const uint32_t l1 = get_write_ptr(cb);
        for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        noc_async_read(sx.get_noc_addr(0, j * 128), l1, 64);              // columns j*32 .. +15 -> face 0 row 0
        noc_async_read(sx.get_noc_addr(0, j * 128 + 64), l1 + 1024, 64);  // columns +16 .. +31 -> face 1 row 0
        noc_async_read_barrier();
        noc_async_write_page(j, sy, l1);
        noc_async_writes_flushed();
        cb_push_back(cb, 1);
        cb_wait_front(cb, 1);
        cb_pop_front(cb, 1);
    }
    noc_async_write_barrier();
}
