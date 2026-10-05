// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt/cpp_fold_rows.py: `[B, 1, 1, D]` (each sample's one row padded out to its own tile row) as `[1, 1, B, D]`
// (the B rows in one tile row). Output column tile j of this core's range gathers row 0 of input tile
// (b, j) into its row b -- two face segments a row -- rows past B zero. Pure data movement.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

inline uint32_t seg(uint32_t r, uint32_t s, uint32_t row_bytes) {
    return (((r >> 4) * 2 + s) * 16 * row_bytes) + (r & 15) * row_bytes;
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t j0 = get_arg_val<uint32_t>(2);
    const uint32_t nj = get_arg_val<uint32_t>(3);

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t DT = get_compile_time_arg_val(1);
    constexpr uint32_t ROWB = get_compile_time_arg_val(2);  // bytes of 16 datums (a face row)
    constexpr auto ax = TensorAccessorArgs<3>();
    constexpr auto ay = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb = 0;
    const uint32_t bytes = get_tile_size(cb);
    for (uint32_t j = j0; j < j0 + nj; ++j) {
        cb_reserve_back(cb, 1);
        const uint32_t l1 = get_write_ptr(cb);
        if constexpr (B < 32) {
            const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
        }
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t s = 0; s < 2; ++s) {
                noc_async_read(sx.get_noc_addr(b * DT + j, seg(0, s, ROWB)), l1 + seg(b, s, ROWB), ROWB);
            }
        }
        noc_async_read_barrier();
        noc_async_write_page(j, sy, l1);
        noc_async_writes_flushed();
        cb_push_back(cb, 1);
        cb_wait_front(cb, 1);
        cb_pop_front(cb, 1);
    }
    noc_async_write_barrier();
}
